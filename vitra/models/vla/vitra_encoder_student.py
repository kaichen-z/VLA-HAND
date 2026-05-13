from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, BatchFeature

from vitra.models.action_model.diffusion_policy import DiffusionPolicy
from vitra.utils.nn_utils import MLPProjector


class EncoderStudentProcessor:
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

    def __call__(self, text, images, return_tensors="pt", **kwargs):
        text_inputs = self.tokenizer(text, return_tensors=return_tensors, truncation=True, padding=True)
        image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
        return BatchFeature({**text_inputs, **image_inputs})


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
        if pixel_values is not None:
            batch = pixel_values.shape[0]
            pooled = pixel_values.flatten(1).mean(dim=1, keepdim=True).expand(batch, self.config.hidden_size)
        else:
            batch = input_ids.shape[0]
            pooled = input_ids.float().mean(dim=1, keepdim=True).expand(batch, self.config.hidden_size)
        projected = self.proj(pooled)
        return type("Output", (), {"pooler_output": projected, "last_hidden_state": projected.unsqueeze(1)})()


class DummyProcessor:
    def __init__(self, model_max_length: int = 32):
        self.tokenizer = type("Tokenizer", (), {"model_max_length": model_max_length, "pad_token_id": 0})()

    def __call__(self, text, images, return_tensors="pt", **kwargs):
        batch = len(text) if isinstance(text, list) else 1
        input_ids = torch.ones(batch, min(8, self.tokenizer.model_max_length), dtype=torch.long)
        pixel_values = torch.zeros(batch, 3, 224, 224, dtype=torch.float32)
        return BatchFeature(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
                "pixel_values": pixel_values,
            }
        )


class VITRA_EncoderStudent(nn.Module):
    """Compact encoder-only VITRA student.

    The student replaces the generative PaliGemma backbone with compact vision
    and text encoders, then projects their fused representation into the VITRA
    action-conditioning feature size expected by the diffusion action head.
    """

    def __init__(
        self,
        configs,
        train_setup_configs=None,
        act_model_configs=None,
        fwd_pred_next_n=1,
        repeated_diffusion_steps: int = 8,
        use_state="DiT",
        use_fov=True,
        use_bf16=False,
        **kwargs,
    ):
        super().__init__()
        self.configs = configs
        self.student_vlm_configs = configs.get("student_vlm", {})
        self.train_setup_configs = train_setup_configs or {}
        self.act_model_configs = act_model_configs or {}
        self.use_state = use_state
        self.use_fov = use_fov
        self.use_bf16 = use_bf16
        self.repeated_diffusion_steps = repeated_diffusion_steps
        self.past_action_window_size = 0
        self.chunk_size = configs.get("fwd_pred_next_n", 16)
        self.future_action_window_size = self.chunk_size - 1
        self.action_type = configs["train_dataset"].get("action_type", "angle")
        self.hand_dim = 51 if self.action_type == "angle" else 69

        self.output_size = int(
            self.student_vlm_configs.get("student_output_size", self.act_model_configs.get("token_size", 2304))
        )
        self.freeze_vision_encoder = bool(self.student_vlm_configs.get("freeze_vision_encoder", True))

        self.processor, self.vision_encoder, self.text_encoder = self._init_encoders()
        self.vision_hidden_size = self._hidden_size(self.vision_encoder)
        self.text_hidden_size = self._hidden_size(self.text_encoder)

        state_dim = int(configs["state_encoder"]["state_dim"])
        fusion_hidden_size = int(self.student_vlm_configs.get("fusion_hidden_size", 1024))
        fusion_input_size = self.vision_hidden_size + self.text_hidden_size

        self.fov_encoder = MLPProjector(2, fusion_hidden_size) if self.use_fov else None
        if self.use_fov:
            fusion_input_size += fusion_hidden_size

        self.student_state_encoder = None
        if self.student_vlm_configs.get("include_state_in_fusion", True):
            self.student_state_encoder = MLPProjector(2 * state_dim, fusion_hidden_size)
            fusion_input_size += fusion_hidden_size

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.GELU(),
            nn.Linear(fusion_hidden_size, fusion_hidden_size),
            nn.GELU(),
        )
        self.cognition_projection = nn.Linear(fusion_hidden_size, self.output_size)
        self.act_model = self._init_act_model()
        self.trainable_params_setup_for_vlm_distill(
            self.train_setup_configs.get("freeze_option", "encoder_student_joint")
        )

    def _init_encoders(self):
        if self.student_vlm_configs.get("use_dummy", False):
            hidden = int(self.student_vlm_configs.get("dummy_hidden_size", 32))
            return DummyProcessor(), DummyEncoder(hidden), DummyEncoder(hidden)

        vision_name = self.student_vlm_configs["vision_encoder_name"]
        text_name = self.student_vlm_configs["text_encoder_name"]
        image_processor = AutoImageProcessor.from_pretrained(vision_name)
        tokenizer = AutoTokenizer.from_pretrained(text_name)
        processor = EncoderStudentProcessor(image_processor, tokenizer)
        vision_encoder = AutoModel.from_pretrained(vision_name)
        text_encoder = AutoModel.from_pretrained(text_name)
        return processor, vision_encoder, text_encoder

    @staticmethod
    def _hidden_size(model) -> int:
        config = model.config
        for attr in ("hidden_size", "projection_dim", "vision_embed_dim"):
            if hasattr(config, attr):
                return int(getattr(config, attr))
        raise ValueError(f"Could not infer hidden size from config {config}")

    def _init_act_model(self):
        return DiffusionPolicy(
            model_type=self.act_model_configs.get("model_type", "DiT-B-6L"),
            token_size=self.act_model_configs.get("token_size", self.output_size),
            in_channels=self.act_model_configs.get("action_dim", 192),
            future_action_window_size=self.future_action_window_size,
            past_action_window_size=self.past_action_window_size,
            use_state=self.use_state,
            action_type=self.action_type,
            state_dim=self.configs["state_encoder"]["state_dim"] if self.use_state == "DiT" else None,
            loss_type=self.configs.get("loss_type", "human"),
        )

    def trainable_params_setup_for_vlm_distill(self, freeze_option: str = "encoder_student_joint"):
        self.requires_grad_(False)
        if freeze_option in {"encoder_student_default", "encoder_student_cognition"}:
            self._enable_vlm_trainables()
        elif freeze_option == "encoder_student_action_head_only":
            self.act_model.requires_grad_(True)
        elif freeze_option in {"encoder_student_action", "encoder_student_joint"}:
            self._enable_vlm_trainables()
            self.act_model.requires_grad_(True)
        else:
            raise ValueError(f"Unsupported encoder-student freeze option: {freeze_option}")

        if not self.freeze_vision_encoder:
            self.vision_encoder.requires_grad_(True)

    def _enable_vlm_trainables(self) -> None:
        self.text_encoder.requires_grad_(True)
        self.fusion.requires_grad_(True)
        self.cognition_projection.requires_grad_(True)
        if self.fov_encoder is not None:
            self.fov_encoder.requires_grad_(True)
        if self.student_state_encoder is not None:
            self.student_state_encoder.requires_grad_(True)

    def _pool_output(self, outputs, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        hidden = outputs.last_hidden_state
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def extract_cognition_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze_vision_encoder and self.training):
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        parts = [
            self._pool_output(vision_outputs),
            self._pool_output(text_outputs, attention_mask),
        ]
        if self.use_fov and self.fov_encoder is not None:
            parts.append(self.fov_encoder(fov))
        if self.student_state_encoder is not None:
            masked_state = current_state * current_state_mask.to(current_state.dtype)
            state_input = torch.cat([masked_state, current_state_mask.to(current_state.dtype)], dim=-1)
            parts.append(self.student_state_encoder(state_input))

        fused = self.fusion(torch.cat(parts, dim=-1))
        return self.cognition_projection(fused)

    def _forward_act_model(
        self,
        action_features: torch.Tensor,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_masks: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        mode: str = "train",
        repeated_diffusion_steps: int = 1,
        cfg_scale: float = 5.0,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        guidance_fn=None,
        guidance_scale: float = 0.0,
        guidance_start_frac: float = 0.0,
        guidance_end_frac: float = 1.0,
        guidance_grad_clip: float = 1.0,
        return_guidance_trace: bool = False,
        fixed_actions=None,
        fixed_action_mask=None,
        return_replan_trace: bool = False,
    ):
        bsz = action_features.shape[0]
        model_dtype = next(self.act_model.net.parameters()).dtype
        action_features = action_features.to(model_dtype).unsqueeze(1)
        action_features = action_features.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
        action_features = action_features.view(bsz * repeated_diffusion_steps, 1, action_features.shape[-1])

        action_masks_repeated = action_masks.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
        action_masks_repeated = action_masks_repeated.view(
            bsz * repeated_diffusion_steps, action_masks.shape[1], action_masks.shape[2]
        )
        fixed_actions_repeated = None
        fixed_action_mask_repeated = None
        if fixed_actions is not None:
            fixed_actions_repeated = fixed_actions.to(
                device=action_masks.device,
                dtype=action_masks_repeated.dtype,
            )
            fixed_actions_repeated = fixed_actions_repeated.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
            fixed_actions_repeated = fixed_actions_repeated.view(
                bsz * repeated_diffusion_steps,
                fixed_actions.shape[1],
                fixed_actions.shape[2],
            )
        if fixed_action_mask is not None:
            fixed_action_mask_repeated = fixed_action_mask.to(
                device=action_masks.device,
                dtype=action_masks_repeated.dtype,
            )
            fixed_action_mask_repeated = fixed_action_mask_repeated.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
            fixed_action_mask_repeated = fixed_action_mask_repeated.view(
                bsz * repeated_diffusion_steps,
                fixed_action_mask.shape[1],
                fixed_action_mask.shape[2],
            )

        if self.use_state == "DiT":
            current_state_repeated = current_state.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
            current_state_repeated = current_state_repeated.view(
                bsz * repeated_diffusion_steps, 1, current_state.shape[1]
            )
            current_state_mask_repeated = current_state_mask.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
            current_state_mask_repeated = current_state_mask_repeated.view(
                bsz * repeated_diffusion_steps, 1, current_state_mask.shape[1]
            )
        else:
            current_state_repeated = None
            current_state_mask_repeated = None

        if mode == "train":
            actions_repeated = action_labels.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
            actions_repeated = actions_repeated.view(
                bsz * repeated_diffusion_steps, action_labels.shape[1], action_labels.shape[2]
            )
            if self.use_state == "DiT":
                return None, self.act_model.loss(
                    actions_repeated,
                    action_features,
                    action_masks_repeated,
                    current_state_repeated,
                    current_state_mask_repeated,
                )
            return None, self.act_model.loss(actions_repeated, action_features, action_masks_repeated)

        actions = self.act_model.sample(
            action_features,
            cfg_scale,
            current_state_repeated,
            current_state_mask_repeated,
            use_ddim,
            num_ddim_steps,
            action_masks_repeated,
            guidance_fn=guidance_fn,
            guidance_scale=guidance_scale,
            guidance_start_frac=guidance_start_frac,
            guidance_end_frac=guidance_end_frac,
            guidance_grad_clip=guidance_grad_clip,
            return_guidance_trace=return_guidance_trace,
            fixed_actions=fixed_actions_repeated,
            fixed_action_mask=fixed_action_mask_repeated,
            return_replan_trace=return_replan_trace,
        )
        if return_guidance_trace or return_replan_trace:
            actions, trace = actions
            return actions, trace
        return actions, None

    @torch.no_grad()
    def encode_action_condition(
        self,
        image,
        instruction: str,
        current_state,
        current_state_mask,
        fov=None,
        use_cache=False,
    ) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        device = current_state.device
        model_inputs = self.processor(text=instruction, images=[image], return_tensors="pt")
        pixel_values = model_inputs["pixel_values"].to(device)
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
        return self.extract_cognition_features(
            pixel_values,
            input_ids,
            attention_mask=attention_mask,
            current_state_mask=current_state_mask.to(device),
            current_state=current_state.to(device),
            fov=fov.to(device) if fov is not None else None,
        )

    def sample_action_from_condition(
        self,
        action_features: torch.Tensor,
        current_state,
        current_state_mask,
        action_mask_torch=None,
        fov=None,
        sample_times=1,
        use_ddim=True,
        num_ddim_steps=10,
        cfg_scale=5.0,
        guidance_fn=None,
        guidance_scale: float = 0.0,
        guidance_start_frac: float = 0.0,
        guidance_end_frac: float = 1.0,
        guidance_grad_clip: float = 1.0,
        return_guidance_trace: bool = False,
        fixed_actions=None,
        fixed_action_mask=None,
        return_replan_trace: bool = False,
    ) -> np.ndarray:
        if action_mask_torch is None:
            raise ValueError("action_mask_torch is required for cached student diffusion sampling.")
        samples, trace = self._forward_act_model(
            action_features,
            action_masks=action_mask_torch,
            current_state=current_state,
            current_state_mask=current_state_mask,
            mode="eval",
            repeated_diffusion_steps=sample_times,
            cfg_scale=cfg_scale,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
            guidance_fn=guidance_fn,
            guidance_scale=guidance_scale,
            guidance_start_frac=guidance_start_frac,
            guidance_end_frac=guidance_end_frac,
            guidance_grad_clip=guidance_grad_clip,
            return_guidance_trace=return_guidance_trace,
            fixed_actions=fixed_actions,
            fixed_action_mask=fixed_action_mask,
            return_replan_trace=return_replan_trace,
        )
        x_mask = action_mask_torch.unsqueeze(0).repeat(sample_times, 1, 1, 1)
        x_mask = x_mask.view(sample_times, action_mask_torch.shape[1], action_mask_torch.shape[2])
        action_np = samples.cpu().numpy() * x_mask.cpu().numpy()
        if return_guidance_trace or return_replan_trace:
            return action_np, trace
        return action_np

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_masks: Optional[torch.BoolTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        mode="train",
        **kwargs,
    ):
        features = self.extract_cognition_features(
            pixel_values,
            input_ids,
            attention_mask=attention_mask,
            current_state_mask=current_state_mask,
            current_state=current_state,
            fov=fov,
        )
        if mode == "vlm_cognition":
            return features
        _, action_loss = self._forward_act_model(
            features,
            action_labels=action_labels,
            action_masks=action_masks,
            current_state=current_state,
            current_state_mask=current_state_mask,
            mode="train",
            repeated_diffusion_steps=self.repeated_diffusion_steps,
        )
        return action_loss

    def predict_action(
        self,
        image,
        instruction: str,
        current_state,
        current_state_mask,
        use_ddim=True,
        num_ddim_steps=10,
        cfg_scale=5.0,
        action_mask_torch=None,
        fov=None,
        sample_times=1,
        use_cache=False,
        guidance_fn=None,
        guidance_scale: float = 0.0,
        guidance_start_frac: float = 0.0,
        guidance_end_frac: float = 1.0,
        guidance_grad_clip: float = 1.0,
        return_guidance_trace: bool = False,
        fixed_actions=None,
        fixed_action_mask=None,
        return_replan_trace: bool = False,
    ) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        model_inputs = self.processor(text=instruction, images=[image], return_tensors="pt")
        device = current_state.device
        pixel_values = model_inputs["pixel_values"].to(device)
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
        features = self.extract_cognition_features(
            pixel_values,
            input_ids,
            attention_mask=attention_mask,
            current_state_mask=current_state_mask,
            current_state=current_state,
            fov=fov,
        )
        samples, trace = self._forward_act_model(
            features,
            action_masks=action_mask_torch,
            current_state=current_state,
            current_state_mask=current_state_mask,
            mode="eval",
            repeated_diffusion_steps=sample_times,
            cfg_scale=cfg_scale,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
            guidance_fn=guidance_fn,
            guidance_scale=guidance_scale,
            guidance_start_frac=guidance_start_frac,
            guidance_end_frac=guidance_end_frac,
            guidance_grad_clip=guidance_grad_clip,
            return_guidance_trace=return_guidance_trace,
            fixed_actions=fixed_actions,
            fixed_action_mask=fixed_action_mask,
            return_replan_trace=return_replan_trace,
        )
        x_mask = action_mask_torch.unsqueeze(0).repeat(sample_times, 1, 1, 1)
        x_mask = x_mask.view(sample_times, action_mask_torch.shape[1], action_mask_torch.shape[2])
        action_np = samples.cpu().numpy() * x_mask.cpu().numpy()
        if return_guidance_trace or return_replan_trace:
            return action_np, trace
        return action_np
