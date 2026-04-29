from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from vitra.models.vla.vitra_paligemma import VITRA_Paligemma


class MaskedFeatureGenerator(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        intermediate_size: int = 3072,
        dropout: float = 0.0,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, features: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        padding_mask = None if token_mask is None else ~token_mask.bool()
        return self.encoder(features, src_key_padding_mask=padding_mask)


class VITRA_SmallPaliGemmaStudent(VITRA_Paligemma):
    """Same-family compact VITRA student for ViTKD-style feature distillation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.student_vlm_configs = self.configs.get("student_vlm", {})
        self.raw_hidden_size = int(self.hidden_size)
        self.output_size = int(
            self.student_vlm_configs.get(
                "student_output_size",
                self.act_model_configs.get("token_size", self.raw_hidden_size),
            )
        )
        self.vitkd_shallow_layers = list(self.student_vlm_configs.get("vitkd_shallow_layers", [0, 1]))
        self.vitkd_deep_layer = int(self.student_vlm_configs.get("vitkd_deep_layer", -1))
        self.vitkd_mask_ratio = float(self.student_vlm_configs.get("vitkd_mask_ratio", 0.5))

        self.cognition_projection = (
            nn.Identity()
            if self.raw_hidden_size == self.output_size
            else nn.Linear(self.raw_hidden_size, self.output_size)
        )
        self.vitkd_shallow_projectors = nn.ModuleList(
            [nn.Linear(self.raw_hidden_size, self.output_size) for _ in self.vitkd_shallow_layers]
        )
        self.vitkd_deep_projector = nn.Linear(self.raw_hidden_size, self.output_size)
        self.vitkd_mask_token = nn.Parameter(torch.zeros(self.output_size))
        self.vitkd_generator = MaskedFeatureGenerator(
            hidden_size=self.output_size,
            num_layers=int(self.student_vlm_configs.get("vitkd_generator_layers", 2)),
            num_heads=int(self.student_vlm_configs.get("vitkd_generator_heads", 8)),
            intermediate_size=int(self.student_vlm_configs.get("vitkd_generator_intermediate_size", 3072)),
            dropout=float(self.student_vlm_configs.get("vitkd_generator_dropout", 0.0)),
        )
        nn.init.normal_(self.vitkd_mask_token, mean=0.0, std=0.02)

    def trainable_params_setup_for_vlm_distill(self, freeze_option: str = "small_vitra_vitkd"):
        self.model.config.use_cache = False
        self.requires_grad_(False)

        if freeze_option in {"small_vitra_vitkd", "small_vitra_cognition"}:
            self.text_tower.requires_grad_(True)
            self.word_embedding.requires_grad_(True)
            self.cognition_projection.requires_grad_(True)
            self.vitkd_shallow_projectors.requires_grad_(True)
            self.vitkd_deep_projector.requires_grad_(True)
            self.vitkd_generator.requires_grad_(True)
            self.vitkd_mask_token.requires_grad_(True)
            if self.use_state == "VLM":
                self.vlm_state_encoder.requires_grad_(True)
            if self.use_fov:
                self.fov_encoder.requires_grad_(True)
            if self.cognition_token is not None:
                self.cognition_token.requires_grad_(True)
        elif freeze_option == "small_vitra_action_head_only":
            if self.act_model is not None:
                self.act_model.requires_grad_(True)
        elif freeze_option == "small_vitra_action":
            self.trainable_params_setup_for_vlm_distill("small_vitra_vitkd")
            if self.act_model is not None:
                self.act_model.requires_grad_(True)
        else:
            raise ValueError(f"Unsupported small VITRA freeze option: {freeze_option}")

        if not bool(self.student_vlm_configs.get("freeze_vision_encoder", True)):
            self.vision_tower.requires_grad_(True)

    def extract_cognition_features(self, *args, **kwargs) -> torch.Tensor:
        raw_cognition = super().extract_cognition_features(*args, **kwargs)
        return self.cognition_projection(raw_cognition)

    def extract_vitkd_features(self, *args, **kwargs) -> dict:
        kwargs.setdefault("shallow_layers", self.vitkd_shallow_layers)
        kwargs.setdefault("deep_layer", self.vitkd_deep_layer)
        features = super().extract_vitkd_features(*args, **kwargs)
        features["raw_cognition"] = features["cognition"]
        features["cognition"] = self.cognition_projection(features["cognition"])
        features["raw_shallow_features"] = features["shallow_features"]
        features["shallow_features"] = self.project_shallow_features(features["raw_shallow_features"])
        features["raw_deep_feature"] = features["deep_feature"]
        (
            features["deep_generated"],
            features["deep_generation_mask"],
            features["deep_feature"],
        ) = self.build_deep_generation(features["raw_deep_feature"], features["token_mask"])
        return features

    def project_shallow_features(self, shallow_features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            projector(feature)
            for projector, feature in zip(self.vitkd_shallow_projectors, shallow_features)
        ]

    def build_deep_generation(
        self,
        deep_feature: torch.Tensor,
        token_mask: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.vitkd_deep_projector(deep_feature)
        valid_mask = token_mask.bool()
        ratio = self.vitkd_mask_ratio if mask_ratio is None else float(mask_ratio)
        sampled_mask = (torch.rand(valid_mask.shape, device=valid_mask.device) < ratio) & valid_mask

        empty_rows = sampled_mask.sum(dim=1) == 0
        if empty_rows.any():
            last_valid = valid_mask.long().cumsum(dim=1).argmax(dim=1)
            sampled_mask[empty_rows, last_valid[empty_rows]] = True

        mask_token = self.vitkd_mask_token.to(dtype=projected.dtype, device=projected.device)
        masked_input = torch.where(sampled_mask.unsqueeze(-1), mask_token.view(1, 1, -1), projected)
        generated = self.vitkd_generator(masked_input, valid_mask)
        return generated, sampled_mask, projected

    def _forward_act_model(self, vlm_features: torch.Tensor, *args, **kwargs):
        projected_features = self.cognition_projection(vlm_features)
        return super()._forward_act_model(projected_features, *args, **kwargs)
