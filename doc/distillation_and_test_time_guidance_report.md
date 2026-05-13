# Distillation and Test-Time Guidance Experiments

## Executive Summary

This report summarizes two compressed VITRA distillation experiments and a new
test-time guidance experiment. The distillation experiments evaluate whether a smaller
student model can preserve GigaHands action prediction performance. The test-time
guidance experiment evaluates whether external polynomial-region guidance can reduce
constraint violation during diffusion-based action generation without any additional
training.

Key takeaways:

- Both distilled students are much better than the original VITRA-3B base model on the
  cleaned GigaHands test split.
- The older distilled student gives the best action MSE in the full-test comparison.
- The newer joint-KD student gives much stronger feature alignment to the finetuned
  step-140000 teacher, but its action MSE is slightly worse.
- Test-time guidance works when evaluated in a fair same-prefix / same-noise ablation:
  moderate guidance scales reduce region violation, while very large scales destabilize
  the trajectory.

## 1. Distillation Experiments

### 1.1 Shared Evaluation Setup

The two distilled checkpoints were evaluated on the same cleaned GigaHands test split.

| Item | Value |
| --- | --- |
| Dataset | `datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked_cleaned` |
| Split | `gigahands_real_test` |
| Evaluation clips | 1,495 clips |
| Evaluation strategy | `middle_per_episode` |
| Valid frames | 23,920 bimanual frames |
| Action type | `keypoints` |
| Action dimension | 192 |
| Prediction horizon | 16 |
| Inference | 10-step DDIM, CFG scale 5.0 |

The evaluation metrics are action MSE values against the same GigaHands converted ground
truth target. Lower is better.

### 1.2 Model Variants

| Model | Role | Checkpoint / Config | Objective |
| --- | --- | --- | --- |
| Base VITRA-3B | Original base model | `checkpoints/vitra-vla-3b.pt` | Original VITRA pretraining |
| Finetuned VITRA step140000 | Strong teacher | `runs/gigahands_real_all_cam0_keypoints_mano_vitra3b_linked_train/.../epoch=0-step=140000.ckpt` | GigaHands finetuning on keypoints-MANO targets |
| Old distilled student step50000 | Compressed student | `runs/finetune_distill_all_cam0_keypoints_mano/.../epoch=0-step=50000.ckpt` | VLM cognition feature distillation from original VITRA + GT diffusion action loss |
| New joint-KD distilled student step50000 | Compressed student | `runs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano/.../epoch=0-step=50000.ckpt` | VLM cognition feature distillation from step140000 teacher + GT action loss + action-head KD |

Both students use the compressed `VITRA_EncoderStudent` architecture:

- DINOv2-base vision encoder.
- DistilBERT text encoder.
- Fused cognition projection to 2304 dimensions.
- 6-layer `DiT-B-6L` action head.
- Copied teacher action-head blocks `[0, 2, 4, 6, 8, 10]` into the 6-layer student head.

### 1.3 Distillation Scheme A: Feature Distillation + Ground-Truth Action Loss

This scheme uses the original VITRA-3B base model as the feature teacher and trains the
compressed student with:

```text
total loss = normalized feature loss + normalized ground-truth action diffusion loss
```

Important training settings:

| Item | Value |
| --- | --- |
| Config | `vitra/configs/finetune_distill_all_cam0_keypoints_mano.json` |
| Launch script | `scripts/run_finetune_distill_all_cam0_full_tmux.sh` |
| Teacher checkpoint | `checkpoints/vitra-vla-3b.pt` |
| Student action head | `DiT-B-6L` |
| Max steps | 50,000 |
| Save steps | 20,000 |
| Per-GPU batch size | 2 |
| Total batch size | 8 |
| Repeated diffusion steps | 4 |
| Feature loss weight | 1.0 |
| Action loss weight | 1.0 |

Result on the full cleaned GigaHands test split:

| Model | Action MSE ↓ | Left MSE ↓ | Right MSE ↓ | Dual-hand MSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Base VITRA-3B | 16.1358 | 3.4089 | 45.2258 | 16.1358 |
| Finetuned VITRA step140000 | 0.4061 | 0.4763 | 0.2456 | 0.4061 |
| Old distilled student | 0.3778 | 0.4301 | 0.2583 | 0.3778 |

Feature alignment to the step140000 teacher:

| Model | VLM cognition MSE ↓ | VLM cognition cosine ↑ |
| --- | ---: | ---: |
| Base VITRA-3B | 0.014132 | 0.6767 |
| Old distilled student | 0.008757 | 0.7642 |

Interpretation:

- The old distilled student improves action MSE by 97.66% relative to the original
  VITRA-3B base.
- It slightly outperforms the step140000 teacher on overall action MSE, although the
  teacher remains slightly better on right-hand MSE.
- It improves feature alignment compared with the base model, but feature alignment is
  not as strong as the newer joint-KD scheme.

### 1.4 Distillation Scheme B: Joint Feature Distillation + Ground-Truth Action Loss + Action-Head KD

This scheme uses the stronger step140000 finetuned VITRA checkpoint as the teacher and
adds an action-head KD loss. The student is trained with:

```text
total loss =
  normalized feature loss
  + normalized ground-truth action diffusion loss
  + normalized action-head KD loss
```

The action-head KD term matches the teacher and student diffusion noise predictions at
the same noisy action sample and diffusion timestep.

Important training settings:

| Item | Value |
| --- | --- |
| Config | `vitra/configs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano.json` |
| Launch script | `scripts/run_finetune_distill_step140000_joint_kd_full_tmux.sh` |
| Teacher checkpoint | step140000 GigaHands-finetuned VITRA |
| Student action head | `DiT-B-6L` |
| Max steps | 50,000 |
| Save steps | 50,000 |
| Per-GPU batch size | 2 |
| Total batch size | 4 |
| Repeated diffusion steps | 4 |
| Feature loss weight | 1.0 |
| Ground-truth action loss weight | 1.0 |
| Action KD loss weight | 1.0 |

Result on the full cleaned GigaHands test split:

| Model | Action MSE ↓ | Left MSE ↓ | Right MSE ↓ | Dual-hand MSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Base VITRA-3B | 16.1358 | 3.4089 | 45.2258 | 16.1358 |
| Finetuned VITRA step140000 | 0.4061 | 0.4763 | 0.2456 | 0.4061 |
| New joint-KD student | 0.4532 | 0.5367 | 0.2624 | 0.4532 |

Feature alignment to the step140000 teacher:

| Model | VLM cognition MSE ↓ | VLM cognition cosine ↑ |
| --- | ---: | ---: |
| Base VITRA-3B | 0.014132 | 0.6767 |
| New joint-KD student | 0.000339 | 0.9776 |

Interpretation:

- The joint-KD student improves action MSE by 97.19% relative to the original VITRA-3B
  base.
- Its feature alignment to the step140000 teacher is much stronger than the old
  distilled student: VLM cognition MSE drops from 0.014132 to 0.000339, and cosine rises
  to 0.9776.
- Its action MSE is worse than the old distilled student and 11.60% worse than the
  step140000 teacher. This suggests that the added action-head KD strongly transfers the
  teacher representation, but the action loss / KD loss balance may need tuning to recover
  the best action MSE.

### 1.5 Distillation Summary

| Model | Action MSE ↓ | Improvement vs Base ↑ | Feature MSE to Teacher ↓ | Feature Cosine to Teacher ↑ |
| --- | ---: | ---: | ---: | ---: |
| Base VITRA-3B | 16.1358 | - | 0.014132 | 0.6767 |
| Finetuned VITRA step140000 | 0.4061 | 97.48% | N/A | N/A |
| Old distilled student | 0.3778 | 97.66% | 0.008757 | 0.7642 |
| New joint-KD student | 0.4532 | 97.19% | 0.000339 | 0.9776 |

The old student is currently better for pure action-MSE performance. The new joint-KD
student is better if the priority is matching the finetuned teacher representation.

### 1.6 Inference-Time Speedup

We also measured segmented inference time for the original VITRA-3B model and the two
compressed students. The timing is GPU-synchronized and excludes model loading and dataset
I/O. The segmented timing also excludes CPU tokenization / image preprocessing, so it
isolates the model-side computation:

- VLA backend: feature / cognition extraction.
- Diffusion part: 10-step DDIM action-head sampling.
- Backend + diffusion: sum of the two model-side stages.

Profiling setup:

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| Input image | `examples/0002.jpg` |
| Instruction | `Left hand: None. Right hand: Pick up the phone on the table.` |
| DDIM steps | 10 |
| CFG scale | 5.0 |
| Warmup iterations | 5 |
| Measured iterations | 45 |

Segmented timing:

| Model | VLA backend ms ↓ | Diffusion ms ↓ | Backend + diffusion ms ↓ | Backend speedup ↑ | Diffusion speedup ↑ | Total model-side speedup ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base VITRA-3B | 74.10 | 73.71 | 147.81 | 1.00x | 1.00x | 1.00x |
| Old distilled student | 8.49 | 44.60 | 53.09 | 8.73x | 1.65x | 2.78x |
| New joint-KD student | 8.47 | 44.52 | 52.99 | 8.75x | 1.66x | 2.79x |

End-to-end `predict_action` timing from the earlier 200-clip evaluation gives a similar
practical speedup:

| Model | Mean sec / clip ↓ | Speedup vs Base ↑ |
| --- | ---: | ---: |
| Base VITRA-3B | 0.1590 | 1.00x |
| Old distilled student | 0.0656 | 2.43x |

Interpretation:

- Most of the acceleration comes from replacing the large PaliGemma VLA backend with the
  compact DINOv2 + DistilBERT encoder backend. This gives about 8.7x backend speedup.
- The diffusion action head is also faster because the student uses a 6-layer `DiT-B-6L`
  instead of the full 12-layer action head. This gives about 1.65x diffusion speedup.
- The combined model-side inference speedup is about 2.8x. The practical end-to-end
  `predict_action` speedup measured over evaluation clips is about 2.4x.

## 2. Test-Time Guidance Experiment

### 2.1 Goal

The test-time guidance experiment checks whether we can reduce violation of an external
polynomial-region constraint during VITRA action generation without training a new model.

The mechanism is true replanning:

1. Generate an initial 16-step action chunk.
2. At replan index `K=5`, clamp the already generated prefix `0:5`.
3. Re-run DDIM and regenerate only the future suffix `5:16` under guidance.
4. At replan index `K=10`, clamp prefix `0:10`.
5. Re-run DDIM and regenerate only suffix `10:16`.

The prefix is clamped at every DDIM denoising step, so already executed actions cannot be
changed by future guidance.

### 2.2 Implementation

The implementation adds a prefix-clamped replanning sampler:

- `GaussianDiffusion.ddim_sample_loop_replanning_guided(...)`
- `DiffusionPolicy.sample(..., fixed_actions, fixed_action_mask, return_replan_trace)`
- `VITRA_Paligemma.predict_action(..., fixed_actions, fixed_action_mask)`
- Toy report script: `scripts/inference_guided_replanning_toy.py`

For classifier-free guidance, the fixed prefix is duplicated consistently across the
conditional and unconditional branches. The external guidance loss is applied through the
existing guidance wrapper and does not modify fixed prefix dimensions.

### 2.3 Evaluation Setup

| Item | Value |
| --- | --- |
| Model | Base VITRA-3B |
| Checkpoint | `checkpoints/vitra-vla-3b.pt` |
| Input image | `examples/0002.jpg` |
| Instruction | `Left hand: None. Right hand: Pick up the phone on the table.` |
| Action horizon | 16 |
| Replan points | `K=5`, `K=10` |
| DDIM steps | 10 |
| CFG scale | 5.0 |
| Guidance dimensions | `[51, 52]` |
| Action mask | right-hand slice `51:102` |
| Guidance loss | polynomial / quadratic region violation |
| Seeds | 10 seeds, `0-9` |
| Tested guidance scales | `0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0` |

The main comparison is a fair same-prefix / same-noise ablation:

- Unguided replan and guided replan use the same fixed prefix.
- They use the same random seed for the diffusion noise.
- The only difference is whether the polynomial guidance gradient is applied.

This is the cleanest way to isolate the effect of guidance. Directly comparing a guided
replan against the initial chunk is less fair, because replanning itself resamples future
noise.

For the final sweep, `temporal_mask=target` was used. This constrains each polynomial
region at its own target timestep only. The earlier `tail` mask can make multiple regions
pull on the whole suffix and create conflicting gradients, so it is less clean for
demonstrating the basic effect of guidance.

### 2.4 Main Results

Lower violation is better. Negative delta means guidance reduced violation.

| Guidance scale | K5 unguided ↓ | K5 guided ↓ | K5 delta | K5 improve rate | K10 unguided ↓ | K10 guided ↓ | K10 delta | K10 improve rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.493522 | 0.493105 | -0.000417 | 60% | 0.984576 | 0.982497 | -0.002079 | 60% |
| 0.05 | 0.493522 | 0.491431 | -0.002091 | 70% | 0.985128 | 0.974727 | -0.010400 | 60% |
| 0.10 | 0.493522 | 0.489310 | -0.004212 | 70% | 0.985763 | 0.964912 | -0.020851 | 60% |
| 0.20 | 0.493522 | 0.484971 | -0.008551 | 60% | 0.986862 | 0.944798 | -0.042064 | 60% |
| 0.50 | 0.493522 | 0.474763 | -0.018759 | 60% | 0.988796 | 0.883104 | -0.105691 | 60% |
| 1.00 | 0.543184 | 0.481091 | -0.062092 | 80% | 0.849722 | 0.823779 | -0.025943 | 70% |
| 2.00 | 0.543184 | 0.456102 | -0.087081 | 60% | 0.823132 | 0.772271 | -0.050861 | 70% |
| 5.00 | 0.543184 | 6.993579 | +6.450395 | 30% | 0.797351 | 1.967689 | +1.170338 | 30% |
| 10.00 | 0.543184 | 16.351751 | +15.808568 | 20% | 0.581281 | 23.015629 | +22.434348 | 10% |

Prefix preservation:

| Metric | Value |
| --- | ---: |
| Max prefix error at K=5 | 0.0 |
| Max prefix error at K=10 | 0.0 |

Runtime overhead:

- Guided replanning is roughly 3x slower than unguided replanning at K=5.
- Guided replanning is roughly 2.4x to 2.6x slower than unguided replanning at K=10.
- This is expected because guidance requires gradient computation during DDIM sampling.

### 2.5 Guidance Result Analysis

The guidance experiment supports three conclusions.

First, moderate guidance works. For scales from `0.01` to `2.0`, guided replanning lowers
the average region violation at both K=5 and K=10. The best scale in this sweep is
approximately `2.0`, where K5 violation drops from 0.543184 to 0.456102 and K10 violation
drops from 0.823132 to 0.772271.

Second, prefix clamping works. Across all tested scales, including unstable large scales,
the maximum prefix error remains exactly 0.0. This means guidance only affects the
unexecuted future suffix, not the already executed prefix.

Third, guidance scale must be tuned. Very large scales, especially `5.0` and `10.0`,
destabilize the trajectory and dramatically increase violation. This is the expected
failure mode of gradient-based guidance: if the external loss dominates the base diffusion
prior, the action chunk can move off the model manifold.

### 2.6 Recommended Reporting Language

For presentation, the safest claim is:

> In a controlled same-prefix / same-noise ablation, test-time polynomial guidance reduces
> future-region violation for moderate guidance scales without modifying the executed
> prefix. The method has a clear scale tradeoff: small scales have weak effects, moderate
> scales improve constraint satisfaction, and overly large scales destabilize the action
> trajectory.

Avoid claiming that guidance always improves all cases. The seed-level improvement rate is
not 100%, and very large guidance scales fail.

## 3. Source Files

Distillation:

- `runs/distill_comparison/report_summary.md`
- `runs/distill_comparison/old_distill_step50000_fulltest/metrics.json`
- `runs/distill_comparison/new_joint_kd_step50000_fulltest/metrics.json`
- `vitra/configs/finetune_distill_all_cam0_keypoints_mano.json`
- `vitra/configs/finetune_distill_step140000_joint_kd_all_cam0_keypoints_mano.json`

Test-time guidance:

- `doc/test-time-guidance/vitra_guided_replanning_toy_plan.md`
- `scripts/inference_guided_replanning_toy.py`
- `outputs/replanning_guidance/fair_ablation_targetmask_seed0_9_scales/report.json`
- `outputs/replanning_guidance/fair_ablation_targetmask_large_scales/report.json`
