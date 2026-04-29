# VITRA DiT and VLM Distillation Results

## Common Setup

- Teacher: VITRA PaliGemma2-3B + DiT-B checkpoint `epoch=0-step=140000.ckpt`.
- Data root: `/home/chonghej/scratch/chonghej/VLA-HAND/datasets/vitra_gigahands_real_all_cam0_keypoints_mano_linked`.
- Distillation train split: `gigahands_real_train`.
- Test split: `gigahands_real_test`, sequential first `200` clips.
- Action representation: all-cam0 keypoints/MANO linked setting, `action_dim=192`, `fwd_pred_next_n=16`.
- Test metric: normalized action MSE, lower is better. All reported 200-clip tests use `3065` valid bimanual frames.
- Evaluation sampling: DDIM `10` steps, CFG scale `5.0`.
- Latency protocol: NVIDIA RTX PRO 6000 Blackwell Max-Q, sample index `0`, BF16 enabled, GPU model forward only, no CPU image loading/tokenization/dataloader time, `5` warmup iterations and `20` profiled iterations.

Important scope note: the DiT-distilled model still uses the original PaliGemma VLM, and the VLM-distilled model still uses the full DiT-B action head. We have not yet built or measured a single combined model that uses both the compact VLM and compact DiT.

## What Was Compressed

| Track | Teacher part | Student part | What stays unchanged |
| --- | --- | --- | --- |
| DiT distillation | DiT-B action diffusion head, `136.59M` action-head params | DiT-B-6L shallow action head, `72.82M` action-head params | PaliGemma2-3B VLM/cognition path |
| VLM distillation | PaliGemma2-3B cognition feature extractor | DINOv2-base vision encoder + DistilBERT text encoder + fusion/projection to 2304-dim cognition feature | DiT-B action head copied from teacher |

DiT distillation uses same-step diffusion KD. For each batch, the teacher PaliGemma produces the cognition feature `z`; both teacher and student DiT denoisers see the same noised action `x_t`, same timestep, state, mask, and `z`. The student loss is:

`loss = MSE(eps_student, sampled_noise) + MSE(eps_student, eps_teacher)`

The full DiT distillation run trained to step `50000`; the selected checkpoint is `best.ckpt`, with final logged loss `0.017386`, `loss_noise_gt=0.013840`, and `loss_eps_kd=0.003547`.

VLM distillation replaces the generative PaliGemma backbone with encoder-only cognition extraction. The training loss for the current run is cognition feature MSE:

`loss = MSE(student_cognition, teacher_cognition)`

The run was stopped at `epoch=0-step=8000.ckpt`; automatic 200-clip evals were produced every 1000 steps. The best observed eval MSE was at step `1000` (`action_mse=0.109947`), but only steps `6000/7000/8000` remain after checkpoint pruning, so the retained model reported below is step `8000`.

## Model Size and Peak GPU Memory

Peak memory is `torch.cuda.max_memory_allocated()` during the unified latency profile. It is a model-forward memory measurement, not total process memory from `nvidia-smi`.

| Model | Total params | Action-head params | Non-action params | Peak profile memory |
| --- | ---: | ---: | ---: | ---: |
| Teacher VITRA step140000 | 3174.15M | 136.59M | 3037.56M | 17.16 GiB |
| DiT student B-6L best | 3110.38M | 72.82M | 3037.56M | 16.92 GiB |
| VLM student step8000 | 299.15M | 136.59M | 162.56M | 1.14 GiB |

Compression summary:

- DiT compression reduces the action head by `46.69%` (`1.88x` smaller), but total model params only drop `2.01%` because the PaliGemma VLM remains.
- VLM compression reduces total params by `90.58%` (`10.61x` smaller) and profile peak GPU memory by `93.33%` (`15.00x` lower), because the 3B PaliGemma backbone is replaced.

## Testing Performance

All rows use `gigahands_real_test`, `200` clips, DDIM `10`, CFG `5.0`.

| Model | Checkpoint | action_mse | left_action_mse | right_action_mse | dual_hand_action_mse |
| --- | --- | ---: | ---: | ---: | ---: |
| Base VITRA-VLA-3B | `/home/chonghej/scratch/chonghej/VLA-HAND/checkpoints/vitra-vla-3b.pt` | 16.101652 | 3.396869 | 45.141144 | 16.101652 |
| Teacher step140000 | `/home/chonghej/scratch/chonghej/VLA-HAND/runs/gigahands_real_all_cam0_keypoints_mano_vitra3b_linked_train/checkpoints/gigahands_real_all_cam0_keypoints_mano_vitra3b_linked_stage1_TB2_B2_bf16True/checkpoints/epoch=0-step=140000.ckpt` | 0.705922 | 0.787389 | 0.519712 | 0.705922 |
| DiT student B-6L best | `/home/chonghej/scratch/chonghej/DiT-Distill/outputs/distill/all_cam0_keypoints_mano_b6l_full/checkpoints/best.ckpt` | 0.760176 | 0.801089 | 0.666663 | 0.760176 |
| VLM student step8000 | `/home/chonghej/scratch/chonghej/VLM-Distill/runs/vlm_distill_encoder_student_all_cam0_keypoints_mano/checkpoints/vlm_distill_encoder_student_all_cam0_keypoints_mano_TB2_B1_bf16True/checkpoints/epoch=0-step=8000.ckpt` | 0.362805 | 0.379350 | 0.324990 | 0.362805 |

Performance relative to teacher:

- DiT B-6L is `+0.054254` action MSE, or `7.69%` worse than the teacher.
- VLM student step8000 is `-0.343117` action MSE, or `48.61%` lower MSE than the teacher on this 200-clip test.
- The best observed VLM eval checkpoint was step `1000` with `action_mse=0.109947`, but that checkpoint is no longer retained.

## Unified Latency Results

Latency rows below use the common protocol from the setup section. "Feature" means PaliGemma/VLM or encoder cognition feature extraction. "Action head" means cached-feature 10-step DDIM action sampling. "End-to-end" means feature extraction + action head GPU forward, excluding CPU preprocessing.

| Track | Model | Feature mean | Action-head mean | End-to-end mean |
| --- | --- | ---: | ---: | ---: |
| DiT | Teacher step140000 | 41.49 ms | 75.09 ms | 113.46 ms |
| DiT | DiT student B-6L best | 40.92 ms | 44.83 ms | 89.57 ms |
| VLM | Teacher step140000 | 40.05 ms | 70.76 ms | 111.21 ms |
| VLM | VLM student step8000 | 5.52 ms | 70.81 ms | 76.84 ms |

Latency reduction:

- DiT compression: action-head latency improves `1.68x` (`40.31%` lower); end-to-end GPU latency improves `1.27x` (`21.06%` lower). VLM feature time is unchanged because PaliGemma remains.
- VLM compression: feature latency improves `7.25x` (`86.20%` lower); end-to-end GPU latency improves `1.45x` (`30.90%` lower). Action-head latency is unchanged because DiT-B remains.

## Artifact Index

- DiT best evaluation: `/home/chonghej/scratch/chonghej/DiT-Distill/outputs/eval/all_cam0_keypoints_mano_b6l_full_best_200/report.md`
- DiT unified latency:
  - `/home/chonghej/scratch/chonghej/DiT-Distill/outputs/profile/unified_all_cam0_keypoints_mano/teacher_step140000.json`
  - `/home/chonghej/scratch/chonghej/DiT-Distill/outputs/profile/unified_all_cam0_keypoints_mano/dit_student_b6l_best.json`
- DiT loss curve:
  - `/home/chonghej/scratch/chonghej/DiT-Distill/outputs/distill/all_cam0_keypoints_mano_b6l_full/logs/loss_curve.csv`
  - `/home/chonghej/scratch/chonghej/DiT-Distill/outputs/distill/all_cam0_keypoints_mano_b6l_full/logs/loss_curve.png`
- VLM 200-clip comparison: `/home/chonghej/scratch/chonghej/VLM-Distill/outputs/eval/vlm_encoder_all_cam0_stop_compare_200/report.md`
- VLM unified latency:
  - `/home/chonghej/scratch/chonghej/VLM-Distill/outputs/latency/vlm_encoder_all_cam0_stop_compare/teacher_step140000.json`
  - `/home/chonghej/scratch/chonghej/VLM-Distill/outputs/latency/vlm_encoder_all_cam0_stop_compare/vlm_student_step8000.json`

## Main Takeaways

DiT distillation successfully reduces the diffusion action head and cuts action-head latency substantially, but its total memory impact is small because the 3B VLM dominates the model. The retained DiT student is slightly worse than the teacher on the 200-clip action MSE test.

VLM distillation is the larger system-level compression. It reduces total params from `3.17B` to `299M`, cuts peak profile memory from `17.16 GiB` to `1.14 GiB`, and reduces end-to-end GPU latency from roughly `111 ms` to `77 ms` while improving action MSE on the current 200-clip test.

The next clean experiment is to compose the two students into one model: encoder-only VLM plus DiT-B-6L action head. That combined model should be measured directly, because adding the separate speedups would overstate what has actually been tested.
