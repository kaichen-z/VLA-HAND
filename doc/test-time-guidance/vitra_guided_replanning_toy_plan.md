# VITRA Test-Time Guided Replanning Toy Plan

## 1. Purpose

This document describes a toy experiment for **test-time guided replanning** in VITRA.
The goal is to validate a closed-loop style mechanism where VITRA first generates an
action chunk, executes part of it, then **re-runs diffusion to regenerate the unexecuted
future suffix** under an additional guidance loss.

This is the only supported diffusion-time guidance path in this repo. The old one-shot
guidance entrypoint has been removed so guidance is always evaluated as prefix-clamped
replanning after part of the chunk has already executed.

The toy guidance target is still a polynomial / quadratic region constraint. Later, the
same replanning interface can be used with tactile, contact, safety, or task-progress
energies.

## 2. Current Repo Anchors

The current repo has the pieces needed for prefix-clamped guided replanning:

- `vitra/guidance/polynomial_guidance.py`
  - `PolynomialRegionLoss`
  - `QuadraticRegion`
  - `CFGAwareGuidanceWrapper`
- `vitra/models/action_model/gaussian_diffusion.py`
  - `ddim_sample_loop_replanning_guided(...)`
- `vitra/models/action_model/diffusion_policy.py`
  - `DiffusionPolicy.sample(...)`
- `vitra/models/vla/vitra_paligemma.py`
  - `VITRA_Paligemma.predict_action(...)`
- `scripts/inference_guided_replanning_toy.py`
  - polynomial guided replanning demo

## 3. Replanning Scenario

Assume VITRA predicts an action chunk of length:

```text
T = 16
```

At real time `t = 0`, we run normal VITRA inference:

```text
A0 = sample(image_0, instruction)
```

Then the policy executes the first few action steps. At selected replanning points:

```text
K = 5
K = 10
```

we call the diffusion sampler again:

```text
A5  = replan(prefix=A0[0:5],  image=image_0, instruction=instruction, guidance=region_loss)
A10 = replan(prefix=A5[0:10], image=image_0, instruction=instruction, guidance=region_loss)
```

The prefix is treated as already executed and must remain fixed. The suffix is resampled:

```text
A5[0:5]    == A0[0:5]
A10[0:10]  == A5[0:10]
A5[5:16]   is regenerated
A10[10:16] is regenerated
```

For the toy experiment, we can reuse the same image and instruction at each replan. In a
real robot loop, the image/state should be refreshed from the live observation.

## 4. Core Algorithm

### 4.1 Prefix-Clamped DDIM Replanning

During diffusion sampling, the sampler maintains a noisy sample `x_t` over the full action
chunk. For prefix-clamped replanning, the prefix part should be overwritten at every DDIM
step with the forward-diffused version of the fixed clean prefix:

```text
x_t[:, :K, :] =
    sqrt(alpha_bar_t) * fixed_x0[:, :K, :]
  + sqrt(1 - alpha_bar_t) * eps_prefix[:, :K, :]
```

where:

- `fixed_x0` is the fixed clean action prefix in the same normalized action space used by
  the diffusion model.
- `eps_prefix` is one fixed noise tensor sampled once per replanning call.
- `K` is the number of already executed action steps.

The same prefix clamping must happen:

1. before each denoiser call,
2. after each DDIM update, or equivalently before the next step,
3. at the final output, where the clean prefix should match `fixed_x0` exactly.

### 4.2 Guidance Should Only Affect the Future Suffix

The guidance loss should be computed on predicted clean actions, usually `pred_xstart`.
For replanning at index `K`, the guidance gradient must be zero on the prefix:

```text
grad[:, :K, :] = 0
```

Only future steps should be guided:

```text
grad[:, K:, guide_dims] != 0
```

This prevents a constraint intended for the future from changing already executed actions.

### 4.3 CFG Handling

VITRA's classifier-free guidance path duplicates the batch into conditional and
unconditional branches. The external guidance loss should only be applied to the
conditional branch.

The current repo already has `CFGAwareGuidanceWrapper` for this purpose. The replanning
sampler should reuse this logic or preserve the same behavior.

If the duplicated batch is:

```text
[conditional samples, unconditional samples]
```

then the polynomial / region guidance should only compute gradients on the first half.

### 4.4 Guide Dimensions

The toy polynomial guidance should operate on dimensions that are inside the active action
mask. For the current GigaHands keypoint/MANO-style action setup, the right-hand action
slice has commonly been:

```text
51:102
```

Therefore a toy 2D region can use dimensions such as:

```text
guide_dims = [51, 52]
```

Using `[0, 1]` is usually a bad default for this repo because those dimensions may be
outside the hand/action slice being evaluated.

## 5. Proposed Code Changes

### 5.1 New Sampler Method

Add a new method in:

```text
vitra/models/action_model/gaussian_diffusion.py
```

Recommended name:

```python
ddim_sample_loop_replanning_guided(...)
```

Recommended arguments:

```python
fixed_xstart=None
fixed_mask=None
prefix_noise=None
guidance_fn=None
guidance_scale=0.0
guidance_grad_clip=None
return_guidance_trace=False
```

Expected shapes:

```text
fixed_xstart: [B, T, D]
fixed_mask:   [B, T, D] or broadcastable bool/float mask
prefix_noise: [B, T, D]
```

The method should:

1. initialize a full action noise tensor,
2. clamp the fixed prefix at every DDIM step,
3. run the denoiser,
4. compute guidance on `pred_xstart`,
5. zero guidance gradients outside the suffix/action mask,
6. perform the guided DDIM update,
7. return the final action and optional trace.

Do not add a separate one-shot guidance path. `ddim_sample_loop_replanning_guided(...)`
is the single DDIM guidance interface.

### 5.2 Policy API

Extend:

```text
vitra/models/action_model/diffusion_policy.py
```

`DiffusionPolicy.sample(...)` should accept:

```python
fixed_actions=None
fixed_action_mask=None
replan_start_idx=None
return_replan_trace=False
```

If `fixed_actions` / `fixed_action_mask` are absent, behavior should remain identical to
the current one-shot sampler.

### 5.3 VITRA Wrapper API

Extend:

```text
vitra/modeling_vitra.py
```

`VITRA_Paligemma.predict_action(...)` should accept and forward:

```python
fixed_actions=None
fixed_action_mask=None
replan_start_idx=None
return_replan_trace=False
```

This keeps the public inference API clean:

```python
model.predict_action(
    image,
    instruction,
    fixed_actions=prefix_actions,
    fixed_action_mask=prefix_mask,
    replan_start_idx=5,
    guidance_fn=guidance_fn,
    guidance_scale=guidance_scale,
)
```

## 6. Proposed Toy Script

Add:

```text
scripts/inference_guided_replanning_toy.py
```

Recommended CLI:

```bash
python scripts/inference_guided_replanning_toy.py \
  --config vitra/configs/human_pretrain_gigahands_real_all_cam0_keypoints_mano_vitra3b_linked.json \
  --checkpoint /path/to/checkpoint \
  --image /path/to/input_frame.png \
  --instruction "..." \
  --output_dir runs/test_time_guidance/replanning_toy \
  --replan_indices 5 10 \
  --guide_dims 51 52 \
  --guidance_scale 0.5 \
  --num_ddim_steps 10 \
  --seed 0
```

Outputs:

```text
initial_action_norm.npy
replan_k5_action_norm.npy
replan_k10_action_norm.npy
metrics.json
replan_trace_k5.json
replan_trace_k10.json
trajectory_replanning_xy.png
region_violation_bars.png
prefix_error.json
```

The saved actions should be in normalized model action space unless the script explicitly
adds an unnormalization step.

## 7. Metrics

The toy experiment should report four categories of metrics.

### 7.1 Prefix Preservation

The most important correctness check:

```text
max_abs_prefix_error_k5  = max(abs(A5[0:5] - A0[0:5]))
max_abs_prefix_error_k10 = max(abs(A10[0:10] - A5[0:10]))
```

Expected result:

```text
prefix error should be near zero
```

If this fails, the replanning implementation is not valid.

### 7.2 Region Violation

Compare region loss before and after replanning:

```text
region_loss_initial_suffix
region_loss_after_k5_suffix
region_loss_after_k10_suffix
```

Expected result:

```text
guided replanning should reduce suffix violation
```

### 7.3 Suffix Change

Measure how much the future trajectory changes:

```text
suffix_l2_delta_k5  = ||A5[5:16] - A0[5:16]||
suffix_l2_delta_k10 = ||A10[10:16] - A5[10:16]||
```

This confirms the sampler is actually regenerating the future.

### 7.4 Smoothness / Boundary Jump

Check whether replanning introduces a discontinuity at the execution boundary:

```text
boundary_jump_k5  = ||A5[5]  - A5[4]||
boundary_jump_k10 = ||A10[10] - A10[9]||
```

A large boundary jump means the guidance is too strong or the suffix is not conditioned
smoothly enough on the prefix.

## 8. Plots

The script should produce at least:

1. `trajectory_replanning_xy.png`
   - plot the chosen 2D action dimensions for:
     - initial chunk,
     - after K=5 replan,
     - after K=10 replan,
     - polynomial target region.
2. `region_violation_bars.png`
   - initial vs K=5 vs K=10 suffix violation.
3. Optional: `guidance_loss_curve_k5.png` and `guidance_loss_curve_k10.png`
   - loss over DDIM steps for each replan call.

## 9. Tests

Add focused tests before running the full VITRA script.

Recommended tests:

1. Prefix clamp test
   - create a toy `fixed_xstart`,
   - run replanning sampler,
   - verify final prefix equals fixed prefix.
2. Gradient mask test
   - verify guidance gradient is zero for `0:K`.
3. CFG branch test
   - verify guidance applies only to the conditional half.
4. Sequential replanning test
   - run K=5 then K=10,
   - verify K=10 output preserves `0:10` from the K=5 trajectory.
5. CLI smoke test
   - run the toy script with tiny settings and verify output files exist.

## 10. Acceptance Criteria

The implementation is successful when:

1. Guidance without `fixed_actions/fixed_action_mask` is rejected.
2. The replanning script runs with a VITRA checkpoint.
3. K=5 and K=10 replans are saved separately.
4. Prefix preservation error is near zero.
5. Suffix region violation decreases for at least one guidance scale.
6. The trajectory plot clearly shows that only the unexecuted future suffix changes.
7. Runtime overhead is reported per replanning call.

## 11. Practical Notes

This is still a toy closed-loop experiment. If we reuse the same image and state at K=5
and K=10, the only new signal is the guidance loss. For a real robot setting, the replan
call should use the latest observation and current state.

The current repo's GigaHands fine-tuning/evaluation mainly works in normalized action
space. The toy guidance should also start in normalized space to avoid unit mismatch.

The polynomial region should target active hand/action dimensions. For the current setup,
`guide_dims=[51, 52]` is a safer default than `[0, 1]`.

The first implementation should use deterministic DDIM (`eta=0`) and fixed seeds. After
the basic mechanism is correct, we can sweep:

```text
guidance_scale
num_ddim_steps
replan_indices
guide_dims
region radius / center / Q
```

## 12. Implementation Order

1. Add tests for prefix clamping and suffix-only guidance.
2. Add `ddim_sample_loop_replanning_guided(...)`.
3. Thread `fixed_actions` and `fixed_action_mask` through `DiffusionPolicy.sample(...)`.
4. Thread the same arguments through `VITRA_Paligemma.predict_action(...)`.
5. Add `scripts/inference_guided_replanning_toy.py`.
6. Run a smoke test with a small DDIM step count.
7. Run the full toy demo and save plots/metrics.

## 13. Relationship to Previous Guidance Plan

The previous guidance experiment answered:

```text
Can we bias one diffusion generation call toward a polynomial region?
```

This replanning experiment answers:

```text
After part of a generated action chunk has already been executed, can we regenerate only
the remaining future actions while preserving the executed prefix?
```

That distinction matters for test-time control. A real signal received at `0.33s` cannot
change actions that were already executed from `0s` to `0.33s`. It can only affect the
future chunk generated during the new replanning call.
