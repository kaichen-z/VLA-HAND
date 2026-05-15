# DPS-Style Tactile-Guided Action Diffusion for VITRA + OpenTouch

## 0. Goal

This document specifies a test-time tactile guidance experiment for VITRA-style action diffusion using our OpenTouch data. The target setting is online action regeneration at physical times such as `0.33s` and `0.66s`: VITRA has already generated a 16-step action chunk at time `0s`, some prefix has been executed, new tactile observations arrive, and we regenerate only the unexecuted suffix using a DPS-style guidance gradient.

The method is inspired by Diffusion Posterior Sampling (DPS):

- Paper: *Diffusion Posterior Sampling for General Noisy Inverse Problems*
- Reference repo: <https://github.com/DPS2022/diffusion-posterior-sampling>
- Optional local reference checkout: `thirdparty/DPS2022-diffusion-posterior-sampling`

The key analogy is:

| DPS image inverse problem | Our tactile action-editing problem |
|---|---|
| sample is image `x` | sample is action chunk `a[0:H]` |
| measurement is degraded image `y` | measurement is online tactile observation/history |
| forward operator `A(x)` is known | forward operator `F_phi(a, state, obs)` is learned |
| guidance edits image sample during diffusion | guidance edits action suffix during diffusion |

The important deployment constraint is that guidance at `0.33s` cannot change actions already executed before `0.33s`. The same applies at `0.66s`.

---

## 1. Current Project Context

### 1.1 VITRA / GigaHands side

Our VITRA setup predicts human hand action chunks. In the recent GigaHands/OpenTouch experiments:

- action chunk length: `16`
- action dimension: `192`
- state dimension: `212`
- action frequency used in evaluation: `8 Hz`
- edit times:
  - `0.33s -> round(0.33 * 8) = action index 3`
  - `0.66s -> round(0.66 * 8) = action index 5`

The current residual touch-editor experiments use frozen VITRA predictions cached from the step-140000 GigaHands-finetuned checkpoint. Those experiments show that residual editors can reduce action MSE, but matched-vs-shuffled touch gaps are still the key measure for true tactile dependence.

### 1.2 OpenTouch side

The converted OpenTouch dataset is expected under this repo:

```text
datasets/vitra_opentouch_keypoint_full_text_aligned
```

It contains:

```python
action_list:          [16, 192]
action_mask:          [16, 192]
current_state:        [212]
current_state_mask:   [212]
touch_pressure:       [16, 2, 16, 16]
touch_mask:           [16, 2]
chunk_phase:          [16]
```

The two tactile channels are the two pressure maps in the converted OpenTouch representation. The current offline eval cache also stores:

```python
a_base:               [16, 192]  # frozen VITRA prediction
a_target:             [16, 192]  # OpenTouch ground-truth action
future_mask:          [16, 192]
edit_start_idx:       scalar
```

For DPS-style diffusion guidance, use the same normalized action/state convention as VITRA and the same OpenTouch train/test split.

---

## 2. What This Method Should Do

At `t=0s`, VITRA samples an initial action chunk:

```text
a_base[0:16]
```

At `t=0.33s`, we assume actions up to index `3` have been executed or are no longer editable. We receive tactile observations up to index `3`:

```text
touch_obs[0:4]
```

Then we run a short diffusion regeneration process initialized from the previous action chunk. The regeneration uses DPS-style tactile guidance and only edits:

```text
action indices >= 3
```

At `t=0.66s`, we repeat the process, now preserving the prefix up to index `5` and editing only:

```text
action indices >= 5
```

This is different from a one-shot residual editor. Here, the action suffix is regenerated through a diffusion denoising process, and tactile consistency enters as a gradient during sampling.

---

## 3. Required Components

The experiment should be integrated modularly into VLA-HAND. Do not rewrite the base VITRA policy. Add a wrapper around the action diffusion sampler.

Recommended layout:

```text
VLA-HAND/
  thirdparty/
    DPS2022-diffusion-posterior-sampling/
  vitra/
    guidance/
      tactile_dps.py
      tactile_forward_model.py
      tactile_losses.py
      tactile_replay_dataset.py
  scripts/
    train_tactile_forward_model.py
    evaluate_tactile_dps_replay.py
    evaluate_tactile_dps_diffusion_replanning.py
    run_tactile_dps_smoke.py
  configs/
    tactile_dps_opentouch.json
  doc/
    test-time-guidance-dps/
      dps_style_tactile_guided_action_diffusion.md
```

The third-party DPS repo is a reference implementation only. We should not vendor its training code into our runtime path. Use it to verify the DPS update pattern: clean estimate, measurement loss, gradient w.r.t. sample, posterior update. The runtime path in this repo is `scripts/evaluate_tactile_dps_diffusion_replanning.py`, which applies the tactile measurement loss inside VITRA's prefix-clamped DDIM replanning sampler.

---

## 4. Tactile Encoder + Forward Model

DPS needs a differentiable measurement operator. For OpenTouch, use a two-part learned measurement model:

```text
E_psi(touch_pressure, touch_mask)
    -> tactile embedding z_touch and tactile statistics s_touch

F_phi(current_state, candidate_action_segment, chunk_phase, optional_touch_history)
    -> predicted tactile embedding z_pred and predicted tactile statistics s_pred
```

The tactile encoder is the stronger version of the plan. Instead of forcing the action model to predict the full `2 x 16 x 16` pressure image, the encoder learns a compact tactile representation from the real OpenTouch pressure maps. The action-conditioned forward model then predicts that representation from candidate actions.

Keep low-dimensional tactile statistics as an auxiliary target. They make training easier to debug and provide interpretable metrics:

```python
touch_stats[t] = {
    "valid_left": touch_mask[t, 0],
    "valid_right": touch_mask[t, 1],
    "mean_abs_left": mean(abs(touch_pressure[t, 0])),
    "mean_abs_right": mean(abs(touch_pressure[t, 1])),
    "max_abs_left": max(abs(touch_pressure[t, 0])),
    "max_abs_right": max(abs(touch_pressure[t, 1])),
    "delta_mean_left": mean(abs(touch_pressure[t, 0] - touch_pressure[t0, 0])),
    "delta_mean_right": mean(abs(touch_pressure[t, 1] - touch_pressure[t0, 1])),
}
```

The embedding should carry richer contact information than mean/max pressure, while the statistics prevent the model from learning an opaque latent that is hard to validate.

### 4.1 Training tuple

For each OpenTouch cached sample and each edit index `k in {3, 5}`:

```python
input = {
    "current_state": current_state,                 # [212]
    "action_segment": a_target[k:k+h],              # [h, 192]
    "touch_history": touch_pressure[:k+1],          # [k+1, 2, 16, 16]
    "touch_history_mask": touch_mask[:k+1],         # [k+1, 2]
    "chunk_phase": chunk_phase[k:k+h],              # [h]
}
target = {
    "future_touch": touch_pressure[k:k+h],          # [h, 2, 16, 16]
    "future_touch_mask": touch_mask[k:k+h],         # [h, 2]
    "future_touch_stats": touch_stats[k:k+h],       # [h, stat_dim]
}
```

Use `a_target` for training the forward model, not `a_base`, because the forward model should learn how real action/hand motion relates to real tactile observations.

### 4.2 Tactile encoder

Use a small CNN encoder per timestep, followed by a temporal encoder. The pressure map is small, so the model should stay lightweight:

```python
class TactileEncoder(nn.Module):
    def __init__(self, embed_dim=64, stat_dim=8, hidden_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.frame_proj = nn.Linear(64, hidden_dim)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=4, batch_first=True),
            num_layers=2,
        )
        self.embed_head = nn.Linear(hidden_dim, embed_dim)
        self.stat_head = nn.Linear(hidden_dim, stat_dim)

    def forward(self, touch_pressure, touch_mask):
        # touch_pressure: [B, h, 2, 16, 16]
        # touch_mask: [B, h, 2]
        b, h = touch_pressure.shape[:2]
        x = touch_pressure.reshape(b * h, 2, 16, 16)
        hidden = self.cnn(x).flatten(1)
        hidden = self.frame_proj(hidden).reshape(b, h, -1)
        hidden = self.temporal(hidden)
        return {
            "embedding": self.embed_head(hidden),   # [B, h, embed_dim]
            "stats": self.stat_head(hidden),        # [B, h, stat_dim]
        }
```

The encoder can be trained jointly with the forward model, but the first stable implementation should use a two-stage setup:

1. Train `E_psi` to reconstruct tactile statistics and optionally a lightly reconstructed pressure map.
2. Freeze or slowly update `E_psi`.
3. Train `F_phi` to predict `E_psi(future_touch)` from action/state.

This avoids a degenerate solution where the encoder and forward model co-adapt to an embedding that does not represent tactile information.

### 4.3 Action-conditioned forward model

Use a small temporal model that can run inside the diffusion sampling loop:

```python
class TactileForwardModel(nn.Module):
    def __init__(self, action_dim=192, state_dim=212, embed_dim=64, stat_dim=8, hidden_dim=256):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.phase_proj = nn.Linear(1, hidden_dim)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True),
            num_layers=2,
        )
        self.embed_head = nn.Linear(hidden_dim, embed_dim)
        self.stat_head = nn.Linear(hidden_dim, stat_dim)

    def forward(self, current_state, action_segment, chunk_phase):
        # current_state: [B, 212]
        # action_segment: [B, h, 192]
        # chunk_phase: [B, h]
        state = self.state_proj(current_state)[:, None, :]
        hidden = self.action_proj(action_segment) + state + self.phase_proj(chunk_phase[..., None])
        hidden = self.temporal(hidden)
        return {
            "embedding": self.embed_head(hidden),
            "stats": self.stat_head(hidden),
        }
```

### 4.4 Forward-model loss

Train the forward model with both embedding and statistics supervision:

```text
touch_target = E_psi(future_touch, future_touch_mask)
touch_pred   = F_phi(current_state, a_target[k:k+h], chunk_phase[k:k+h])
valid_step_mask = any(future_touch_mask > 0, dim=-1)

L_forward =
    lambda_embed * masked_mse(touch_pred.embedding, stopgrad(touch_target.embedding), valid_step_mask)
  + lambda_stats  * masked_mse(touch_pred.stats, future_touch_stats, valid_step_mask)
```

Use `stopgrad(touch_target.embedding)` if the encoder is frozen. If the encoder is jointly trained, keep an auxiliary statistics/reconstruction loss on `E_psi` so the latent remains tactile-grounded:

```text
L_encoder =
    masked_mse(touch_target.stats, future_touch_stats, valid_step_mask)
  + lambda_recon * masked_mse(reconstructed_touch, future_touch, future_touch_mask)
```

The first implementation should set `lambda_recon = 0` unless tactile-map reconstruction is needed for debugging.

---

## 5. DPS Guidance Loss

At edit index `k`, use the observed tactile history up to `k` as the online measurement. The clean action estimate from the current diffusion state is:

```text
a_hat_0 = clean_estimate(a_s, model_output, scheduler_state)
```

Extract the future segment:

```python
action_segment = a_hat_0[:, k:k+h, :]
```

Predict tactile:

```python
pred = F_phi(current_state, action_segment, chunk_phase[:, k:k+h])
```

Target:

```python
target = E_psi(touch_pressure[:, k:k+h], touch_mask[:, k:k+h])
valid_step_mask = torch.any(touch_mask[:, k:k+h] > 0, dim=-1)
```

Guidance loss:

```text
L_guide =
    lambda_embed * masked_mse(pred.embedding, target.embedding, valid_step_mask)
  + lambda_stats * masked_mse(pred.stats, touch_stats[:, k:k+h], valid_step_mask)
  + lambda_boundary * boundary_smoothness(a_hat_0, executed_prefix, k)
  + lambda_prior * action_prior_penalty(a_hat_0, a_old)
```

Use the prior penalty lightly. Its role is to prevent unrealistic jumps when tactile guidance is noisy:

```text
L_prior = ||a_hat_0[:, k:, :] - a_old[:, k:, :]||^2
```

Do not optimize model parameters during guidance. The only gradient target is the current noisy action sample `a_s`.

---

## 6. Prefix Preservation

Prefix preservation is mandatory. Use both gradient masking and hard overwrite.

For a chunk with shape `[B, 16, 192]` and edit index `k`:

```python
def future_suffix_mask(action, edit_idx):
    mask = torch.zeros_like(action)
    mask[:, edit_idx:, :] = 1.0
    return mask
```

Gradient masking:

```python
grad = grad * future_suffix_mask(a_s, k)
```

Hard overwrite after every diffusion step:

```python
a_prev[:, :k, :] = executed_prefix[:, :k, :]
```

For our current convention, `edit_start_idx = 3` means index `3` is editable, matching the residual-editor experiments. If the real robot controller has already executed action index `3` at exactly `0.33s`, change the mask to `k+1`. The offline replay should report which convention is used.

---

## 7. Regeneration Procedure

At online edit time, initialize from the old chunk using SDEdit-style noising:

```text
a_S = sqrt(alpha_bar_S) * a_old + sqrt(1 - alpha_bar_S) * eps
```

Then denoise for a short schedule while applying DPS guidance.

Recommended initial parameters:

```yaml
chunk_len: 16
action_dim: 192
action_frequency_hz: 8
edit_times_sec: [0.33, 0.66]
edit_indices: [3, 5]
num_edit_steps: [5, 10]
edit_noise_level: [0.15, 0.3]
lambda_tac: [0.3, 1.0, 3.0]
lambda_boundary: 0.05
lambda_prior: 0.01
guidance_scale: [0.01, 0.03, 0.1]
gradient_clip_norm: 1.0
```

`num_edit_steps=5` is important because previous timing experiments suggested that full VITRA regeneration can be expensive. A 0.33s online guidance method must either finish before the next control boundary or be framed as offline replay first.

---

## 8. Pseudocode

```python
def tactile_dps_regenerate(
    old_action_chunk,          # [B, 16, 192], normalized
    executed_prefix,           # [B, k, 192]
    current_state,             # [B, 212]
    action_mask,               # [B, 16, 192]
    chunk_phase,               # [B, 16]
    touch_target_stats,        # [B, h, touch_stat_dim]
    touch_target_mask,         # [B, h, touch_stat_dim]
    edit_idx: int,
    diffusion_model,
    scheduler,
    tactile_forward_model,
    num_edit_steps: int,
    edit_noise_level: float,
    lambda_tac: float,
    lambda_boundary: float,
    lambda_prior: float,
    guidance_scale: float,
):
    # 1. Initialize from previous chunk plus noise.
    noise = torch.randn_like(old_action_chunk)
    s_start = scheduler.timestep_from_fraction(edit_noise_level)
    a_s = scheduler.add_noise(old_action_chunk, noise, s_start)
    a_s[:, :edit_idx, :] = executed_prefix

    timesteps = scheduler.subsample_from(s_start, num_edit_steps)

    for s in timesteps:
        a_s = a_s.detach().requires_grad_(True)

        model_out = diffusion_model(a_s, s)
        a_hat_0 = scheduler.predict_x0(a_s, model_out, s)

        h = touch_target_stats.shape[1]
        action_segment = a_hat_0[:, edit_idx:edit_idx + h, :]
        phase_segment = chunk_phase[:, edit_idx:edit_idx + h]

        touch_pred = tactile_forward_model(
            current_state=current_state,
            action_segment=action_segment,
            chunk_phase=phase_segment,
        )

        loss_tac = masked_mse(touch_pred - touch_target_stats, touch_target_mask)
        loss_boundary = boundary_loss(a_hat_0, executed_prefix, edit_idx)
        loss_prior = masked_mse(
            a_hat_0[:, edit_idx:, :] - old_action_chunk[:, edit_idx:, :],
            action_mask[:, edit_idx:, :],
        )
        loss = lambda_tac * loss_tac + lambda_boundary * loss_boundary + lambda_prior * loss_prior

        grad = torch.autograd.grad(loss, a_s)[0]
        grad = grad * future_suffix_mask(a_s, edit_idx)
        grad = clip_batch_grad_norm(grad, max_norm=1.0)

        with torch.no_grad():
            prev = scheduler.step(model_out.detach(), s, a_s.detach()).prev_sample
            prev = prev - guidance_scale * grad
            prev[:, :edit_idx, :] = executed_prefix
        a_s = prev

    final = scheduler.final_clean_sample(a_s)
    final[:, :edit_idx, :] = executed_prefix
    return final
```

---

## 9. Offline Replay on OpenTouch

The first experiment should be offline replay, not robot deployment.

For each test cache sample:

1. Load `a_base`, `a_target`, `current_state`, `action_mask`, `touch_pressure`, `touch_mask`.
2. Use `a_base` as the initial generated chunk.
3. At edit index `3`, pretend prefix `a_base[:, :3]` has already executed.
4. Build tactile target from `touch_pressure[:, 3:3+h]` or from tactile history up to index `3`, depending on forward-model target design.
5. Run DPS-style regeneration for the suffix.
6. Optionally repeat at edit index `5` using the edited chunk from index `3`.
7. Compare against `a_target` on the editable suffix.

The offline replay should support:

```text
--edit_times 0.33
--edit_times 0.33 0.66
--ablation matched
--ablation shuffled_touch
--ablation zero_touch
--ablation no_guidance
```

The crucial test is whether matched tactile guidance beats shuffled tactile guidance. This is the same lesson from the residual touch-editor experiments.

---

## 10. Metrics

Report these metrics for each edit index:

```text
base_mse
guided_mse
improvement_pct
matched_vs_shuffled_gap
matched_vs_zero_gap
prefix_change_l2
boundary_jump_l2
tactile_prediction_mse
guidance_grad_norm
num_edit_steps
latency_ms
```

Definitions:

- `guided_mse`: MSE between guided action and `a_target` over editable suffix.
- `prefix_change_l2`: L2 change in non-editable prefix. Must be `0`.
- `matched_vs_shuffled_gap`: `shuffled_guided_mse - matched_guided_mse`.
- `matched_vs_zero_gap`: `zero_touch_guided_mse - matched_guided_mse`.
- `latency_ms`: wall-clock time for one regeneration at one edit index.

Success criteria for the first OpenTouch result:

```text
guided_mse < base_mse
matched_vs_shuffled_gap > 0
prefix_change_l2 == 0
latency is measured and reported
```

Do not claim tactile causality from MSE improvement alone. The matched-vs-shuffled gap is required.

---

## 11. Baselines

Compare against:

1. **No guidance**
   - Use `a_base` unchanged.

2. **Residual touch editor**
   - Use the existing OpenTouch residual editor results as the direct-editor baseline.
   - Current best tactile-sensitive variant: `contrastive_full`.

3. **DPS with matched touch**
   - Proposed method.

4. **DPS with shuffled touch**
   - Same sample/action target but tactile target from another sample.

5. **DPS with zero touch**
   - All tactile target/mask set to zero.

6. **DPS without prefix mask**
   - Diagnostic only. Expected to produce nonzero prefix drift.

7. **DPS without tactile loss**
   - Tests whether SDEdit/noising alone changes action MSE.

---

## 12. Third-Party DPS Reference

Use the DPS repo only as an algorithm reference:

```bash
cd /home/chonghej/scratch/chonghej/VLA-HAND
mkdir -p thirdparty
git clone https://github.com/DPS2022/diffusion-posterior-sampling \
  thirdparty/DPS2022-diffusion-posterior-sampling
```

Files to inspect after cloning:

```text
sample_condition.py
guided_diffusion/condition_methods.py
guided_diffusion/gaussian_diffusion.py
```

What to port conceptually:

- compute clean estimate during denoising;
- evaluate measurement consistency loss;
- take gradient w.r.t. current sample, not model parameters;
- apply guidance update inside the sampling loop.

What not to port directly:

- image-specific degradation operators;
- image-specific logging and datasets;
- assumptions about pixel-space dimensions.

---

## 13. Implementation Milestones

### Milestone 1: OpenTouch tactile target builder

Build a utility that converts `touch_pressure` and `touch_mask` into:

- low-dimensional tactile statistics;
- masked future tactile windows for edit indices `3` and `5`;
- train/test tuples for the tactile encoder and forward model.

Unit test it on synthetic pressure maps.

### Milestone 2: Tactile encoder

Train `E_psi` on OpenTouch train cache to produce tactile embeddings and reconstruct tactile statistics. Report train/test tactile-stat prediction MSE and check that shuffled pressure maps produce different embeddings from matched pressure maps.

### Milestone 3: Action-conditioned tactile forward model

Train `F_phi` on OpenTouch train cache using `a_target`, `current_state`, and `chunk_phase`. The target is `E_psi(future_touch)` plus tactile statistics. Report:

- embedding prediction MSE;
- tactile-stat prediction MSE;
- matched-vs-shuffled tactile prediction error;
- high-contact subset performance.

### Milestone 4: Diffusion sampler hook

Add a wrapper around the VITRA diffusion head that can:

- initialize from an existing action chunk plus noise;
- run a short denoising schedule;
- recover clean action estimates;
- keep gradients w.r.t. the noisy action variable.

### Milestone 5: DPS tactile guidance

Implement embedding guidance, statistics guidance, gradient masking, gradient clipping, and hard prefix overwrite.

### Milestone 6: Offline replay evaluation

Run OpenTouch test cache at edit index `3`, then at `[3, 5]`. Compare matched, shuffled, zero-touch, and no-guidance.

### Milestone 7: Latency report

Measure regeneration time for `num_edit_steps = 5` and `10`, including the tactile encoder and forward model calls. Report whether it can fit within the 0.33s replanning budget.

---

## 14. Main Risks and Mitigations

### Risk 1: The tactile encoder learns an unhelpful latent

If `E_psi` learns an embedding that does not preserve contact information, the DPS loss may look reasonable but fail to use tactile signal. Keep tactile-stat supervision, compare matched vs shuffled embeddings, and visualize nearest neighbors in embedding space.

### Risk 2: The forward model is weak

If `F_phi` cannot predict `E_psi(future_touch)` from actions, DPS guidance will be noisy. Start with short horizons, high-contact subsets, and the auxiliary statistics loss before relying on embedding-only guidance.

### Risk 3: Guidance improves MSE without using touch

Always report matched-vs-shuffled and matched-vs-zero gaps. MSE alone is not enough.

### Risk 4: Regeneration is too slow

Evaluate diffusion-only regeneration time separately from VITRA full inference. Use `num_edit_steps=5` as the first online target.

### Risk 5: Prefix drift

Use both gradient masking and hard overwrite. Fail the run if `prefix_change_l2` is nonzero.

### Risk 6: Action normalization mismatch

All guidance should happen in the normalized action space used by VITRA. Only unnormalize for visualization or human-readable metrics.

---

## 15. One-Sentence Method Summary

We treat online tactile feedback as a measurement for action diffusion: a learned OpenTouch tactile forward model predicts tactile statistics from candidate clean action chunks, and a DPS-style measurement gradient regenerates only the unexecuted suffix at `0.33s` and `0.66s` while preserving the executed prefix.
