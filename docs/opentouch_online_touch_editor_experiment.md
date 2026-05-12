# Online OpenTouch Touch-Editor Experiment

## Goal

We implemented and evaluated an online touch-conditioned residual editor for VITRA action chunks. The setting is: VITRA generates a 16-step future action chunk at time 0, then at approximately 0.33 seconds we observe tactile feedback and edit only the remaining unexecuted suffix of the chunk.

The editor is intentionally not allowed to change the action prefix that would already have been executed.

## Training Setup

- Base policy: frozen VITRA checkpoint from the GigaHands keypoint-MANO finetuning run.
- Dataset: full converted OpenTouch dataset.
- Converted data size: 2,958 clips.
- Train cache: 50,000 randomly sampled OpenTouch windows.
- Test cache: 10,000 randomly sampled OpenTouch windows.
- Action target: normalized 16-step future hand keypoint action chunk.
- Edit time: 0.33 seconds, mapped to action index 3 at 8 FPS.
- Editor input: VITRA base action, current hand state, action mask, and tactile history observed up to the edit time.
- Editor output: residual action correction applied only to the suffix from the edit index onward.
- Loss: residual/action MSE on the editable suffix, with regularization on edit magnitude, smoothness, and mask behavior.

The expensive VITRA inference is cached first. Each cache sample stores the frozen VITRA action prediction, OpenTouch ground-truth future action, tactile pressure window, current state, masks, and edit index. This lets the touch editor train without repeatedly running the 3B VITRA model.

## Difference From the Original Main-Branch Prototype

The original touch-editing prototype was closer to a one-shot action editing setup. It could use a full future tactile window, which is not realistic for deployment because future touch measurements are not available at 0.33 seconds.

This branch changes the problem to causal online editing:

- The editor only receives tactile observations available up to the edit time.
- The edit is applied only to the future suffix of the action chunk.
- Evaluation reports prefix-change metrics to verify that already executed actions remain unchanged.
- Ablations compare matched touch, zero touch, shuffled touch, and future-touch oracle inputs.

## Evaluation Results

All results are on the 10,000-sample OpenTouch test cache.

| Editor / Evaluation | Base MSE | Edited MSE | Improvement |
|---|---:|---:|---:|
| matched editor + matched touch | 0.7610 | 0.2080 | 72.67% |
| matched editor + shuffled touch | 0.7610 | 0.2099 | 72.42% |
| matched editor + zero touch | 0.7610 | 0.6217 | 18.31% |
| high-contact editor + matched touch | 0.7610 | 0.3170 | 58.34% |
| high-contact editor + shuffled touch | 0.7610 | 0.3176 | 58.26% |
| zero-touch editor + zero touch | 0.7610 | 0.2040 | 73.20% |

For all evaluated settings, `prefix_change_l2 = 0.0`, which confirms that the editor did not modify the already executed prefix.

## Interpreting Shuffled Touch

`shuffled touch` is an ablation where the action sample and ground truth remain fixed, but the tactile signal is replaced with tactile data from another sample. It tests whether the editor depends on sample-specific tactile information.

If the model truly uses tactile input in a sample-specific way, matched touch should be clearly better than shuffled touch. In this experiment, matched touch and shuffled touch are almost identical:

- matched touch edited MSE: 0.2080
- shuffled touch edited MSE: 0.2099

This means the editor's large MSE reduction is real, but the evidence for sample-specific tactile conditioning is weak. The editor is likely learning a dataset-level residual prior that maps frozen VITRA actions toward OpenTouch ground-truth actions. Tactile input is helpful compared with a literal zero-touch input for the matched editor, but shuffled touch does not degrade performance enough to prove strong tactile causality.

## Conclusion

The current system works well as an action residual editor: it substantially reduces OpenTouch test MSE while preserving the executed action prefix. However, it is not yet a strong demonstration that high-frequency tactile feedback is being used in a sample-specific online-control sense.

The next step should be a stricter tactile-dependent benchmark, such as:

- train/test splits where touch patterns distinguish otherwise similar visual states,
- evaluation tasks where shuffled touch should necessarily produce the wrong correction,
- stronger contact-event targets around grasp, slip, press, lift, or release moments,
- or a contrastive/object-condition split that reduces the ability to solve the task from action priors alone.

## Reproduction Entry Point

The full experiment launcher is:

```bash
GPU=7 bash scripts/run_online_touch_editor_large.sh
```

Main outputs are written to:

```bash
runs/online_touch_editor_large/
```

The final result summary is:

```bash
runs/online_touch_editor_large/eval_summary.json
```
