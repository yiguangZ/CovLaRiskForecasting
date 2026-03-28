# CoVLA Driving Risk Forecasting with ProTeGi Prompt Optimization

This repository contains the inference, evaluation, and prompt optimization pipeline for near-future driving risk forecasting on the [CoVLA dataset](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset) using **Qwen3-VL-32B**. Prompt quality is iteratively improved using **ProTeGi** (Prompt Optimization with Textual Gradients).

**ECE 594 Project — UC Santa Barbara**
Authors: Yiguang Zhu, Mengyao Li

---

## Results

Evaluated on 206 clips from the CoVLA dataset. Top-3 multi-label mechanism prediction compared against partner's hand-written baseline prompt.

| Metric | Baseline | ProTeGi Improved | Change |
|--------|----------|-----------------|--------|
| Precision | 0.385 | 0.368 | ↓ 4.4% |
| Recall | 0.234 | 0.328 | ↑ 40.2% |
| F1 | 0.291 | 0.345 | ↑ 19.2% |

> **Evaluation method:** Multi-label top-3 prediction. For each clip, the model predicts its 3 most likely risk mechanisms. A clip is correct if any of the 3 predictions matches any label in the ground truth mechanism list.

---

## Pipeline Overview

```
infer.py → results.json → match.py → errors.json → protegi.py → improved prompt
                        ↓
                     eval.py → Precision / Recall / F1
```

### `infer.py`
Runs batched inference on CoVLA video clips using Qwen3-VL-32B. For each clip it:
- Loads 5 front-view frames spanning the past 4 seconds (t-4s, t-3s, t-2s, t-1s, t)
- Loads the 4-second past ego trajectory (16 steps at 4 Hz) from ECEF state files
- Passes frames + trajectory + prompt template to the model
- Outputs a JSON prediction containing risk level (0-3), top-3 forecast mechanisms, trigger tags, and explanation

Results are saved incrementally to `results.json` after every clip.

**Key config:**
```python
NUM_SCENES = 206          # number of clips to run
CUDA_VISIBLE_DEVICES = "0,1"
MODEL_NAME = "~/covla_project/models/qwen3vl32b"
```

---

### `match.py`
Matches inference results against ground truth labels from the partner's merged trigger/mechanism file (`triggers_mechanisms_merged.jsonl`). For each result:
- Looks up ground truth by `{video_id}_s{second}`, registering every second in `win_secs` as a valid key for maximum match rate
- Checks if the top-1 predicted mechanism appears in the GT mechanism list
- Saves mismatches to `errors.json` for downstream ProTeGi analysis

Outputs:
- Match rate (how many clips found a GT entry)
- Top-3 mechanism accuracy
- Per-clip error breakdown printed to stdout
- `errors.json` with full mismatch details

---

### `eval.py`
Computes multi-label Precision, Recall, and F1 over all matched clips using the same metrics as the partner baseline for direct comparison.

For each clip:
- `TP` = predicted mechanisms that appear in GT
- `FP` = predicted mechanisms that do NOT appear in GT
- `FN` = GT mechanisms not covered by any prediction

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * P * R / (P + R)
```

---

### `protegi.py`
Runs one ProTeGi iteration using Qwen3-VL-32B in text-only mode:
1. **Gradient generation** — feeds error cases to the model and asks it to identify 3 root causes of prompt failure
2. **Prompt editing** — uses the gradient to rewrite the Mechanism definitions, Boundary rules, and Near-future risk mechanism sections of the prompt
3. Saves gradient + improved prompt sections to `protegi_output.txt`

The improved sections are then manually applied to the `TEMPLATE` in `infer.py` before the next inference run.

---

## Dependencies

```bash
pip install torch transformers datasets accelerate
pip install av fsspec scipy pillow numpy
pip install qwen-vl-utils
```

**Model:** Qwen3-VL-32B (~66.7GB, bfloat16). Requires 2× NVIDIA A6000 (49GB each) or equivalent.

**Dataset:** [turing-motors/CoVLA-Dataset](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset) — streamed via HuggingFace `datasets`.

**Ground truth labels:** `triggers_mechanisms_merged.jsonl` — partner-generated multi-label trigger and mechanism annotations for 783 CoVLA videos.

**State files:** Per-video JSONL files containing ECEF positions and orientations at 20 Hz, used to compute ego trajectory. Place in `~/covla_project/covla_data/states/`.

---

## Usage

```bash
# 1. Run inference
python3 infer.py

# 2. Match predictions to ground truth
python3 match.py

# 3. Compute Precision / Recall / F1
python3 eval.py

# 4. Run ProTeGi on errors to improve prompt
python3 protegi.py
# → apply improved sections from protegi_output.txt to TEMPLATE in infer.py
# → repeat from step 1
```

---
## Example of Output
<img width="606" height="430" alt="image" src="https://github.com/user-attachments/assets/7308e82a-1606-416c-9839-e8bfa7686d23" />
pred_triggers:  ['cyclist','nearby_vehicle','pedestrian','traffic_element','weather_or_low_visibility']
pred_mechanisms: ['cyclist_conflict','hard_brake_likely','low_visibility_weather','pedestrian_crossing_conflict']
explanation: Pedestrians and a cyclist are crossing the intersection directly ahead, with the ego vehicle approaching at a steady pace. The overcast weather reduces visibility, increasing collision risk. The leading vehicle is braking, indicating a stop is required. Immediate hard braking is necessary to avoid collision in the next 1-3 seconds.

## ProTeGi Iterations

**Iteration 1 — Root causes identified:**
1. Overuse of `visibility_limited_uncertainty` as a generic fallback (~80% of predictions)
2. No mechanism priority rules — model picks broad over specific when both are supported
3. Vague future-facing requirement — model describes current scene instead of forecasting conflict

**Fixes applied:** Added explicit if/then priority rules, rewrote all mechanism definitions as future conflicts, restricted `visibility_limited_uncertainty` to only when no specific mechanism is supported.

**Iteration 2 — New biases identified:**
- `occluded_intersection_conflict` over-predicted as a new fallback
- Confusion between `hard_brake_likely` and `rear_end_risk`

**Fix applied:** Switched from top-1 to top-3 mechanism prediction to better match multi-label GT structure, scaled evaluation to 206 clips.

---

## Dataset

CoVLA (Corpus of Vision-Language for Autonomous driving) is a large-scale Japanese driving dataset featuring left-hand traffic scenarios. Ground truth labels provide multi-label risk triggers and mechanisms per 3-second temporal window.

- **Videos evaluated:** 206 clips
- **Label source:** `triggers_mechanisms_merged.jsonl` (783 labeled videos, 19,574 labeled windows)
- **Traffic:** Left-hand traffic (Japan) — oncoming vehicles approach from the right side of the frame



