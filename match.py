import os
import json

# ---- Config ----
RESULTS_PATH = os.path.expanduser("~/covla_project/results.json")
MERGED_LABELS_PATH = os.path.expanduser("~/covla_project/triggers_mechanisms_merged.jsonl")
ERRORS_PATH = os.path.expanduser("~/covla_project/errors.json")

# ---- Load results ----
with open(RESULTS_PATH) as f:
    results = json.load(f)
print(f"Loaded {len(results)} results")

# ---- Load ground truth from merged file ----
# Register every second in win_secs as a separate key for maximum match rate
ground_truth = {}
with open(MERGED_LABELS_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            video_id = entry["video_id"]
            for sec in entry["win_secs"]:
                key = f"{video_id}_s{sec}"
                ground_truth[key] = {
                    "video_id": video_id,
                    "win_start_s": entry["win_secs"][0],
                    "mechanisms": entry.get("mechanisms", []),
                    "triggers": entry.get("triggers", []),
                }
        except:
            continue
print(f"Loaded {len(ground_truth)} labeled entries")

# ---- Match results to ground truth ----
matched = 0
correct = 0
errors = []

for result in results:
    scene_id = result['scene_id']
    ego_frame = result['ego_frame']
    gt_second = ego_frame // 20
    key = f"{scene_id}_s{gt_second}"

    gt = ground_truth.get(key)
    if gt is None:
        continue

    matched += 1
    pred_mechanisms = result['output'].get('forecast_mechanisms', ['none'])
    if not isinstance(pred_mechanisms, list):
        pred_mechanisms = [pred_mechanisms]
    gt_mechanisms = gt['mechanisms']

    # Check if ANY of the top-3 predictions matches ANY GT mechanism
    is_correct = any(p in gt_mechanisms for p in pred_mechanisms)

    if is_correct:
        correct += 1
    else:
        errors.append({
            "scene_id": scene_id,
            "ego_frame": ego_frame,
            "gt_key": key,
            "model_output": {
                "risk_level": result['output'].get('risk_level', '?'),
                "forecast_mechanisms": pred_mechanisms,
                "trigger_tags": result['output'].get('trigger_tags', []),
                "explanation": result['output'].get('explanation', '')
            },
            "gt_mechanisms": gt_mechanisms,
            "gt_triggers": gt['triggers'],
            "what_went_wrong": f"Model predicted {pred_mechanisms} but GT mechanisms are {gt_mechanisms}"
        })

accuracy = correct / matched if matched > 0 else 0
print(f"\nMatched {matched} clips to ground truth")
print(f"Top-3 mechanism accuracy: {correct}/{matched} = {accuracy:.1%}")
print(f"\nErrors ({len(errors)} mismatches):")
for e in errors:
    print(f"  {e['gt_key']}: pred={e['model_output']['forecast_mechanisms']} | gt={e['gt_mechanisms']}")

# ---- Save errors ----
with open(ERRORS_PATH, "w") as f:
    json.dump(errors, f, indent=2)
print(f"\nSaved {len(errors)} errors to {ERRORS_PATH}")
