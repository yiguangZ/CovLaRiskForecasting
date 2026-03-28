import os
import json

RESULTS_PATH = os.path.expanduser("~/covla_project/results.json")
MERGED_LABELS_PATH = os.path.expanduser("~/covla_project/triggers_mechanisms_merged.jsonl")

# Load ground truth
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
                ground_truth[key] = set(entry.get("mechanisms", []))
        except:
            continue

# Load results
with open(RESULTS_PATH) as f:
    results = json.load(f)

total_tp = 0
total_fp = 0
total_fn = 0

for result in results:
    scene_id = result['scene_id']
    ego_frame = result['ego_frame']
    key = f"{scene_id}_s{ego_frame // 20}"

    gt = ground_truth.get(key)
    if gt is None:
        continue

    preds = result['output'].get('forecast_mechanisms', ['none'])
    if not isinstance(preds, list):
        preds = [preds]
    pred_set = set(preds)

    tp = len(pred_set & gt)
    fp = len(pred_set - gt)
    fn = len(gt - pred_set)

    total_tp += tp
    total_fp += fp
    total_fn += fn

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Evaluated {len(results)} clips")
print(f"TP={total_tp}  FP={total_fp}  FN={total_fn}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1:        {f1:.3f}")
