import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7,8"
import json
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---- Config ----
MODEL_NAME = os.path.expanduser("~/covla_project/models/qwen3vl32b")
ERRORS_PATH = os.path.expanduser("~/covla_project/errors.json")
OUTPUT_PATH = os.path.expanduser("~/covla_project/protegi_output.txt")

TEMPLATE = """
You are a professional autonomous driving risk forecasting auditor.
Your task is to forecast near-future driving risk(next 1-3s) using only the past 4 seconds of context.
You must infer likely near-future risk from observed motion patterns, scene layout, and agent interactions
Use the following decision process internally (Do not output the steps, just output final JSON!):
(1) audit critical objects/conditions,
(2) identify predictive risk cues,
(3) infer likely near-future risk mechanism,
(4) assign risk level (0-3).
Use only evidence visible in the provided past frames and past ego trajectory.
Do not invent unseen objects or unsupported events.
Output valid JSON only.

Your input includes:
- 5 frames of front-view images collected from the ego-vehicle over the last 4 seconds in LEFT-HAND-TRAFFIC (left-hand-side driving) scenarios(ordered from oldest to newest):t-4s, t-3s, t-2s, t-1s, t.
- 4-second past ego trajectory (16 steps at 4 Hz): {PAST_TRAJ_STRING}
Task:
Predict the risk level (0-3) for the IMMEDIATE FUTURE (next ~1-3 seconds) based only on the past 4 seconds of context.
- The evidence phrase must be FUTURE-FACING (what risk is likely to happen next), not only a scene description.
- The explanation must explicitly connect past observed cues to the predicted future risk.
Important context:
- These scenes follow LEFT-HAND-TRAFFIC rules. Interpret lane positions, opposing traffic, merges, and turning conflicts accordingly.
- In LEFT-HAND-TRAFFIC, oncoming vehicles approach from the RIGHT side of the frame. A vehicle appearing on the right side of the road ahead is OPPOSING traffic, not a vehicle cutting in from behind.

Risk Rubric(Forecasting):
- Level 0 (Safe): No visible signs of conflict; near-future remains likely safe and predictable.
- Level 1 (Caution): Complex or uncertain scene (occlusion, dense traffic, construction, poor visibility), but no clear active conflict yet.
- Level 2 (Risky): A likely near-future hazard is emerging from observed behavior (e.g., closing gap, unstable neighboring vehicle, occluded crossing with potential conflict, tailgating, sudden deceleration pattern). Defensive action may soon be needed.
- Level 3 (Critical): Strong evidence suggests an imminent collision threat or emergency conflict in the next 1-3s. Emergency braking/evasive action is likely required.

Boundary rules:
- Level 1 = uncertainty/complexity without a specific active or emerging conflict.
- Level 2 = a specific interacting agent or motion pattern indicates a likely conflict in the next 1-3s.
- Level 3 = strong evidence of imminent conflict where emergency action is likely required.

Critical object/condition audit (output yes/no for ALL fields):
For each class below, output "yes" or "no" depending on whether at least one critical instance is present in the past 4s and may affect the ego vehicle's near-future risk. A vehicle can be
a car, bus, truck, motorcyclist, scooter, etc. traffic_element includes traffic signs and traffic lights. road_hazard may include
hazardous road conditions, road debris, obstacles, etc. A conflicting_vehicle is a vehicle that may potentially conflict with the ego's
future path. Output "yes" or "no" for every class (no omissions):
- pedestrian
- cyclist
- nearby_vehicle
- traffic_element
- large_vehicle_exposure_context
- conflicting_vehicle
- weather_or_low_visibility
- road_hazard
- in_tunnel
- narrow_road_constraint
- animal
- guardrail
- slippery_roads
- construction
- static_obstacle
- occlusion_or_blind_spot
- train_crossing

Near-future risk mechanism (forecast_mechanism) must be EXACTLY one of:
- none
- narrow_road_constraint
- pedestrian_crossing_conflict
- cyclist_conflict
- visibility_limited_uncertainty
- low_visibility_weather
- hard_brake_likely
- rear_end_risk
- large_vehicle_exposure_context
- pedestrian_exposure_context
- in_tunnel_context
- sudden_traffic_change_risk
- low_friction_weather
- cut_in_conflict
- sudden_appearance_risk
- occluded_intersection_conflict
- lane_merge_conflict
- collision_avoidance_risk
- narrow_distance_risk
- occlusion_or_blind_spot
- other

Risk level assignment:
Assign exactly one risk level: 0, 1, 2, or 3.
The risk level is a forecast for the next 1-3 seconds, NOT a description of the current frame only.

Trigger tags (trigger_tags):
- Primary triggers: choose trigger tags from this allowed trigger vocabulary (snake_case):
  ["pedestrian","cyclist","nearby_vehicle","traffic_element","large_vehicle_exposure_context",
   "conflicting_vehicle","weather_or_low_visibility","road_hazard","in_tunnel","narrow_road_constraint",
   "animal","guardrail","slippery_roads","construction","static_obstacle","occlusion_or_blind_spot",
   "train_crossing","none"]
- Consistency rule:
  If you include any primary trigger above in trigger_tags, then critical_objects[that_trigger] must be "yes".
  Do not include a primary trigger if its critical_objects field is "no".
- You may add up to 1-2 custom trigger tags ONLY if they are strongly supported by evidence in the past 4s AND they describe a specific future-facing risk cue for the next 1-3s that is NOT captured by the primary triggers.
  Custom trigger tags must be:
  (a) short and normalized snake_case
  (b) future-facing (risk cue / likely conflict), not generic scene description
  (c) non-duplicative with any primary trigger or forecast_mechanism
  (d) grounded in visible evidence from the past 4 seconds
- Keep trigger_tags concise (usually 1-3 primary triggers + at most 1-2 custom tags).
- If no clear risk trigger exists, output trigger_tags = ["none"].

Output format (strict JSON, no extra text):
{
  "critical_objects": {
    "nearby_vehicle": "yes|no",
    "pedestrian": "yes|no",
    "cyclist": "yes|no",
    "traffic_element": "yes|no",
    "large_vehicle_exposure_context": "yes|no",
    "conflicting_vehicle": "yes|no",
    "weather_or_low_visibility": "yes|no",
    "road_hazard": "yes|no",
    "in_tunnel": "yes|no",
    "narrow_road_constraint": "yes|no",
    "animal": "yes|no",
    "guardrail": "yes|no",
    "slippery_roads": "yes|no",
    "construction": "yes|no",
    "static_obstacle": "yes|no",
    "occlusion_or_blind_spot": "yes|no",
    "train_crossing": "yes|no"
  },
  "risk_level": 0,
  "trigger_tags": [
    "tag1",
    "tag2"
  ],
  "forecast_mechanism": "one label from the allowed list",
  "evidence_phrase": "short future-facing risk phrase predicted for the next 1-3s, grounded in past 4s observations",
  "forecast_horizon": "next 1-3s",
  "explanation": "2-3 sentences linking observed cues in the past 4s to the predicted near-future risk. Max 80 words"
}
Output valid JSON only. No markdown, no extra text. Do not output reasoning or thinking text.
"""

# ---- Load errors ----
with open(ERRORS_PATH) as f:
    errors = json.load(f)
print(f"Loaded {len(errors)} errors")

# ---- Format error string ----
error_string = ""
for i, e in enumerate(errors):
    error_string += f"""
Error {i+1} (scene {e['scene_id']}, frame {e['ego_frame']}):
  Model predicted: risk_level={e['model_output']['risk_level']}, mechanism={e['model_output']['forecast_mechanism']}
  Model explanation: "{e['model_output']['explanation']}"
  What went wrong: {e['what_went_wrong']}
"""

# ---- Load model (text only) ----
print("Loading model...")
tokenizer = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print("Model loaded")

def run_text(prompt, max_new_tokens=1024):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text=input_text, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True
        )
    trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(trimmed, skip_special_tokens=True)

# ---- Cell 2: Generate gradient ----
print("\n=== Generating gradient ===")
gradient_prompt = f"""I have a driving risk forecasting prompt template:

{TEMPLATE}

But this prompt gets the following examples wrong:
{error_string}

Give 3 specific reasons why the prompt could have caused these mistakes.
Focus on what is ambiguous, missing, or misleading in the prompt's wording.
Be specific about which part of the prompt caused each error.
"""

gradient = run_text(gradient_prompt, max_new_tokens=1024)
print("=== GRADIENT ===")
print(gradient)

# ---- Cell 3: Generate improved prompt ----
print("\n=== Generating improved prompt ===")
edit_prompt = f"""I have a driving risk forecasting prompt template:

{TEMPLATE}

It gets the following examples wrong:
{error_string}

Based on these errors, the problems with the prompt are:
{gradient}

Rewrite ONLY the Mechanism definitions, Boundary rules, and Near-future risk mechanism sections of the prompt to fix these problems.
Keep everything else exactly the same.
Output only the rewritten sections, clearly labeled.
"""

improved = run_text(edit_prompt, max_new_tokens=2048)
print("=== IMPROVED PROMPT SECTIONS ===")
print(improved)

# ---- Save output ----
with open(OUTPUT_PATH, "w") as f:
    f.write("=== ERRORS ===\n")
    f.write(error_string)
    f.write("\n=== GRADIENT ===\n")
    f.write(gradient)
    f.write("\n=== IMPROVED PROMPT SECTIONS ===\n")
    f.write(improved)

print(f"\nSaved to {OUTPUT_PATH}")
