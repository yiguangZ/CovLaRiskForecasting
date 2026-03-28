import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import av
import fsspec
import json
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
from datasets import load_dataset, Video
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

# ---- Config ----
MODEL_NAME = os.path.expanduser("~/covla_project/models/qwen3vl32b")
NUM_SCENES = 300
SAVE_PATH = os.path.expanduser("~/covla_project/results.json")
STATES_DIR = os.path.expanduser("~/covla_project/covla_data/states")
MERGED_LABELS_PATH = os.path.expanduser("~/covla_project/triggers_mechanisms_merged.jsonl")

past_second = 4
future_second = 5
target_freq = 4
source_freq = 20
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
- Mechanism Selection Priority: When multiple mechanisms are supported by evidence, select the most specific, actionable mechanism that directly describes the likely near-future conflict. For example:
  - If brake lights are visible and ego is following closely → use hard_brake_likely or rear_end_risk, not visibility_limited_uncertainty.
  - If road is wet and vehicle is braking → use low_friction_weather or hard_brake_likely, not visibility_limited_uncertainty.
  - If road is narrow and vehicles are close → use narrow_road_constraint or narrow_distance_risk, not visibility_limited_uncertainty.
  - visibility_limited_uncertainty should be used ONLY when no more specific mechanism is clearly supported and the scene contains multiple overlapping risk factors (e.g., occlusion + dense traffic + poor lighting) that create high uncertainty.
- Future-Facing Requirement: The forecast_mechanisms must reflect specific, likely future conflicts based on observed motion patterns and scene dynamics, not general scene descriptions. For example:
  - Incorrect: "low visibility due to rain" → this is a scene description.
  - Correct: "hydroplaning likely due to wet road and high speed" → this is future-facing.
  - Incorrect: "dense traffic ahead" → this is a scene description.
  - Correct: "sudden braking likely due to lead vehicle decelerating in congested flow" → this is future-facing.
- Consistency with Evidence: Each chosen mechanism must be directly grounded in observable cues from the past 4 seconds. Do not infer unsupported events.
- Level 1 = uncertainty/complexity without a specific active or emerging conflict → Use visibility_limited_uncertainty only when no specific conflict is imminent and the scene is complex due to multiple overlapping risk factors.
- Level 2 = a specific interacting agent or motion pattern indicates a likely conflict in the next 1-3s → Use specific mechanisms like rear_end_risk, hard_brake_likely, low_friction_weather, etc.
- Level 3 = strong evidence of imminent collision threat or emergency conflict in the next 1-3s → Use mechanisms like collision_avoidance_risk, sudden_appearance_risk, or occluded_intersection_conflict when emergency action is likely required.

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

Near-future risk mechanisms (forecast_mechanisms) — output your TOP 3 most likely mechanisms, ranked from most to least likely, each chosen from this list:
- none: No foreseeable risk in the next 1-3s based on observed cues.
- narrow_road_constraint: The road geometry is physically constrained (e.g., narrow lanes, no shoulder, tight curves), limiting maneuverability and increasing risk of lateral or rear-end conflict.
- pedestrian_crossing_conflict: A pedestrian is actively crossing or about to cross the ego's path, with high likelihood of collision if ego proceeds without yielding.
- cyclist_conflict: A cyclist is in or entering the ego's path with insufficient clearance or unpredictable motion, posing a direct collision risk.
- visibility_limited_uncertainty: The scene contains multiple overlapping risk factors (e.g., occlusion, dense traffic, poor lighting) that collectively create high uncertainty about future agent behavior, but no single specific conflict is clearly imminent. Use ONLY when no more specific mechanism is supported.
- low_visibility_weather: Weather conditions (e.g., rain, fog, snow, dusk) significantly reduce visibility, increasing risk of collision due to delayed detection of hazards or braking.
- hard_brake_likely: Observed brake lights, sudden deceleration, or traffic flow patterns indicate a high probability of abrupt braking by a lead vehicle in the next 1-3s.
- rear_end_risk: The ego vehicle is following closely behind a vehicle that is slowing or stopped, with insufficient stopping distance to avoid collision if the lead vehicle brakes suddenly.
- large_vehicle_exposure_context: The ego vehicle is in close proximity to a large vehicle (truck, bus, etc.) that poses increased risk due to size, blind spots, or longer stopping distances.
- pedestrian_exposure_context: A pedestrian is present near the roadway (e.g., sidewalk, crosswalk) and may enter the roadway, creating a potential conflict.
- in_tunnel_context: The ego vehicle is in or approaching a tunnel, where lighting, visibility, and confined space increase risk.
- sudden_traffic_change_risk: Traffic flow is unstable (e.g., lane changes, merging, erratic behavior) with high likelihood of unexpected maneuvers by nearby vehicles.
- low_friction_weather: Wet, icy, or slippery road conditions reduce tire grip, increasing risk of skidding, hydroplaning, or loss of control.
- cut_in_conflict: A vehicle is actively or about to cut into the ego's lane with insufficient clearance, creating a lateral collision or hard-braking risk.
- sudden_appearance_risk: A vehicle, pedestrian, or obstacle is likely to appear suddenly from an occluded area (e.g., blind corner, behind a vehicle) into the ego's path.
- occluded_intersection_conflict: The ego vehicle is approaching an intersection where visibility is blocked (e.g., by vehicles, curves, vegetation), and conflicting traffic or pedestrians may enter unexpectedly.
- lane_merge_conflict: Vehicles are merging into the ego's lane from an adjacent lane or ramp, with insufficient clearance or timing, creating a collision risk.
- collision_avoidance_risk: The ego vehicle must take evasive action (braking, steering) to avoid an imminent collision with a vehicle or obstacle in its path.
- narrow_distance_risk: The ego vehicle is traveling in close proximity to another vehicle or obstacle with minimal lateral or longitudinal clearance, increasing risk of contact.
- occlusion_or_blind_spot: A vehicle or object is blocking the ego's forward or side view, creating a blind spot where a hazard may appear unexpectedly.
- truck_right_side_conflict: A truck or large vehicle on the right side of the ego vehicle poses a lateral conflict risk due to wide turns, blind spots, or lane encroachment.
- green_light_miss_risk: A traffic signal is turning or has turned green but a vehicle ahead has not yet moved, creating a rear-end or stop-and-go risk.
- unexpected_events_risk: An unpredictable or anomalous event (e.g., sudden obstacle, erratic agent behavior) creates an unclassifiable but clear near-future hazard.
- other: Any other risk mechanism not listed above, supported by clear evidence.

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
  (c) non-duplicative with any primary trigger or forecast_mechanisms
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
  "forecast_mechanisms": ["most_likely", "second_likely", "third_likely"],
  "evidence_phrase": "short future-facing risk phrase predicted for the next 1-3s, grounded in past 4s observations",
  "forecast_horizon": "next 1-3s",
  "explanation": "2-3 sentences linking observed cues in the past 4s to the predicted near-future risk. Max 80 words"
}
Output valid JSON only. No markdown, no extra text. Do not output reasoning or thinking text.
"""

def load_video_frames(scene):
    video_path = scene["video"]["path"]
    frames = []
    with fsspec.open(video_path, "rb") as f:
        container = av.open(f)
        try:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
        finally:
            container.close()
    video_np = np.stack(frames, axis=0)
    video_np = np.transpose(video_np, (0, 3, 1, 2))
    return torch.from_numpy(video_np)

def frame_to_pil(frame_tensor):
    img = frame_tensor.cpu()
    if img.max() <= 1.0:
        img = img * 255
    img_array = img.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(img_array)

# ---- Load model ----
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print("Model loaded")

# ---- Load target video IDs from partner's merged labels ----
target_video_ids = set()
with open(MERGED_LABELS_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            target_video_ids.add(e['video_id'])
        except:
            continue
print(f"Targeting {len(target_video_ids)} labeled video IDs from partner's labels")

# ---- Load dataset, only keeping scenes that match partner's video IDs ----
print("Loading dataset...")
dataset = load_dataset(
    "turing-motors/CoVLA-Dataset",
    split="train",
    streaming=True
)
dataset = dataset.cast_column("video", Video(decode=False))

scenes = []
for idx, scene in enumerate(dataset):
    if scene['video_id'] in target_video_ids:
        scenes.append(scene)
        if len(scenes) >= NUM_SCENES:
            break
print(f"Loaded {len(scenes)} matching scenes")

# ---- Main loop ----
random.seed(42)
results = []

for scene_idx, scene in enumerate(scenes):
    print(f"\n--- Scene {scene_idx+1}/{NUM_SCENES} | video_id: {scene['video_id']} ---")

    try:
        jsonl_file_path = os.path.join(STATES_DIR, f"{scene['video_id']}.jsonl")
        with open(jsonl_file_path, "r") as f:
            state_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"  State file not found, skipping")
        continue

    try:
        video_frames = load_video_frames(scene)
        print(f"  Video loaded: {video_frames.shape}")
    except Exception as e:
        print(f"  Video load failed: {e}, skipping")
        continue

    positions_ecefs = []
    orientations_ecefs = []
    for i in range(len(state_data)):
        positions_ecefs.append(state_data[i][f'{i}']['positions_ecef'])
        orientations_ecefs.append(state_data[i][f'{i}']['orientations_ecef'])
    positions_ecefs = np.array(positions_ecefs)
    orientations_ecefs = np.array(orientations_ecefs)

    # Sample ego_frame aligned to GT label windows (even seconds * 20fps)
    ego_frame = random.choice([80, 120, 160, 200, 240, 280, 320, 360, 400])

    # Need 4 seconds back (4*20=80 frames) for images
    if ego_frame - 4*source_freq < 0 or ego_frame >= len(video_frames):
        print(f"  Frame out of bounds, skipping")
        continue

    end_idx = ego_frame + int(future_second * source_freq)
    start_idx = ego_frame - int(past_second * source_freq)
    step = int(source_freq / target_freq)
    indices = np.arange(start_idx, end_idx, step)

    if indices[-1] >= len(positions_ecefs):
        print(f"  Trajectory index out of bounds, skipping")
        continue

    target_pos_ecef = positions_ecefs[indices, :]
    origin_pos_ecef = positions_ecefs[ego_frame, :]
    origin_orient_ecef = orientations_ecefs[ego_frame, :]
    delta_pos_ecef = target_pos_ecef - origin_pos_ecef
    r_ego_to_ecef = R.from_euler('xyz', origin_orient_ecef)
    r_ecef_to_ego = r_ego_to_ecef.inv()
    trajectory_local = r_ecef_to_ego.apply(delta_pos_ecef)
    trajectory_local[:, 1] = -trajectory_local[:, 1]
    xy = trajectory_local[:, :2]
    past_trajectory = ", ".join([f"({x:.2f}, {y:.2f})" for x, y in xy[:past_second * target_freq]])

    label_prompt = TEMPLATE.replace('{PAST_TRAJ_STRING}', past_trajectory)

    # Correctly sample 5 frames spanning t-4s to t (1 second apart each)
    frames = [
        frame_to_pil(video_frames[ego_frame - 4*source_freq]),  # t-4s
        frame_to_pil(video_frames[ego_frame - 3*source_freq]),  # t-3s
        frame_to_pil(video_frames[ego_frame - 2*source_freq]),  # t-2s
        frame_to_pil(video_frames[ego_frame - 1*source_freq]),  # t-1s
        frame_to_pil(video_frames[ego_frame]),                  # t
    ]

    messages = [{"role": "user", "content": [
        {"type": "image", "image": frames[0]},
        {"type": "image", "image": frames[1]},
        {"type": "image", "image": frames[2]},
        {"type": "image", "image": frames[3]},
        {"type": "image", "image": frames[4]},
        {"type": "text", "text": label_prompt}
    ]}]

    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512,
                                       temperature=0.1, do_sample=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    try:
        clean = output.strip().replace("```json", "").replace("```", "").strip()
        result_json = json.loads(clean)
    except:
        result_json = {"raw_output": output, "parse_error": True}

    results.append({
        "scene_idx": scene_idx,
        "scene_id": scene['video_id'],
        "ego_frame": ego_frame,
        "past_trajectory": past_trajectory,
        "output": result_json
    })

    pred_mechanisms = result_json.get('forecast_mechanisms', ['PARSE ERROR'])
    print(f"  risk_level = {result_json.get('risk_level', 'PARSE ERROR')} | mechanisms = {pred_mechanisms}")

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)

print(f"\nDone. Saved {len(results)} results to {SAVE_PATH}")
