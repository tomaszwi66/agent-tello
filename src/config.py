"""TELLO EXPLORER V1.2 - central configuration.

All tunables live here. No magic numbers in other modules.
Spec hard limits are enforced at clamp sites (tello_client.send_rc).
"""

LOOP_HZ: int = 20
LOOP_PERIOD_S: float = 1.0 / LOOP_HZ

COMMAND_TIMEOUT_MS: int = 600
COMMAND_LAND_TIMEOUT_MS: int = 3000

MIN_BATTERY_TAKEOFF_PCT: int = 30
MIN_BATTERY_ABORT_PCT: int = 15

MAX_FORWARD_SPEED_MPS: float = 0.40
RECOMMENDED_SPEED_MPS: float = 0.30
MAX_YAW_RATE_DPS: float = 35.0

# Tello rc_control: [-100, 100] %. EMPIRICAL deadzone is much larger than spec.
# Field observation: RC=13 produced ZERO forward motion (only drift). The Tello
# slow-flight deadzone is ~20+ for translation and ~30+ for yaw. We pin caps
# directly (overriding the linear-map calc) to break the deadzone while keeping
# flight slow & safe.
TELLO_MAX_LINEAR_MPS: float = 1.5
TELLO_MAX_YAW_DPS: float = 100.0
RC_FORWARD_CAP: int = 30   # slower again - must brake before crashing into wall
RC_LATERAL_CAP: int = 25
RC_VERTICAL_CAP: int = 0   # spec: no up/down flight
RC_YAW_CAP: int = 50       # yaw needs to be decisive

# Tello internal "speed" setting (cm/s, 10..100). RC channels are a % of this.
TELLO_SPEED_CMS: int = 50   # RC=30 is about 15 cm/s forward at this speed.

TARGET_STREAM_FPS: int = 10
FRAME_STALENESS_WARN_MS: int = 300

TELEMETRY_SAMPLE_HZ: int = 2
LOG_DIR: str = "logs"

# --- Stage 2 perception ---
# IMPORTANT: DEPTH_MODEL_ID must be pre-downloaded with internet BEFORE drone session.
# Run: python -c "from transformers import AutoImageProcessor, AutoModelForDepthEstimation; \
#   AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID); \
#   AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)"
#
# Relative model (DA V2 Small) - larger output = CLOSER (inverse depth).
DEPTH_MODEL_ID: str = "depth-anything/Depth-Anything-V2-Small-hf"
# Metric model - larger output = FARTHER (actual metres). Set DEPTH_MODEL_METRIC=True.
# !!! Verify exact HuggingFace ID before using - update this after checking HF !!!
DEPTH_MODEL_ID_METRIC: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
# DA3 non-metric benchmark reference (used only by s2_depth_benchmark --da3).
DEPTH_MODEL_ID_DA3: str = "depth-anything/Depth-Anything-3-Small-hf"

# Set True when using a metric model (depth in metres, larger = farther).
# Set False for relative/inverse depth models (larger = closer).
DEPTH_MODEL_METRIC: bool = True    # DA V2 Metric Indoor - depth in metres, larger = farther

# FP16 inference on RTX 4080. CPU fallback is allowed only for MockDepth.
DEPTH_DEVICE: str = "cuda"
DEPTH_DTYPE: str = "float16"

# Symmetric EMA (Stage 2 default, used when asymmetric mode is off).
SMOOTHING_ALPHA: float = 0.5
# Asymmetric EMA (Stage 4, spec sec 4.1): faster on approach, slower on recede.
# Direction depends on the active depth convention; DepthPipeline passes it
# explicitly to TemporalSmoother.
SMOOTHING_ALPHA_APPROACH: float = 0.8   # fast - react quickly when obstacle closes in
SMOOTHING_ALPHA_RECEDE: float = 0.2     # slow - keep stop active longer as obstacle moves away

# Acceptance gate for inference latency.
DEPTH_INFER_LATENCY_GATE_P95_MS: float = 100.0

# --- Stage 3 sectors ---
# Frame is split into SECTOR_COUNT equal vertical columns: LEFT / CENTER / RIGHT.
SECTOR_COUNT: int = 3

# --- Relative depth safety (DEPTH_MODEL_METRIC = False) ---
# Normalized threshold 0-1: pixel triggers stop if norm_depth > threshold.
# 1.0 = closest point in scene. Calibrated by s3_calibrate.py.
SAFETY_STOP_NORM_THRESHOLD: float = 0.926

# --- Metric depth safety (DEPTH_MODEL_METRIC = True) ---
# Absolute distance in metres: pixel triggers stop if depth_metres < threshold.
# Spec pseudo-metric: <0.8 m = danger, 1-2 m = safe. This is a hard
# near-field signal; navigation adds its own inflation margin on top.
SAFETY_STOP_DISTANCE_M: float = 0.85

# Fraction of pixels within a sector (after floor crop) that must exceed/subced threshold.
SAFETY_STOP_PIXEL_FRACTION: float = 0.05

# Bottom fraction of the frame to ignore - Tello camera points slightly downward,
# so the lower rows are mostly floor and would cause false positives.
SECTOR_FLOOR_CROP: float = 0.25

# --- YOLO hazard detection (Stage 6) ---
# YOLOv8n model - detects TV/screens/glass objects whose surfaces fool depth model.
# Pre-download BEFORE drone session: python scripts/download_yolo.py
# Set to "" to disable YOLO entirely (depth-only navigation).
YOLO_MODEL_PATH: str = "models/yolov8n.pt"
YOLO_ENABLED: bool = True
YOLO_IMG_SIZE: int = 320          # smaller = faster; 320 is fine for hazard detection
YOLO_MIN_CONF: float = 0.25
YOLO_HAZARD_MIN_CONF: float = 0.40
YOLO_DETECT_ALL_CLASSES: bool = True
# COCO class IDs whose surfaces give unreliable depth readings:
# 0=person, 39=bottle, 40=wine glass, 62=tv, 63=laptop, 74=clock (glass).
# Do not include 72=refrigerator: depth handles solid appliances, and the
# COCO fridge class can falsely turn kitchen entrances into forbidden space.
YOLO_HAZARD_CLASSES: tuple = (0, 39, 40, 62, 63, 74)

# --- Navigation policy thresholds (Stage 5/6) ---
# Stop forward only when CENTER is genuinely close.
NAV_FORWARD_STOP_M: float = 0.75
# Resume forward early; prefer moving through free space.
NAV_FORWARD_RESUME_M: float = 0.90
# Consecutive depth frames required before resuming forward:
NAV_FORWARD_RESUME_FRAMES: int = 2
# Yaw toward a side sector when it is this much farther than CENTER:
NAV_SIDE_FREER_RATIO: float = 1.70    # side must be much farther before yawing away
NAV_SIDE_MIN_M: float = 1.10          # ...and side itself at least this far
# Corner artifact: CENTER depth > max(L,R) * ratio -> false infinity, don't trust
NAV_CORNER_RATIO: float = 1.80
NAV_CORNER_FLOOR_M: float = 0.80      # only activate when L,R > this (real walls)
# Window/mirror anomaly: sector p10 above this is physically suspicious indoors
NAV_ANOMALY_DEPTH_M: float = 5.50
NAV_ANOMALY_NEIGHBOR_RATIO: float = 2.0
# Sustain: require N consecutive frames with anomaly pattern before yawing away.
# Prevents single-frame artifacts (and ordinary corridors briefly looking deep)
# from steering the drone into a side wall.
NAV_ANOMALY_SUSTAIN_FRAMES: int = 4
# Corridor guard: if both side sectors are similarly close (symmetric walls),
# the "deep centre" reading is just a corridor, NOT a mirror - skip anomaly.
# Active when max(L,R) < NAV_ANOMALY_CORRIDOR_SIDE_M AND
# min(L,R)/max(L,R) > NAV_ANOMALY_CORRIDOR_SYMMETRY.
NAV_ANOMALY_CORRIDOR_SIDE_M: float = 3.00
NAV_ANOMALY_CORRIDOR_SYMMETRY: float = 0.65
# YOLO distance gate: a hazard class (tv/laptop/fridge/clock/bottle) is only
# treated as obstacle when the depth model also reports the centre is close.
# Stops false "tv" hits on furniture deep in a corridor from steering away.
NAV_YOLO_DISTANCE_GATE_M: float = 1.80
# Exploration (when stuck hovering with no clear forward path)
NAV_EXPLORE_TRIGGER_S: float = 2.5   # seconds of non-forward before exploring
NAV_EXPLORE_SPIN_S: float = 5.0       # scan long enough to reveal side/behind exits
NAV_EXPLORE_COOLDOWN_S: float = 1.5   # cooldown before next explore cycle
NAV_RECOVERY_YAW_CAP: float = 36.0
NAV_RECOVERY_YAW_STEP: float = 8.0
NAV_RECOVERY_MIN_SCAN_S: float = 2.2
NAV_RECOVERY_EXIT_CENTER_M: float = 1.25
NAV_RECOVERY_ALTERNATE_BAND_M: float = 0.35
NAV_RECOVERY_FAR_BAND_M: float = 0.45
NAV_RECOVERY_EXIT_FRAMES: int = 10
NAV_RECOVERY_CAPTURE_CENTER_M: float = 1.75
NAV_RECOVERY_CAPTURE_FAR_M: float = 3.20
NAV_RECOVERY_CAPTURE_SIDE_M: float = 0.70
# Strong openings seen during a recovery turn should be captured quickly.
# Otherwise the drone can rotate past a kitchen/room doorway before the normal
# recovery exit hysteresis lets it fly forward.
NAV_RECOVERY_FAST_CAPTURE_MIN_SCAN_S: float = 0.65
NAV_RECOVERY_FAST_CAPTURE_FRAMES: int = 1
NAV_RECOVERY_FAST_CAPTURE_CENTER_M: float = 1.40
NAV_RECOVERY_FAST_CAPTURE_FAR_M: float = 2.35
NAV_RECOVERY_FAST_CAPTURE_SIDE_M: float = 0.58

# Dynamic forward velocity scheduler. Steering logic chooses direction; this
# layer only decides how much forward speed is safe for the current clearance.
NAV_SPEED_OPEN_M: float = 2.10
NAV_SPEED_CRUISE_M: float = 1.45
NAV_SPEED_CAUTION_M: float = 1.12
NAV_SPEED_SLOW_M: float = 0.92
NAV_SPEED_MIN_FORWARD_M: float = 0.78
NAV_SIDE_TIGHT_M: float = 0.62
NAV_SIDE_CAUTION_M: float = 0.82
# Do not squeeze into chair/table gaps: if the side clearance is this low,
# forward motion is allowed only when the centre is very open.
NAV_NARROW_NO_FORWARD_SIDE_M: float = 0.50
NAV_NARROW_OPEN_CENTER_M: float = 2.20
NAV_NARROW_SLOW_SIDE_M: float = 0.68
# Doorways often look like a far center with close vertical edges. If the
# center is clearly open, keep moving slowly instead of freezing at the frame.
NAV_DOORWAY_CENTER_M: float = 1.70
NAV_DOORWAY_MIN_SIDE_M: float = 0.35
NAV_DOORWAY_SPEED_CAP: float = 22.0
# Closed doors and walls often look like three similar close sectors. Do not
# resume forward into a flat front even if the centre is slightly above resume.
NAV_FLAT_FRONT_CENTER_M: float = 0.90
NAV_FLAT_FRONT_SPREAD_M: float = 0.12
NAV_FLAT_FRONT_FAR_M: float = 1.25

# --- Stage 8: obstacle inflation ---
# Safety margin subtracted from every raw distance before comparing with stop
# thresholds. Effectively shrinks corridor passability - drone won't squeeze
# through gaps smaller than 2 x inflation + drone radius.
NAV_INFLATION_M: float = 0.25

# --- Stage 9: motion prediction ---
# Total reaction latency (depth infer + control loop + Tello brake time).
# predicted_distance = raw_distance - forward_velocity_mps * latency_s
# Compensates Wi-Fi + processing lag when flying forward.
NAV_MOTION_LATENCY_S: float = 0.20

