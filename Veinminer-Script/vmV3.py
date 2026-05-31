#!/usr/bin/env python3
"""
VeinMiner – intelligent block‑by‑block vein mining with reinforcement learning.
Now powered by PyTorch.
"""

import minescript
import sys
import time
import threading
import random
import math
import os
import glob
from collections import deque

# ----------------------------------------------------------------------
# PyTorch import – if missing, fall back gracefully (the script won't work
# but at least it prints a clear error).
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ----------------------------------------------------------------------
# Environment check
if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"

# ======================================================================
# PATHS & CONSTANTS
# ======================================================================
MODEL_DIR = os.path.expanduser(
    "/home/lunar/.local/share/PrismLauncher/instances/"
    "Skyblock Ironman/minecraft/minescript/veinminer/model"
)
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

STATE_FILE = os.path.join(MODEL_DIR, "state.txt")

# ----------------------------------------------------------------------
# MINING & BEHAVIOUR CONFIG
# ----------------------------------------------------------------------
MINE_HOLD_SECONDS      = 0.8
BETWEEN_BLOCKS_SECONDS = 0.02
MAX_RADIUS             = 5
MAX_PLAYER_REACH       = 4.5
CONFIRM_POLL_INTERVAL  = 0.03
CONFIRM_TIMEOUT        = 1.0
LOOK_STEPS             = 13
LOOK_STEP_SLEEP        = 0.024
JITTER_YAW_MAX         = 0.2
JITTER_PITCH_MAX       = 0.15
JITTER_CHANCE          = 0.05
BLACKLIST_DURATION     = 60.0

# --- AI constants ---
MAX_CANDIDATES_MODEL   = 25          # blocks evaluated per cycle
EPSILON                = 0.10        # exploration rate
SESSION_TIME           = 900         # 15 minutes per training session
REPLAY_BUFFER_SIZE     = 10000
BATCH_SIZE             = 32

# Net‑profit reward weights (all costs are subtracted from a base +1 per block)
BASE_REWARD            = 1.0
YAW_COST_WEIGHT        = 0.02
PITCH_COST_WEIGHT      = 0.03
DISTANCE_COST_WEIGHT   = 0.05
CENTER_COST_WEIGHT     = 0.1         # penalty for moving away from cluster origin

VERBOSE                = False

# ======================================================================
# SHARED STATE
# ======================================================================
_lock    = threading.Lock()
_running = True
_stop    = False

def is_running():
    with _lock: return _running

def set_running(val):
    with _lock:
        global _running
        _running = val

def is_stopped():
    with _lock: return _stop

def set_stop():
    with _lock:
        global _stop
        _stop = True

# ======================================================================
# HELPERS (unchanged)
# ======================================================================
def strip_ns(name: str) -> str:
    return name.split(":")[-1].split("[")[0].lower().strip()

def block_at(x, y, z) -> str:
    try:
        return strip_ns(minescript.getblock(x, y, z))
    except:
        return "air"

def is_block_air(x, y, z) -> bool:
    return block_at(x, y, z) in ("air", "cave_air", "void_air")

def get_eye_position() -> tuple:
    try:
        return minescript.player_eye_position()
    except AttributeError:
        px, py, pz = minescript.player_position()
        return (px, py + 1.62, pz)

def has_line_of_sight(eye, tx, ty, tz) -> bool:
    ex, ey, ez = eye
    cx, cy, cz = tx + 0.5, ty + 0.5, tz + 0.5
    dx, dy, dz = cx - ex, cy - ey, cz - ez
    steps = int(max(abs(dx), abs(dy), abs(dz)) * 10)
    if steps < 1:
        steps = 1
    step_x, step_y, step_z = dx/steps, dy/steps, dz/steps
    last_bx, last_by, last_bz = None, None, None
    for i in range(steps):
        px = ex + step_x * i
        py = ey + step_y * i
        pz = ez + step_z * i
        bx, by, bz = int(math.floor(px)), int(math.floor(py)), int(math.floor(pz))
        if (bx, by, bz) == (tx, ty, tz):
            return True
        if (bx, by, bz) != (last_bx, last_by, last_bz):
            if not is_block_air(bx, by, bz):
                return False
            last_bx, last_by, last_bz = bx, by, bz
    return True

# ======================================================================
# NEIGHBOR & VEIN ANALYSIS
# ======================================================================
def count_neighbors(x, y, z, target_name, radius=1):
    cnt = 0
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            for dz in range(-radius, radius+1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if abs(dx)+abs(dy)+abs(dz) <= radius:
                    if block_at(x+dx, y+dy, z+dz) == target_name:
                        cnt += 1
    return cnt

def count_visible_neighbors(x, y, z, target_name, eye):
    cnt = 0
    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        nx, ny, nz = x+dx, y+dy, z+dz
        if block_at(nx, ny, nz) == target_name:
            if has_line_of_sight(eye, nx, ny, nz):
                cnt += 1
    return cnt

def manhattan_center_dist(pos, origin):
    return abs(pos[0]-origin[0]) + abs(pos[1]-origin[1]) + abs(pos[2]-origin[2])

# ======================================================================
# ENHANCED FEATURE EXTRACTION (now includes center distance)
# ======================================================================
def extract_features(eye, yaw, pitch, block_pos, origin, target_name):
    x, y, z = block_pos
    cx, cy, cz = x + 0.5, y + 0.5, z + 0.5
    rel_x = cx - eye[0]
    rel_y = cy - eye[1]
    rel_z = cz - eye[2]
    distance = math.sqrt(rel_x*rel_x + rel_y*rel_y + rel_z*rel_z) + 1e-6

    # direction cosines
    nx, ny, nz = rel_x/distance, rel_y/distance, rel_z/distance
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    look_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
    look_y = -math.sin(pitch_rad)
    look_z = math.cos(yaw_rad) * math.cos(pitch_rad)
    dot = nx*look_x + ny*look_y + nz*look_z

    # rotation deltas
    dx, dz = rel_x, rel_z
    target_yaw = math.degrees(math.atan2(-dx, dz)) % 360
    target_pitch = math.degrees(math.atan2(-rel_y, math.sqrt(dx*dx + dz*dz)))
    target_pitch = max(-90.0, min(90.0, target_pitch))
    yaw_delta = ((target_yaw - yaw + 180) % 360) - 180
    pitch_delta = target_pitch - pitch
    total_rotation = abs(yaw_delta) + abs(pitch_delta)

    neighbor_cnt = count_neighbors(x, y, z, target_name, radius=1)
    visible_neighbor_cnt = count_visible_neighbors(x, y, z, target_name, eye)
    center_dist = manhattan_center_dist(block_pos, origin)

    return [
        distance,
        abs(yaw_delta),
        abs(pitch_delta),
        total_rotation,
        neighbor_cnt,
        visible_neighbor_cnt,
        center_dist,            # new feature – distance from vein origin
        rel_x,
        rel_y,
        rel_z,
        dot
    ]

# ======================================================================
# NET PROFIT REWARD (immediate per‑block reward)
# ======================================================================
def compute_net_profit(yaw_delta, pitch_delta, distance, center_dist):
    """Reward = base - movement costs. Can be negative if costs are high."""
    cost = (yaw_delta * YAW_COST_WEIGHT +
            pitch_delta * PITCH_COST_WEIGHT +
            distance * DISTANCE_COST_WEIGHT +
            center_dist * CENTER_COST_WEIGHT)
    return BASE_REWARD - cost

# ======================================================================
# PYTORCH NEURAL NETWORK
# ======================================================================
class MinerNet(nn.Module):
    def __init__(self, input_size=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(filepath, input_size=11):
    model = MinerNet(input_size)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

# ======================================================================
# MODEL MANAGEMENT (using .pth extension)
# ======================================================================
def parse_model_filename(filename):
    """Extract block name and grade from a checkpoint filename.
    Expected format: <block>.<grade>.<timestamp>.pth"""
    base = os.path.basename(filename)
    if not base.endswith(".pth"):
        return None
    name = base[:-4]
    parts = name.rsplit(".", 2)  # split into [block, grade, timestamp]
    if len(parts) != 3:
        return None
    block = parts[0]
    try:
        grade = int(parts[1])
    except ValueError:
        return None
    return block, grade

def best_model_path(target_block):
    """Returns path to best checkpoint or the best_{target}.pth file."""
    # First look in checkpoints
    pattern = os.path.join(CHECKPOINT_DIR, f"{target_block}.*.*.pth")
    candidates = glob.glob(pattern)
    best = None
    best_grade = -1
    for c in candidates:
        info = parse_model_filename(c)
        if info and info[0] == target_block and info[1] > best_grade:
            best_grade = info[1]
            best = c
    # Also consider the promoted best model
    best_file = os.path.join(MODEL_DIR, f"best_{target_block}.pth")
    if os.path.isfile(best_file):
        # For the best file we don't have a grade in the name, so assume it's good
        # but compare with the best checkpoint grade if known.
        # We'll just return it if no checkpoint or if it's explicitly the best.
        if not best or best_grade < 999:   # best file always considered highest if exists
            best = best_file
    return best

def grade_from_points(total_profit, blocks_mined):
    """Convert average net profit per block into a 0‑100 grade."""
    if blocks_mined == 0:
        return 0
    avg = total_profit / blocks_mined
    # Map avg from [-0.2, 0.8] to 0‑100 (typical range after tuning costs)
    # Clamp extremes
    return max(0, min(100, int((avg + 0.2) / 1.0 * 100)))

def save_and_promote_model(model, target_block, total_profit, blocks_mined):
    grade = grade_from_points(total_profit, blocks_mined)
    timestamp = int(time.time())
    filename = f"{target_block}.{grade}.{timestamp}.pth"
    path = os.path.join(CHECKPOINT_DIR, filename)
    save_model(model, path)
    minescript.echo(f"[VeinMiner] Checkpoint saved: {filename} (grade {grade}/100)")

    # Promote if better than current best
    best_existing = best_model_path(target_block)
    current_best_grade = -1
    if best_existing and best_existing != os.path.join(MODEL_DIR, f"best_{target_block}.pth"):
        info = parse_model_filename(best_existing)
        if info:
            current_best_grade = info[1]
    if grade > current_best_grade:
        best_path = os.path.join(MODEL_DIR, f"best_{target_block}.pth")
        save_model(model, best_path)
        minescript.echo(f"[VeinMiner] 🏆 New best model! (grade {grade} > {current_best_grade}) -> {best_path}")

# ======================================================================
# CANDIDATE COLLECTION WITH EPSILON‑GREEDY (uses network for prediction)
# ======================================================================
def collect_candidates(origin, target_name, blacklist, eye, yaw, pitch,
                       max_candidates, model=None, device='cpu'):
    """BFS that collects visible target blocks and returns best block + all candidates."""
    ox, oy, oz = origin
    visited = set()
    queue = deque([origin])
    faces = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    now = time.time()
    candidates = []  # (pos, features, reward, prediction)

    while queue and len(candidates) < max_candidates:
        pos = queue.popleft()
        if pos in visited:
            continue
        visited.add(pos)
        x, y, z = pos
        if abs(x-ox) > MAX_RADIUS or abs(y-oy) > MAX_RADIUS or abs(z-oz) > MAX_RADIUS:
            continue
        if pos in blacklist and now - blacklist[pos] < BLACKLIST_DURATION:
            continue
        if block_at(x, y, z) == target_name:
            dist_to_player = math.sqrt((x+0.5-eye[0])**2 + (y+0.5-eye[1])**2 + (z+0.5-eye[2])**2)
            if dist_to_player <= MAX_PLAYER_REACH and has_line_of_sight(eye, x, y, z):
                feat = extract_features(eye, yaw, pitch, (x,y,z), origin, target_name)
                # Compute ground‑truth reward (used later for training)
                yaw_delta, pitch_delta, distance = feat[1], feat[2], feat[0]
                center_dist = feat[6]
                reward = compute_net_profit(yaw_delta, pitch_delta, distance, center_dist)
                # Prediction from network (if available)
                if model is not None:
                    with torch.no_grad():
                        tensor = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
                        pred = model(tensor).item()
                else:
                    pred = 0.0
                candidates.append(((x,y,z), feat, reward, pred))
                if len(candidates) >= max_candidates:
                    break
        for dx, dy, dz in faces:
            nb = (x+dx, y+dy, z+dz)
            if nb not in visited:
                queue.append(nb)

    if not candidates:
        return None, []

    # Epsilon‑greedy selection
    if model is not None and random.random() < EPSILON:
        best_idx = random.randrange(len(candidates))
    else:
        # Choose candidate with highest predicted net profit
        best_idx = max(range(len(candidates)), key=lambda i: candidates[i][3])

    best_pos = candidates[best_idx][0]
    return best_pos, candidates

# ======================================================================
# SMOOTH LOOK (unchanged)
# ======================================================================
def smooth_look_at(tx, ty, tz):
    eye_x, eye_y, eye_z = get_eye_position()
    dx, dy, dz = tx - eye_x, ty - eye_y, tz - eye_z
    dist_xz = math.sqrt(dx*dx + dz*dz)
    target_yaw = math.degrees(math.atan2(-dx, dz)) % 360
    target_pitch = math.degrees(math.atan2(-dy, dist_xz))
    target_pitch = max(-90.0, min(90.0, target_pitch))
    cur_yaw, cur_pitch = minescript.player_orientation()
    yaw_delta = ((target_yaw - cur_yaw + 180) % 360) - 180
    pitch_delta = target_pitch - cur_pitch
    for step in range(1, LOOK_STEPS+1):
        if is_stopped():
            return
        t = step / LOOK_STEPS
        ease = t*t*(3-2*t)
        if random.random() < JITTER_CHANCE:
            jy = random.uniform(-JITTER_YAW_MAX, JITTER_YAW_MAX)
            jp = random.uniform(-JITTER_PITCH_MAX, JITTER_PITCH_MAX)
        else:
            jy = jp = 0.0
        new_yaw = cur_yaw + yaw_delta*ease + jy
        new_pitch = cur_pitch + pitch_delta*ease + jp
        new_pitch = max(-90.0, min(90.0, new_pitch))
        minescript.player_set_orientation(new_yaw, new_pitch)
        time.sleep(LOOK_STEP_SLEEP)
    minescript.player_set_orientation(target_yaw, target_pitch)

# ======================================================================
# STATE LISTENER (unchanged)
# ======================================================================
def state_listener(score_holder):
    try:
        with open(STATE_FILE, "w") as f:
            f.write("run")
    except Exception as e:
        minescript.echo(f"[VeinMiner] Error creating state file: {e}")
        return

    last_state = "run"
    while not is_stopped():
        try:
            with open(STATE_FILE, "r") as f:
                current_state = f.read().strip()
            if current_state != last_state:
                if current_state == "stop":
                    set_stop()
                    minescript.echo(f"[VeinMiner] Stopped cleanly. Total points: {score_holder[0]:.2f}")
                    break
                elif current_state == "pause":
                    set_running(False)
                    minescript.echo("[VeinMiner] ⏸ PAUSED")
                elif current_state == "run":
                    set_running(True)
                    minescript.echo("[VeinMiner] ▶ RESUMED")
                last_state = current_state
        except Exception:
            pass
        time.sleep(0.2)

# ======================================================================
# MAIN MINING LOOP WITH PYTORCH TRAINING
# ======================================================================
def mine_loop(start_pos, target_name, mode="normal", model=None, device='cpu'):
    last_pos = start_pos
    mined = 0
    blacklist = {}
    total_profit = 0.0
    score_holder = [0.0]

    # Experience replay buffer (features, reward)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Optimizer and loss if training/hybrid
    optimizer = None
    loss_fn = nn.MSELoss()
    if model is not None and mode in ("train", "hybrid"):
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start file listener
    threading.Thread(target=state_listener, args=(score_holder,), daemon=True).start()

    session_start = time.time()

    while not is_stopped():
        # Pause handling
        while not is_running():
            if is_stopped():
                break
            time.sleep(0.1)
        if is_stopped():
            break

        # Session timeout
        if time.time() - session_start > SESSION_TIME:
            minescript.echo(f"[VeinMiner] Session time limit ({SESSION_TIME//60} min) reached.")
            break

        eye = get_eye_position()
        yaw, pitch = minescript.player_orientation()

        # Collect candidates
        if mode in ("train", "hybrid", "model") and model is not None:
            best_block, candidates = collect_candidates(
                last_pos, target_name, blacklist, eye, yaw, pitch,
                MAX_CANDIDATES_MODEL, model, device
            )
        else:
            best_block, candidates = collect_candidates(
                last_pos, target_name, blacklist, eye, yaw, pitch,
                1, None, device
            )

        if best_block is None:
            time.sleep(0.3)
            blacklist.clear()
            continue

        x, y, z = best_block

        # Train on all candidates (online update) if in training mode
        if model is not None and mode in ("train", "hybrid") and optimizer is not None:
            model.train()
            for _, feat, reward, _ in candidates:
                feat_tensor = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
                target = torch.tensor([reward], dtype=torch.float32, device=device).unsqueeze(0)
                optimizer.zero_grad()
                pred = model(feat_tensor)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()
                replay_buffer.append((feat, reward))

        # Mini‑batch training from replay buffer
        if model is not None and mode in ("train", "hybrid") and optimizer is not None and len(replay_buffer) >= BATCH_SIZE:
            model.train()
            batch = random.sample(replay_buffer, BATCH_SIZE)
            feats = torch.tensor([f for f, r in batch], dtype=torch.float32, device=device)
            targets = torch.tensor([[r] for f, r in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            preds = model(feats)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

        # Find the reward of the chosen block (to add to total)
        chosen_reward = 0.0
        for pos, feat, reward, _ in candidates:
            if pos == best_block:
                chosen_reward = reward
                break
        total_profit += chosen_reward
        score_holder[0] = total_profit

        # Smooth look at block centre
        smooth_look_at(x+0.5, y+0.5, z+0.5)
        if is_stopped():
            break

        # Final check: still the same ore?
        if block_at(x, y, z) != target_name:
            if VERBOSE:
                minescript.echo(f"[VeinMiner] Skipping {best_block} – block changed")
            continue

        # Mine
        minescript.player_press_attack(True)
        time.sleep(MINE_HOLD_SECONDS)
        minescript.player_press_attack(False)

        if wait_for_break(x, y, z, target_name):
            mined += 1
            last_pos = best_block
            blacklist.clear()
            if VERBOSE:
                minescript.echo(f"[VeinMiner] #{mined} {best_block}  net profit: {chosen_reward:.3f}  total: {total_profit:.2f}")
        else:
            blacklist[best_block] = time.time()

        time.sleep(BETWEEN_BLOCKS_SECONDS)

    # End of session
    minescript.echo(f"[VeinMiner] Session ended. Mined: {mined}, Total net profit: {total_profit:.2f}")
    if model is not None and mode in ("train", "hybrid") and mined > 0:
        save_and_promote_model(model, target_name, total_profit, mined)

def wait_for_break(x, y, z, target_name) -> bool:
    deadline = time.time() + CONFIRM_TIMEOUT
    while time.time() < deadline:
        if is_stopped():
            return False
        if block_at(x, y, z) != target_name:
            return True
        time.sleep(CONFIRM_POLL_INTERVAL)
    return False

# ======================================================================
# ENTRY POINT
# ======================================================================
def main():
    if not TORCH_OK:
        minescript.echo("[VeinMiner] PyTorch is not installed. Please install it (pip install torch) and try again.")
        return

    args = sys.argv[1:]

    mode = "normal"
    if args:
        if args[0].lower() in ("train", "model", "hybrid"):
            mode = args[0].lower()
            args.pop(0)

    targeted = minescript.player_get_targeted_block(max_distance=6)
    if targeted is None:
        minescript.echo("[VeinMiner] Not looking at any block.")
        return
    start_pos = tuple(int(v) for v in targeted.position)
    detected = strip_ns(targeted.type)

    if args:
        target_name = strip_ns(args[0])
        if detected != target_name:
            minescript.echo(f"[VeinMiner] Looking at '{detected}', requested '{target_name}'.")
            return
    else:
        target_name = detected

    if target_name in ("air", "cave_air", "void_air"):
        minescript.echo("[VeinMiner] Can't mine air.")
        return

    # Device: CPU (GPU not needed for this tiny network)
    device = torch.device('cpu')

    model = None
    if mode in ("train", "hybrid", "model"):
        if mode == "train":
            model = MinerNet().to(device)
            minescript.echo("[VeinMiner] Training mode – fresh PyTorch network.")
        elif mode == "hybrid":
            best = best_model_path(target_name)
            if best:
                model = load_model(best).to(device)
                minescript.echo(f"[VeinMiner] Hybrid mode – loaded {os.path.basename(best)}")
            else:
                model = MinerNet().to(device)
                minescript.echo("[VeinMiner] Hybrid mode – no checkpoint, starting fresh.")
        elif mode == "model":
            best = best_model_path(target_name)
            if best:
                model = load_model(best).to(device)
                minescript.echo(f"[VeinMiner] Model mode – using {os.path.basename(best)}")
            else:
                minescript.echo("[VeinMiner] No model found, falling back to normal mode")
                mode = "normal"

    set_running(True)
    mine_loop(start_pos, target_name, mode, model, device)

if __name__ == "__main__":
    main()