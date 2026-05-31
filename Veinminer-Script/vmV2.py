
import minescript
import sys
import time
import threading
import random
import math
import json
import os
import glob
from collections import deque

if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"

# ============================================================
#  PATHS & CONSTANTS
# ============================================================
MODEL_DIR = os.path.expanduser(
    "/home/lunar/.local/share/PrismLauncher/instances/"
    "Skyblock Ironman/minecraft/minescript/veinminer/model"
)
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

STATE_FILE = os.path.join(MODEL_DIR, "state.txt")

# ============================================================
#  MINING & BEHAVIOUR CONFIG
# ============================================================
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

# === New AI constants ===
MAX_CANDIDATES_MODEL   = 25          # blocks evaluated per cycle
EPSILON                = 0.10        # exploration rate
SESSION_TIME           = 900         # 15 minutes per training session
REPLAY_BUFFER_SIZE     = 10000
BATCH_SIZE             = 32

# Net‑profit reward weights
YAW_COST_WEIGHT        = 1.0
PITCH_COST_WEIGHT      = 1.2
DISTANCE_COST_WEIGHT   = 0.2
NEIGHBOR_POTENTIAL_MULT = 3.0

VERBOSE                = False

# ============================================================
#  SHARED STATE
# ============================================================
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

# ============================================================
#  HELPERS
# ============================================================
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

# ============================================================
#  FAST LINE‑OF‑SIGHT (integer DDA)
# ============================================================
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

# ============================================================
#  NEIGHBOR & VEIN ANALYSIS
# ============================================================
def count_neighbors(x, y, z, target_name, radius=1):
    """Count adjacent target blocks (Manhattan distance <= radius)."""
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
    """Count adjacent target blocks that have line of sight from the player's eye."""
    cnt = 0
    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        nx, ny, nz = x+dx, y+dy, z+dz
        if block_at(nx, ny, nz) == target_name:
            if has_line_of_sight(eye, nx, ny, nz):
                cnt += 1
    return cnt

def vein_depth(origin, pos):
    """Manhattan distance from origin as a simple depth metric."""
    return abs(origin[0]-pos[0]) + abs(origin[1]-pos[1]) + abs(origin[2]-pos[2])

# ============================================================
#  NET PROFIT REWARD
# ============================================================
def compute_net_profit(yaw_delta, pitch_delta, distance, visible_neighbor_count):
    movement_cost = (yaw_delta * YAW_COST_WEIGHT) + (pitch_delta * PITCH_COST_WEIGHT) + (distance * DISTANCE_COST_WEIGHT)
    potential_gain = visible_neighbor_count * NEIGHBOR_POTENTIAL_MULT
    return potential_gain - movement_cost

# ============================================================
#  ENHANCED NEURAL NETWORK (64→32→1)
# ============================================================
class NeuralNetwork:
    def __init__(self, input_size=11, hidden1=64, hidden2=32, output_size=1, lr=0.01):
        self.lr = lr
        # Layer 1
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(hidden1)]
        self.b1 = [0.0] * hidden1
        # Layer 2
        self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden1)] for _ in range(hidden2)]
        self.b2 = [0.0] * hidden2
        # Output layer
        self.w3 = [[random.uniform(-0.5, 0.5) for _ in range(hidden2)] for _ in range(output_size)]
        self.b3 = [0.0] * output_size

    def _relu(self, x): return max(0, x)
    def _relu_deriv(self, x): return 1 if x > 0 else 0

    def forward(self, x):
        # Layer 1
        self.z1 = [sum(x[j] * self.w1[i][j] for j in range(len(x))) + self.b1[i] for i in range(len(self.w1))]
        self.a1 = [self._relu(v) for v in self.z1]
        # Layer 2
        self.z2 = [sum(self.a1[j] * self.w2[i][j] for j in range(len(self.a1))) + self.b2[i] for i in range(len(self.w2))]
        self.a2 = [self._relu(v) for v in self.z2]
        # Output
        self.z3 = [sum(self.a2[j] * self.w3[0][j] for j in range(len(self.a2))) + self.b3[0]]
        self.a3 = self.z3[0]
        return self.a3

    def backward(self, x, y_true):
        # Output layer gradients
        error = 2 * (self.a3 - y_true)
        d_w3 = [[error * self.a2[i] for i in range(len(self.a2))]]
        d_b3 = [error]

        # Backprop through layer 2
        d_a2 = [error * self.w3[0][i] for i in range(len(self.a2))]
        d_z2 = [d_a2[i] * self._relu_deriv(self.z2[i]) for i in range(len(self.z2))]
        d_w2 = [[d_z2[i] * self.a1[j] for j in range(len(self.a1))] for i in range(len(d_z2))]
        d_b2 = d_z2

        # Backprop through layer 1
        d_a1 = [sum(d_z2[i] * self.w2[i][j] for i in range(len(self.w2))) for j in range(len(self.a1))]
        d_z1 = [d_a1[i] * self._relu_deriv(self.z1[i]) for i in range(len(self.z1))]
        d_w1 = [[d_z1[i] * x[j] for j in range(len(x))] for i in range(len(d_z1))]
        d_b1 = d_z1

        # Update weights and biases
        for i in range(len(self.w3)):
            for j in range(len(self.w3[i])):
                self.w3[i][j] -= self.lr * d_w3[i][j]
        for i in range(len(self.b3)):
            self.b3[i] -= self.lr * d_b3[i]

        for i in range(len(self.w2)):
            for j in range(len(self.w2[i])):
                self.w2[i][j] -= self.lr * d_w2[i][j]
        for i in range(len(self.b2)):
            self.b2[i] -= self.lr * d_b2[i]

        for i in range(len(self.w1)):
            for j in range(len(self.w1[i])):
                self.w1[i][j] -= self.lr * d_w1[i][j]
        for i in range(len(self.b1)):
            self.b1[i] -= self.lr * d_b1[i]

    def save(self, filepath):
        data = {
            "w1": self.w1, "b1": self.b1,
            "w2": self.w2, "b2": self.b2,
            "w3": self.w3, "b3": self.b3,
            "lr": self.lr
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        net = cls()
        net.w1 = data["w1"]
        net.b1 = data["b1"]
        net.w2 = data["w2"]
        net.b2 = data["b2"]
        net.w3 = data["w3"]
        net.b3 = data["b3"]
        net.lr = data.get("lr", 0.01)
        return net

# ============================================================
#  ENHANCED FEATURE EXTRACTION (11 features)
# ============================================================
def extract_features(eye, yaw, pitch, block_pos, origin, target_name):
    x, y, z = block_pos
    cx, cy, cz = x + 0.5, y + 0.5, z + 0.5
    rel_x = cx - eye[0]
    rel_y = cy - eye[1]
    rel_z = cz - eye[2]
    distance = math.sqrt(rel_x*rel_x + rel_y*rel_y + rel_z*rel_z) + 1e-6

    # Direction from player to block
    nx, ny, nz = rel_x/distance, rel_y/distance, rel_z/distance
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    look_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
    look_y = -math.sin(pitch_rad)
    look_z = math.cos(yaw_rad) * math.cos(pitch_rad)
    dot = nx*look_x + ny*look_y + nz*look_z

    # Rotation deltas
    dx, dz = rel_x, rel_z
    target_yaw = math.degrees(math.atan2(-dx, dz)) % 360
    target_pitch = math.degrees(math.atan2(-rel_y, math.sqrt(dx*dx + dz*dz)))
    target_pitch = max(-90.0, min(90.0, target_pitch))
    yaw_delta = ((target_yaw - yaw + 180) % 360) - 180
    pitch_delta = target_pitch - pitch
    total_rotation = abs(yaw_delta) + abs(pitch_delta)

    # Neighbor counts
    neighbor_cnt = count_neighbors(x, y, z, target_name, radius=1)
    visible_neighbor_cnt = count_visible_neighbors(x, y, z, target_name, eye)
    depth = vein_depth(origin, block_pos)

    return [
        distance,
        abs(yaw_delta),
        abs(pitch_delta),
        total_rotation,
        neighbor_cnt,
        visible_neighbor_cnt,
        depth,
        rel_x,
        rel_y,
        rel_z,
        dot
    ]

# ============================================================
#  CANDIDATE COLLECTION WITH EPSILON‑GREEDY
# ============================================================
def collect_candidates(origin, target_name, blacklist, eye, yaw, pitch, max_candidates, net=None):
    """BFS that collects up to max_candidates visible target blocks.
    Returns (best_pos, candidate_list) where candidate_list contains tuples
    (pos, features, reward, prediction). Uses epsilon‑greedy if net is given."""
    ox, oy, oz = origin
    visited = set()
    queue = deque([origin])
    faces = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    now = time.time()
    candidates = []   # (pos, features, reward, pred)

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
                # Compute features and net profit reward
                feat = extract_features(eye, yaw, pitch, (x,y,z), origin, target_name)
                # Compute reward using net‑profit formula
                yaw_delta, pitch_delta, distance = feat[1], feat[2], feat[0]
                visible_neighbors = feat[5]
                reward = compute_net_profit(yaw_delta, pitch_delta, distance, visible_neighbors)
                # Get prediction from network if available
                pred = net.forward(feat) if net is not None else 0.0
                candidates.append(((x,y,z), feat, reward, pred))
                if len(candidates) >= max_candidates:
                    break
        for dx, dy, dz in faces:
            nb = (x+dx, y+dy, z+dz)
            if nb not in visited:
                queue.append(nb)

    if not candidates:
        return None, []

    # Epsilon‑greedy: choose random candidate 10% of the time if we have a network
    if net is not None and random.random() < EPSILON:
        best_idx = random.randrange(len(candidates))
    else:
        # Choose candidate with highest predicted reward (network output)
        best_idx = max(range(len(candidates)), key=lambda i: candidates[i][3])

    best_pos = candidates[best_idx][0]
    return best_pos, candidates

# ============================================================
#  SMOOTH LOOK
# ============================================================
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

# ============================================================
#  FILE‑BASED STATE LISTENER
# ============================================================
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
                    minescript.echo(f"[VeinMiner] Stopped cleanly. Total points: {score_holder[0]}")
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

# ============================================================
#  MAIN MINING LOOP WITH EXPERIENCE REPLAY & SESSION LIMIT
# ============================================================
def mine_loop(start_pos, target_name, mode="normal", net=None):
    last_pos = start_pos
    mined = 0
    blacklist = {}
    total_points = 0.0
    score_holder = [0.0]

    # Experience replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Start file listener
    threading.Thread(target=state_listener, args=(score_holder,), daemon=True).start()

    # Session timer
    session_start = time.time()

    while not is_stopped():
        # Pause handling
        while not is_running():
            if is_stopped():
                break
            time.sleep(0.1)
        if is_stopped():
            break

        # Check session timeout
        if time.time() - session_start > SESSION_TIME:
            minescript.echo(f"[VeinMiner] Session time limit ({SESSION_TIME//60} min) reached.")
            break

        eye = get_eye_position()
        yaw, pitch = minescript.player_orientation()

        # Collect candidates using network if available
        if mode in ("model", "train", "hybrid") and net is not None:
            best_block, candidates = collect_candidates(
                last_pos, target_name, blacklist, eye, yaw, pitch,
                MAX_CANDIDATES_MODEL, net
            )
        else:
            # Normal mode: simple BFS, no network, no training
            best_block, candidates = collect_candidates(
                last_pos, target_name, blacklist, eye, yaw, pitch,
                1, None  # only one candidate, no prediction
            )

        if best_block is None:
            time.sleep(0.3)
            blacklist.clear()
            continue

        x, y, z = best_block

        # Train on all collected candidates (if in training/hybrid mode)
        if mode in ("train", "hybrid") and net is not None and candidates:
            for pos, feat, reward, _ in candidates:
                net.forward(feat)
                net.backward(feat, reward)
                replay_buffer.append((feat, reward))

        # Mini‑batch training from replay buffer
        if mode in ("train", "hybrid") and net is not None and len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            for feat, reward in batch:
                net.forward(feat)
                net.backward(feat, reward)

        # Find reward of the chosen block (from candidates)
        chosen_reward = 0.0
        for pos, feat, reward, _ in candidates:
            if pos == best_block:
                chosen_reward = reward
                break
        total_points += chosen_reward
        score_holder[0] = total_points

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
                minescript.echo(f"[VeinMiner] #{mined} {best_block}  net profit: {chosen_reward:.2f}  total: {total_points:.2f}")
        else:
            blacklist[best_block] = time.time()

        time.sleep(BETWEEN_BLOCKS_SECONDS)

    # End of session: save model and promote if better
    minescript.echo(f"[VeinMiner] Session ended. Mined: {mined}, Total net profit: {total_points:.2f}")
    if mode in ("train", "hybrid") and net is not None and mined > 0:
        save_and_promote_model(net, target_name, total_points, mined)

def wait_for_break(x, y, z, target_name) -> bool:
    deadline = time.time() + CONFIRM_TIMEOUT
    while time.time() < deadline:
        if is_stopped():
            return False
        if block_at(x, y, z) != target_name:
            return True
        time.sleep(CONFIRM_POLL_INTERVAL)
    return False

# ============================================================
#  MODEL MANAGEMENT WITH PROMOTION
# ============================================================
def parse_model_filename(filename):
    base = os.path.basename(filename)
    if not base.endswith(".xyz"):
        return None
    name = base[:-4]
    parts = name.rsplit(".", 1)
    if len(parts) != 2:
        return None
    block = parts[0]
    try:
        score = int(parts[1])
    except ValueError:
        grade_map = {"A":95,"B":85,"C":75,"D":65,"F":50}
        if parts[1] in grade_map:
            score = grade_map[parts[1]]
        else:
            return None
    return block, score

def best_model_path(target_block, checkpoints_only=True):
    """Returns path to best model (checkpoint or best.xyz) for given target."""
    search_dir = CHECKPOINT_DIR if checkpoints_only else MODEL_DIR
    pattern = os.path.join(search_dir, f"{target_block}.*.xyz")
    candidates = glob.glob(pattern)
    best = None
    best_score = -1
    for c in candidates:
        info = parse_model_filename(c)
        if info and info[0] == target_block and info[1] > best_score:
            best_score = info[1]
            best = c
    # Also check the explicit best file if allowed
    if not checkpoints_only:
        best_file = os.path.join(MODEL_DIR, f"best_{target_block}.xyz")
        if os.path.isfile(best_file):
            info = parse_model_filename(best_file)
            if info and info[1] > best_score:
                best = best_file
    return best

def grade_from_points(total_points, blocks_mined):
    if blocks_mined == 0:
        return 50
    avg = total_points / blocks_mined
    # Convert average net profit to 0-100 grade (clamped)
    # Net profit can be negative; map -5..+15 -> 0..100 roughly
    grade = (avg + 5) / 20 * 100
    return max(0, min(100, int(grade)))

def save_and_promote_model(net, target_block, total_points, blocks_mined):
    grade = grade_from_points(total_points, blocks_mined)
    timestamp = int(time.time())
    filename = f"{target_block}.{grade}.{timestamp}.xyz"
    path = os.path.join(CHECKPOINT_DIR, filename)
    net.save(path)
    minescript.echo(f"[VeinMiner] Checkpoint saved: {filename} (grade {grade}/100)")

    # Promote if better than best known
    best_existing = best_model_path(target_block, checkpoints_only=False)
    current_best_grade = -1
    if best_existing:
        info = parse_model_filename(best_existing)
        if info:
            current_best_grade = info[1]
    if grade > current_best_grade:
        best_path = os.path.join(MODEL_DIR, f"best_{target_block}.xyz")
        net.save(best_path)
        minescript.echo(f"[VeinMiner] 🏆 New best model! (grade {grade} > {current_best_grade}) -> {best_path}")

# ============================================================
#  ENTRY POINT
# ============================================================
def main():
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

    net = None
    if mode in ("train", "hybrid", "model"):
        if mode == "train":
            net = NeuralNetwork()
            minescript.echo("[VeinMiner] Training mode – fresh network (64→32→1)")
        elif mode == "hybrid":
            best = best_model_path(target_name, checkpoints_only=False)
            net = NeuralNetwork.load(best) if best else NeuralNetwork()
            minescript.echo(f"[VeinMiner] Hybrid mode – {'loaded '+os.path.basename(best) if best else 'starting fresh'}")
        elif mode == "model":
            best = best_model_path(target_name, checkpoints_only=False)
            if best:
                net = NeuralNetwork.load(best)
                minescript.echo(f"[VeinMiner] Model mode – using {os.path.basename(best)}")
            else:
                minescript.echo("[VeinMiner] No model found, falling back to normal mode")
                mode = "normal"

    set_running(True)
    mine_loop(start_pos, target_name, mode, net)

if __name__ == "__main__":
    main()