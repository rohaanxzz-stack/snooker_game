import streamlit as st
import streamlit.components.v1 as components
import json
import math
import random

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Snooker AI",
    page_icon="🎱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── AI Engine (pure Python, no ML) ────────────────────────────────────────────
class SnookerAI:
    """
    Rule-based AI that simulates strategic snooker shot selection.
    Difficulty controls accuracy noise and planning depth.
    """
    def __init__(self, difficulty="medium"):
        self.difficulty = difficulty
        noise_map = {"easy": 0.18, "medium": 0.07, "hard": 0.02}
        self.noise = noise_map.get(difficulty, 0.07)

    def choose_shot(self, game_state):
        balls       = game_state["balls"]
        cue_ball    = next((b for b in balls if b["id"] == "cue"), None)
        target_color = game_state.get("target_color", "red")
        if not cue_ball:
            return self._default_shot()

        targets = self._get_valid_targets(balls, target_color, game_state)
        if not targets:
            targets = [b for b in balls if b["id"] != "cue" and not b.get("potted")]

        if not targets:
            return self._default_shot()

        best = self._evaluate_shots(cue_ball, targets, game_state)
        return self._add_noise(best)

    def _get_valid_targets(self, balls, target_color, game_state):
        if target_color == "red":
            return [b for b in balls if b.get("color") == "red" and not b.get("potted")]
        elif target_color == "color":
            on_red_count = sum(1 for b in balls if b.get("color") == "red" and not b.get("potted"))
            if on_red_count == 0:
                # pot colors in sequence
                seq = ["yellow","green","brown","blue","pink","black"]
                for c in seq:
                    t = [b for b in balls if b.get("color") == c and not b.get("potted")]
                    if t:
                        return t
            return [b for b in balls if b.get("color") != "red" and b["id"] != "cue" and not b.get("potted")]
        return [b for b in balls if b["id"] != "cue" and not b.get("potted")]

    def _evaluate_shots(self, cue_ball, targets, game_state):
        pockets = game_state.get("pockets", [])
        scored_shots = []
        for target in targets:
            for pocket in pockets:
                shot = self._compute_shot(cue_ball, target, pocket, game_state)
                if shot:
                    scored_shots.append(shot)

        if not scored_shots:
            # fallback: hit nearest target
            nearest = min(targets, key=lambda b: self._dist(cue_ball, b))
            dx = nearest["x"] - cue_ball["x"]
            dy = nearest["y"] - cue_ball["y"]
            angle = math.atan2(dy, dx)
            return {"angle": angle, "power": 0.55, "target_id": nearest["id"]}

        scored_shots.sort(key=lambda s: s["score"], reverse=True)
        return scored_shots[0]

    def _compute_shot(self, cue, target, pocket, game_state):
        # Ghost ball position: where cue must strike target to send it to pocket
        tx, ty = target["x"], target["y"]
        px, py = pocket["x"], pocket["y"]
        r = game_state.get("ball_radius", 10)

        dp = math.hypot(px - tx, py - ty)
        if dp < 1:
            return None
        # Ghost ball is offset along line from pocket through target
        gx = tx + r * 2 * (tx - px) / dp
        gy = ty + r * 2 * (ty - py) / dp

        # Angle from cue to ghost ball
        dx = gx - cue["x"]
        dy = gy - cue["y"]
        dist_cue_ghost = math.hypot(dx, dy)
        if dist_cue_ghost < 1:
            return None
        angle = math.atan2(dy, dx)

        # Score: prefer easy pots (straight, close) and valuable colors
        cut_angle = abs(self._cut_angle(cue, target, pocket))
        dist_penalty = dist_cue_ghost / 800
        angle_penalty = cut_angle / math.pi
        color_bonus = self._color_value(target.get("color","red")) / 7.0
        clearance = self._check_clearance(cue, gx, gy, game_state["balls"], target["id"])
        score = (1 - angle_penalty) * 0.5 + (1 - dist_penalty) * 0.3 + color_bonus * 0.2
        if not clearance:
            score -= 0.4

        dist_to_pocket = math.hypot(px - tx, py - ty)
        power = min(0.9, 0.35 + dist_cue_ghost / 1200 + dist_to_pocket / 1200)

        return {
            "angle": angle,
            "power": power,
            "target_id": target["id"],
            "score": score,
        }

    def _cut_angle(self, cue, target, pocket):
        v1 = (target["x"] - cue["x"], target["y"] - cue["y"])
        v2 = (pocket["x"] - target["x"], pocket["y"] - target["y"])
        a1 = math.atan2(v1[1], v1[0])
        a2 = math.atan2(v2[1], v2[0])
        diff = a2 - a1
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        return diff

    def _check_clearance(self, cue, gx, gy, balls, target_id):
        for b in balls:
            if b["id"] == "cue" or b["id"] == target_id or b.get("potted"):
                continue
            if self._point_near_segment(b["x"], b["y"], cue["x"], cue["y"], gx, gy, 12):
                return False
        return True

    def _point_near_segment(self, px, py, ax, ay, bx, by, tol):
        dx, dy = bx - ax, by - ay
        l2 = dx*dx + dy*dy
        if l2 == 0:
            return math.hypot(px-ax, py-ay) < tol
        t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / l2))
        nx, ny = ax + t*dx, ay + t*dy
        return math.hypot(px-nx, py-ny) < tol

    def _color_value(self, color):
        vals = {"red":1,"yellow":2,"green":3,"brown":4,"blue":5,"pink":6,"black":7}
        return vals.get(color, 1)

    def _dist(self, a, b):
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    def _add_noise(self, shot):
        shot["angle"] += random.gauss(0, self.noise)
        shot["power"]  = max(0.15, min(0.95, shot["power"] + random.gauss(0, self.noise * 0.3)))
        return shot

    def _default_shot(self):
        return {"angle": random.uniform(0, 2*math.pi), "power": 0.4, "target_id": None}


# ─── Session State Init ─────────────────────────────────────────────────────────
def init_state():
    if "game_initialized" not in st.session_state:
        st.session_state.game_initialized   = False
        st.session_state.game_phase         = "menu"  # menu | playing | game_over
        st.session_state.difficulty         = "medium"
        st.session_state.player_name        = "Player"
        st.session_state.scores             = {"player": 0, "cpu": 0}
        st.session_state.current_turn       = "player"
        st.session_state.message            = "Welcome to Snooker AI! Set up your game below."
        st.session_state.shot_result        = None
        st.session_state.ai_shot            = None
        st.session_state.frames_won         = {"player": 0, "cpu": 0}
        st.session_state.frames_to_win      = 3
        st.session_state.balls              = []
        st.session_state.target_color       = "red"
        st.session_state.consecutive_fouls  = 0
        st.session_state.reds_remaining     = 15
        st.session_state.colors_phase       = False
        st.session_state.shot_history       = []

init_state()


# ─── Ball Layout ────────────────────────────────────────────────────────────────
def create_balls():
    """Standard snooker ball positions (scaled to 780×440 table)."""
    W, H = 780, 440
    r = 9

    balls = []

    # Cue ball
    balls.append({"id":"cue","color":"white","x": W*0.22,"y": H*0.5,"potted":False})

    # Reds — triangle near pink spot
    cx, cy = W*0.72, H*0.5
    idx = 0
    for row in range(5):
        for col in range(row+1):
            bx = cx + row * (r*2+1)
            by = cy - row * (r+0.5) + col * (r*2+1)
            balls.append({"id":f"red_{idx}","color":"red","x":bx,"y":by,"potted":False})
            idx += 1

    # Colours on their spots
    colours = [
        ("yellow","yellow", W*0.355, H*0.639),
        ("green", "green",  W*0.355, H*0.361),
        ("brown", "brown",  W*0.355, H*0.5),
        ("blue",  "blue",   W*0.5,   H*0.5),
        ("pink",  "pink",   W*0.677, H*0.5),
        ("black", "black",  W*0.857, H*0.5),
    ]
    for (cid, col, cx2, cy2) in colours:
        balls.append({"id":cid,"color":col,"x":cx2,"y":cy2,"potted":False})

    return balls


# ─── Game Logic Helpers ─────────────────────────────────────────────────────────
def get_game_state():
    return {
        "balls":        st.session_state.balls,
        "target_color": st.session_state.target_color,
        "pockets":      get_pockets(),
        "ball_radius":  9,
    }

def get_pockets():
    W, H = 780, 440
    return [
        {"x":18,     "y":18},
        {"x":W//2,   "y":8},
        {"x":W-18,   "y":18},
        {"x":18,     "y":H-18},
        {"x":W//2,   "y":H-8},
        {"x":W-18,   "y":H-18},
    ]

def simulate_shot(angle, power, shooter):
    """
    Simplified physics: compute which balls are potted and update scores.
    Returns (potted_balls, foul, message).
    """
    balls    = st.session_state.balls
    cue      = next((b for b in balls if b["id"]=="cue"), None)
    if not cue:
        return [], True, "Cue ball missing!"

    r        = 9
    pockets  = get_pockets()
    W, H     = 780, 440
    margin   = 22

    speed    = power * 18
    vx       = math.cos(angle) * speed
    vy       = math.sin(angle) * speed

    cx, cy   = cue["x"], cue["y"]
    steps    = 220
    dt       = 1.0
    friction = 0.978

    # Detect first ball hit
    first_hit_id = None
    ball_vx  = {b["id"]: 0.0 for b in balls if not b.get("potted")}
    ball_vy  = {b["id"]: 0.0 for b in balls if not b.get("potted")}
    ball_x   = {b["id"]: b["x"] for b in balls if not b.get("potted")}
    ball_y   = {b["id"]: b["y"] for b in balls if not b.get("potted")}

    for step in range(steps):
        vx  *= friction
        vy  *= friction
        cx  += vx * dt
        cy  += vy * dt

        # Wall bounce
        if cx - r < margin:   cx = margin + r;   vx = abs(vx)
        if cx + r > W-margin: cx = W-margin - r; vx = -abs(vx)
        if cy - r < margin:   cy = margin + r;   vy = abs(vy)
        if cy + r > H-margin: cy = H-margin - r; vy = -abs(vy)

        # Cue pocket
        for pk in pockets:
            if math.hypot(cx - pk["x"], cy - pk["y"]) < r*1.6:
                return [], True, "⚠️ Cue ball potted — foul!"

        # Collision with other balls
        for b in balls:
            if b.get("potted") or b["id"] == "cue":
                continue
            bid = b["id"]
            bx2, by2 = ball_x[bid], ball_y[bid]
            dist = math.hypot(cx - bx2, cy - by2)
            if dist < r * 2 and dist > 0:
                if first_hit_id is None:
                    first_hit_id = bid

                nx = (bx2 - cx) / dist
                ny = (by2 - cy) / dist
                rel_v = vx * nx + vy * ny
                if rel_v > 0:
                    ball_vx[bid] += rel_v * nx * 0.92
                    ball_vy[bid] += rel_v * ny * 0.92
                    vx -= rel_v * nx
                    vy -= rel_v * ny

        # Move other balls
        for b in balls:
            if b.get("potted") or b["id"] == "cue":
                continue
            bid = b["id"]
            ball_vx[bid] *= friction
            ball_vy[bid] *= friction
            ball_x[bid]  += ball_vx[bid] * dt
            ball_y[bid]  += ball_vy[bid] * dt

            # Ball wall bounce
            bx2, by2 = ball_x[bid], ball_y[bid]
            if bx2 - r < margin:   ball_x[bid] = margin + r;   ball_vx[bid] = abs(ball_vx[bid])
            if bx2 + r > W-margin: ball_x[bid] = W-margin - r; ball_vx[bid] = -abs(ball_vx[bid])
            if by2 - r < margin:   ball_y[bid] = margin + r;   ball_vy[bid] = abs(ball_vy[bid])
            if by2 + r > H-margin: ball_y[bid] = H-margin - r; ball_vy[bid] = -abs(ball_vy[bid])

        if abs(vx) < 0.05 and abs(vy) < 0.05:
            break

    # Check which balls landed in pockets
    potted_ids = []
    for b in balls:
        if b.get("potted") or b["id"] == "cue":
            continue
        bid = b["id"]
        bx2, by2 = ball_x[bid], ball_y[bid]
        for pk in pockets:
            if math.hypot(bx2 - pk["x"], by2 - pk["y"]) < r * 1.8:
                potted_ids.append(bid)
                break

    # Update final positions
    for b in balls:
        if b.get("potted") or b["id"] == "cue":
            continue
        bid  = b["id"]
        b["x"] = ball_x[bid]
        b["y"] = ball_y[bid]
    cue["x"] = cx
    cue["y"] = cy

    # Rule validation
    target = st.session_state.target_color
    foul   = False
    msg    = ""

    if first_hit_id is None:
        foul = True
        msg  = "⚠️ No ball hit — foul! Opponent gets 4 points."
    else:
        first_ball = next((b for b in balls if b["id"] == first_hit_id), None)
        if first_ball:
            fc = first_ball.get("color","")
            if target == "red" and fc != "red":
                foul = True
                msg  = f"⚠️ Must hit red first — foul! Opponent gets 4 points."
            elif target == "color" and fc == "red":
                foul = True
                msg  = f"⚠️ Must hit a colour first — foul! Opponent gets 4 points."

    return potted_ids, foul, msg


def apply_shot_result(potted_ids, foul, msg, shooter):
    balls  = st.session_state.balls
    target = st.session_state.target_color
    score_key = "player" if shooter == "player" else "cpu"
    opponent  = "cpu" if shooter == "player" else "player"

    COLOR_VALS = {"red":1,"yellow":2,"green":3,"brown":4,"blue":5,"pink":6,"black":7}

    if foul:
        foul_pts = 4
        if potted_ids:
            for pid in potted_ids:
                b = next((bb for bb in balls if bb["id"] == pid), None)
                if b:
                    cv = COLOR_VALS.get(b.get("color","red"), 4)
                    foul_pts = max(foul_pts, cv)
                    if b.get("color") != "red":
                        b["potted"] = False  # colours go back
        st.session_state.scores[opponent] += foul_pts
        st.session_state.consecutive_fouls = (st.session_state.consecutive_fouls + 1) if shooter == "player" else 0
        st.session_state.current_turn = opponent
        st.session_state.message = msg + f" +{foul_pts} to {'CPU' if shooter=='player' else 'You'}."
        return

    st.session_state.consecutive_fouls = 0
    earned = 0
    potted_reds   = 0
    potted_colors = []

    for pid in potted_ids:
        b = next((bb for bb in balls if bb["id"] == pid), None)
        if b:
            c = b.get("color","red")
            earned += COLOR_VALS.get(c, 1)
            b["potted"] = True
            if c == "red":
                potted_reds += 1
                st.session_state.reds_remaining -= 1
            else:
                potted_colors.append(c)

    st.session_state.scores[score_key] += earned

    # Turn logic
    potted_legal = False
    if target == "red" and potted_reds > 0:
        potted_legal = True
        st.session_state.target_color = "color"
    elif target == "color" and potted_colors:
        potted_legal = True
        reds_left = st.session_state.reds_remaining
        if reds_left > 0:
            st.session_state.target_color = "red"
        else:
            st.session_state.colors_phase = True
            st.session_state.target_color = "color"
    
    who = "You" if shooter == "player" else "CPU"
    if earned > 0:
        st.session_state.message = f"{'🟢' if shooter=='player' else '🔴'} {who} scored {earned} pts! {'🎯 Continue!' if potted_legal else ''}"
    else:
        st.session_state.message = f"{'🟢' if shooter=='player' else '🔴'} {who} missed. Turn changes."

    if not potted_legal:
        st.session_state.current_turn = opponent

    # Check end of frame
    all_potted = all(b.get("potted") for b in balls if b["id"] != "cue")
    if all_potted or (st.session_state.reds_remaining == 0 and st.session_state.colors_phase
                      and len([b for b in balls if b["id"] != "cue" and not b.get("potted")]) == 0):
        end_frame()


def end_frame():
    p = st.session_state.scores["player"]
    c = st.session_state.scores["cpu"]
    winner = "player" if p > c else "cpu"
    st.session_state.frames_won[winner] += 1
    st.session_state.message = (
        f"🏆 Frame over! {'You win' if winner=='player' else 'CPU wins'} the frame! "
        f"({p} – {c}). Frames: You {st.session_state.frames_won['player']} – CPU {st.session_state.frames_won['cpu']}"
    )
    if st.session_state.frames_won[winner] >= st.session_state.frames_to_win:
        st.session_state.game_phase = "game_over"
        st.session_state.message = (
            f"🎱 MATCH OVER! {'You win' if winner=='player' else 'CPU wins'} the match "
            f"({st.session_state.frames_won['player']}–{st.session_state.frames_won['cpu']})!"
        )
    else:
        new_frame()


def new_frame():
    st.session_state.balls             = create_balls()
    st.session_state.scores            = {"player": 0, "cpu": 0}
    st.session_state.target_color      = "red"
    st.session_state.reds_remaining    = 15
    st.session_state.colors_phase      = False
    st.session_state.current_turn      = "player"
    st.session_state.consecutive_fouls = 0


# ─── HTML Canvas Renderer ────────────────────────────────────────────────────────
def render_table(balls, ai_shot_data=None, mode="player"):
    balls_json   = json.dumps(balls)
    pockets_json = json.dumps(get_pockets())
    ai_json      = json.dumps(ai_shot_data) if ai_shot_data else "null"
    is_player    = "true" if mode == "player" else "false"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin:0; padding:0; }}
  body {{ background:#0d1b0e; font-family:'Segoe UI',sans-serif; user-select:none; }}
  #wrap {{ display:flex; flex-direction:column; align-items:center; padding:10px; }}
  canvas {{ border-radius:8px; box-shadow:0 0 32px #000a; cursor:crosshair; display:block; }}
  #controls {{
    margin-top:12px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; justify-content:center;
  }}
  #power-wrap {{ display:flex; flex-direction:column; align-items:center; gap:4px; }}
  label {{ color:#c8e6c9; font-size:13px; }}
  input[type=range] {{ width:180px; accent-color:#66bb6a; }}
  button {{
    padding:9px 22px; border:none; border-radius:6px; font-size:14px; font-weight:700;
    cursor:pointer; transition:all .18s;
  }}
  #shoot-btn {{ background:#2e7d32; color:#fff; }}
  #shoot-btn:hover {{ background:#388e3c; }}
  #shoot-btn:disabled {{ background:#444; color:#888; cursor:not-allowed; }}
  #hint-btn {{ background:#1565c0; color:#fff; }}
  #hint-btn:hover {{ background:#1976d2; }}
  #info {{ color:#a5d6a7; font-size:12px; margin-top:6px; }}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="table" width="780" height="440"></canvas>
  <div id="controls">
    <div id="power-wrap">
      <label>Power: <span id="power-val">50</span>%</label>
      <input type="range" id="power" min="5" max="100" value="50">
    </div>
    <button id="shoot-btn" {'disabled' if mode != 'player' else ''}>🎱 Shoot</button>
    <button id="hint-btn">💡 Hint</button>
  </div>
  <div id="info">Click on table to aim · Drag slider to set power</div>
</div>

<script>
const BALLS   = {balls_json};
const POCKETS = {pockets_json};
const AI_SHOT = {ai_json};
const IS_PLAYER = {is_player};
const R = 9;
const W = 780, H = 440, MARGIN = 22;

const canvas = document.getElementById('table');
const ctx    = canvas.getContext('2d');
const powerSlider = document.getElementById('power');
const powerVal    = document.getElementById('power-val');
const shootBtn    = document.getElementById('shoot-btn');
const hintBtn     = document.getElementById('hint-btn');

let aimAngle  = 0;
let showHint  = false;
let animating = false;

// Ball rendering colours
const BALL_COLORS = {{
  white:'#f5f5f5', red:'#c62828', yellow:'#f9a825', green:'#2e7d32',
  brown:'#6d4c41', blue:'#1565c0', pink:'#e91e8c', black:'#212121'
}};
const BALL_STRIPES = {{
  white:null, red:null, yellow:null, green:null,
  brown:null, blue:null, pink:null, black:null
}};

powerSlider.addEventListener('input', () => {{
  powerVal.textContent = powerSlider.value;
  drawAll();
}});

canvas.addEventListener('mousemove', e => {{
  if (!IS_PLAYER || animating) return;
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (W / rect.width);
  const my = (e.clientY - rect.top)  * (H / rect.height);
  const cue = BALLS.find(b => b.id === 'cue');
  if (cue) {{
    aimAngle = Math.atan2(my - cue.y, mx - cue.x);
    drawAll();
  }}
}});

shootBtn.addEventListener('click', () => {{
  if (!IS_PLAYER || animating) return;
  const power = parseInt(powerSlider.value) / 100;
  window.parent.postMessage({{
    type: 'snooker_shot',
    angle: aimAngle,
    power: power
  }}, '*');
  animateShot(aimAngle, power, () => {{}});
}});

hintBtn.addEventListener('click', () => {{
  showHint = !showHint;
  drawAll();
}});

// ─── Drawing ───────────────────────────────────────────────────────────────────
function drawTable() {{
  // Felt
  const grad = ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0,'#1b5e20');
  grad.addColorStop(1,'#2e7d32');
  ctx.fillStyle = grad;
  ctx.fillRect(0,0,W,H);

  // Cushions
  ctx.strokeStyle = '#4a2c0a';
  ctx.lineWidth   = MARGIN;
  ctx.strokeRect(MARGIN/2, MARGIN/2, W-MARGIN, H-MARGIN);

  // Baulk line
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(W*0.355, MARGIN);
  ctx.lineTo(W*0.355, H-MARGIN);
  ctx.stroke();

  // D semi-circle
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.arc(W*0.355, H*0.5, H*0.139, -Math.PI/2, Math.PI/2);
  ctx.stroke();

  // Spots
  const spots = [
    [W*0.355, H*0.639],
    [W*0.355, H*0.361],
    [W*0.355, H*0.5],
    [W*0.5,   H*0.5],
    [W*0.677, H*0.5],
    [W*0.857, H*0.5],
  ];
  spots.forEach(([sx,sy]) => {{
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.beginPath();
    ctx.arc(sx, sy, 2.5, 0, Math.PI*2);
    ctx.fill();
  }});

  // Pockets
  POCKETS.forEach(p => {{
    ctx.fillStyle   = '#0a0a0a';
    ctx.strokeStyle = '#6d4c41';
    ctx.lineWidth   = 2;
    ctx.beginPath();
    ctx.arc(p.x, p.y, R*1.5, 0, Math.PI*2);
    ctx.fill();
    ctx.stroke();
  }});
}}

function drawBalls(overrides) {{
  const state = overrides || {{}};
  BALLS.forEach(b => {{
    if (b.potted) return;
    const bx = state[b.id] ? state[b.id].x : b.x;
    const by = state[b.id] ? state[b.id].y : b.y;
    const col = BALL_COLORS[b.color] || '#fff';

    // Shadow
    ctx.fillStyle = 'rgba(0,0,0,0.35)';
    ctx.beginPath();
    ctx.ellipse(bx+2, by+4, R*0.9, R*0.5, 0, 0, Math.PI*2);
    ctx.fill();

    // Ball
    const g = ctx.createRadialGradient(bx-R*0.3, by-R*0.35, R*0.08, bx, by, R);
    g.addColorStop(0, lighten(col, 55));
    g.addColorStop(0.5, col);
    g.addColorStop(1, darken(col, 40));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(bx, by, R, 0, Math.PI*2);
    ctx.fill();

    // Highlight
    ctx.fillStyle = 'rgba(255,255,255,0.45)';
    ctx.beginPath();
    ctx.ellipse(bx-R*0.28, by-R*0.32, R*0.22, R*0.14, -0.5, 0, Math.PI*2);
    ctx.fill();

    // Label for non-cue balls
    if (b.color !== 'white') {{
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.font = `bold ${{R*0.9}}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const labels = {{red:'●',yellow:'Y',green:'G',brown:'Br',blue:'B',pink:'P',black:'Bk'}};
      // skip label for red to avoid clutter
    }}
  }});
}}

function drawAimLine() {{
  const cue = BALLS.find(b => b.id === 'cue');
  if (!cue || cue.potted) return;
  const power = parseInt(powerSlider.value) / 100;
  const len   = 60 + power * 120;

  // Dashed aim line
  ctx.save();
  ctx.setLineDash([6,5]);
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.moveTo(cue.x, cue.y);
  ctx.lineTo(cue.x + Math.cos(aimAngle)*len, cue.y + Math.sin(aimAngle)*len);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // Cue stick
  const cueLen = 90;
  const ex = cue.x - Math.cos(aimAngle)*(R+4);
  const ey = cue.y - Math.sin(aimAngle)*(R+4);
  const sx = ex - Math.cos(aimAngle)*cueLen;
  const sy = ey - Math.sin(aimAngle)*cueLen;
  const cueGrad = ctx.createLinearGradient(sx,sy,ex,ey);
  cueGrad.addColorStop(0,'#8d6e63');
  cueGrad.addColorStop(1,'#f5f5f5');
  ctx.strokeStyle = cueGrad;
  ctx.lineWidth   = 5;
  ctx.lineCap     = 'round';
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  ctx.lineCap = 'butt';
}}

function drawHint() {{
  if (!showHint) return;
  // Show ghost-ball hint for nearest red/target
  const cue = BALLS.find(b=>b.id==='cue');
  if (!cue) return;
  const targets = BALLS.filter(b=>!b.potted && b.id!=='cue' && b.color==='red');
  if (!targets.length) return;
  const nearest = targets.reduce((a,b) =>
    Math.hypot(a.x-cue.x,a.y-cue.y) < Math.hypot(b.x-cue.x,b.y-cue.y) ? a : b);
  const pk = POCKETS.reduce((a,b) =>
    Math.hypot(a.x-nearest.x,a.y-nearest.y) < Math.hypot(b.x-nearest.x,b.y-nearest.y) ? a : b);
  const dp = Math.hypot(pk.x-nearest.x, pk.y-nearest.y);
  const gx = nearest.x + R*2*(nearest.x-pk.x)/dp;
  const gy = nearest.y + R*2*(nearest.y-pk.y)/dp;

  ctx.globalAlpha = 0.4;
  ctx.strokeStyle = '#fff176';
  ctx.lineWidth   = 1.5;
  ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.arc(gx,gy,R,0,Math.PI*2); ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(cue.x,cue.y); ctx.lineTo(gx,gy);
  ctx.stroke();
  ctx.globalAlpha = 1;
}}

function drawAIShot() {{
  if (!AI_SHOT) return;
  const cue = BALLS.find(b=>b.id==='cue');
  if (!cue) return;
  const ang = AI_SHOT.angle;
  const len = 150;
  ctx.save();
  ctx.globalAlpha = 0.6;
  ctx.setLineDash([8,5]);
  ctx.strokeStyle = '#ff7043';
  ctx.lineWidth   = 2;
  ctx.beginPath();
  ctx.moveTo(cue.x, cue.y);
  ctx.lineTo(cue.x + Math.cos(ang)*len, cue.y + Math.sin(ang)*len);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}}

function drawAll(overrides) {{
  ctx.clearRect(0,0,W,H);
  drawTable();
  drawBalls(overrides);
  if (IS_PLAYER && !animating) drawAimLine();
  drawHint();
  if (!IS_PLAYER) drawAIShot();
}}

// ─── Shot Animation ─────────────────────────────────────────────────────────────
function animateShot(angle, power, done) {{
  animating = true;
  if (IS_PLAYER) shootBtn.disabled = true;

  const speed = power * 18;
  let vx = Math.cos(angle)*speed;
  let vy = Math.sin(angle)*speed;
  const friction = 0.978;
  const cue = BALLS.find(b=>b.id==='cue');
  if (!cue) {{ done(); return; }}

  // Live positions
  const pos = {{}};
  BALLS.forEach(b => {{ if (!b.potted) pos[b.id] = {{x:b.x, y:b.y, vx:0, vy:0}}; }});
  let cx = cue.x, cy = cue.y;

  let firstHit = false;
  let frame = 0;
  const MAX = 260;

  function step() {{
    vx *= friction; vy *= friction;
    cx += vx; cy += vy;
    if (cx-R<MARGIN)  {{cx=MARGIN+R; vx=Math.abs(vx);}}
    if (cx+R>W-MARGIN){{cx=W-MARGIN-R; vx=-Math.abs(vx);}}
    if (cy-R<MARGIN)  {{cy=MARGIN+R; vy=Math.abs(vy);}}
    if (cy+R>H-MARGIN){{cy=H-MARGIN-R; vy=-Math.abs(vy);}}

    Object.keys(pos).forEach(bid => {{
      if (bid==='cue') return;
      const bp = pos[bid];
      bp.vx *= friction; bp.vy *= friction;
      bp.x  += bp.vx;   bp.y  += bp.vy;
      if (bp.x-R<MARGIN)  {{bp.x=MARGIN+R; bp.vx=Math.abs(bp.vx);}}
      if (bp.x+R>W-MARGIN){{bp.x=W-MARGIN-R; bp.vx=-Math.abs(bp.vx);}}
      if (bp.y-R<MARGIN)  {{bp.y=MARGIN+R; bp.vy=Math.abs(bp.vy);}}
      if (bp.y+R>H-MARGIN){{bp.y=H-MARGIN-R; bp.vy=-Math.abs(bp.vy);}}

      // Collision with cue
      const d = Math.hypot(cx-bp.x, cy-bp.y);
      if (d < R*2 && d > 0) {{
        firstHit = true;
        const nx=(bp.x-cx)/d, ny=(bp.y-cy)/d;
        const rv=vx*nx+vy*ny;
        if(rv>0){{
          bp.vx+=rv*nx*0.92; bp.vy+=rv*ny*0.92;
          vx-=rv*nx; vy-=rv*ny;
        }}
      }}
    }});

    const overrides = {{}};
    Object.keys(pos).forEach(bid => {{ overrides[bid] = pos[bid]; }});
    overrides['cue'] = {{x:cx, y:cy}};
    drawAll(overrides);

    frame++;
    const stillMoving = Math.abs(vx)>0.15 || Math.abs(vy)>0.15 ||
      Object.values(pos).some(p=>Math.abs(p.vx)>0.15||Math.abs(p.vy)>0.15);

    if (frame < MAX && stillMoving) {{
      requestAnimationFrame(step);
    }} else {{
      animating = false;
      if (IS_PLAYER) shootBtn.disabled = false;
      done();
    }}
  }}
  requestAnimationFrame(step);
}}

// Colour helpers
function lighten(hex, amt) {{
  const n = parseInt(hex.slice(1),16);
  let r=Math.min(255,((n>>16)&0xff)+amt);
  let g=Math.min(255,((n>>8)&0xff)+amt);
  let b=Math.min(255,(n&0xff)+amt);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function darken(hex, amt) {{
  const n = parseInt(hex.slice(1),16);
  let r=Math.max(0,((n>>16)&0xff)-amt);
  let g=Math.max(0,((n>>8)&0xff)-amt);
  let b=Math.max(0,(n&0xff)-amt);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

// Auto-play AI shot animation
if (!IS_PLAYER && AI_SHOT) {{
  setTimeout(() => {{
    animateShot(AI_SHOT.angle, AI_SHOT.power, () => {{
      window.parent.postMessage({{
        type:'snooker_ai_done',
        angle: AI_SHOT.angle,
        power: AI_SHOT.power
      }}, '*');
    }});
  }}, 800);
}}

drawAll();
</script>
</body>
</html>
"""
    return html


# ─── Streamlit UI ───────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style="text-align:center; padding:8px 0 4px">
      <span style="font-size:2.2rem;">🎱</span>
      <span style="font-size:1.8rem; font-weight:800; color:#a5d6a7; letter-spacing:2px;">
        &nbsp;SNOOKER AI
      </span>
      <span style="color:#81c784; font-size:0.85rem; display:block; margin-top:2px;">
        Rule-Based AI Opponent · No External APIs
      </span>
    </div>
    """, unsafe_allow_html=True)


def render_scoreboard():
    p  = st.session_state.scores["player"]
    c  = st.session_state.scores["cpu"]
    fp = st.session_state.frames_won["player"]
    fc = st.session_state.frames_won["cpu"]
    t  = st.session_state.current_turn
    tgt = st.session_state.target_color.upper()
    name = st.session_state.player_name

    st.markdown(f"""
    <div style="display:flex; justify-content:center; gap:24px; padding:6px 0 10px;
                flex-wrap:wrap; font-family:'Segoe UI',sans-serif;">
      <div style="background:{'#1b5e20' if t=='player' else '#1a1a1a'};
                  border:2px solid {'#66bb6a' if t=='player' else '#444'};
                  border-radius:10px; padding:10px 28px; text-align:center; min-width:140px;">
        <div style="color:#a5d6a7; font-size:11px; text-transform:uppercase; letter-spacing:1px;">
          {'▶ ' if t=='player' else ''}{name}
        </div>
        <div style="color:#fff; font-size:2rem; font-weight:800;">{p}</div>
        <div style="color:#81c784; font-size:11px;">Frames: {fp}</div>
      </div>
      <div style="display:flex; flex-direction:column; justify-content:center; align-items:center; gap:4px;">
        <div style="color:#c8e6c9; font-size:11px;">Target</div>
        <div style="background:#333; border-radius:6px; padding:4px 12px;
                    color:#ffcc02; font-size:13px; font-weight:700;">{tgt}</div>
        <div style="color:#888; font-size:10px;">Reds: {st.session_state.reds_remaining}</div>
      </div>
      <div style="background:{'#b71c1c' if t=='cpu' else '#1a1a1a'};
                  border:2px solid {'#ef5350' if t=='cpu' else '#444'};
                  border-radius:10px; padding:10px 28px; text-align:center; min-width:140px;">
        <div style="color:#ffcdd2; font-size:11px; text-transform:uppercase; letter-spacing:1px;">
          {'▶ ' if t=='cpu' else ''}CPU ({st.session_state.difficulty.title()})
        </div>
        <div style="color:#fff; font-size:2rem; font-weight:800;">{c}</div>
        <div style="color:#ef9a9a; font-size:11px;">Frames: {fc}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_message():
    msg = st.session_state.message
    color = "#a5d6a7"
    if "foul" in msg.lower() or "⚠️" in msg:
        color = "#ffcc02"
    elif "win" in msg.lower() or "🏆" in msg:
        color = "#ffd54f"
    st.markdown(f"""
    <div style="text-align:center; background:#1a2a1a; border-radius:8px;
                padding:8px 16px; margin:4px auto; max-width:700px;
                color:{color}; font-size:14px; font-weight:600; border:1px solid #2e4a2e;">
      {msg}
    </div>
    """, unsafe_allow_html=True)


# ─── Main App ───────────────────────────────────────────────────────────────────
render_header()

# ── MENU PHASE ──────────────────────────────────────────────────────────────────
if st.session_state.game_phase == "menu":
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("""
            <div style="background:#0f2210; border:1px solid #2e7d32; border-radius:14px;
                        padding:30px; text-align:center;">
              <div style="color:#a5d6a7; font-size:1.1rem; margin-bottom:20px;">
                🎱 Set up your match
              </div>
            """, unsafe_allow_html=True)

            name = st.text_input("Your Name", value="Player", max_chars=20)
            diff = st.selectbox("AI Difficulty", ["easy", "medium", "hard"],
                                index=1,
                                format_func=lambda x: {"easy":"🟢 Easy","medium":"🟡 Medium","hard":"🔴 Hard"}[x])
            frames = st.selectbox("Frames to Win", [1, 2, 3, 4, 5], index=2)

            st.markdown("""
            <div style="color:#81c784; font-size:12px; margin:12px 0; line-height:1.7;">
              <b>How to play:</b><br>
              🖱️ Move mouse to aim · 🎚️ Slider for power · 🎱 Click Shoot<br>
              Pot a red, then a colour, alternating<br>
              💡 Hint button shows suggested shot
            </div>
            """, unsafe_allow_html=True)

            if st.button("🚀 Start Match", use_container_width=True, type="primary"):
                st.session_state.player_name    = name
                st.session_state.difficulty     = diff
                st.session_state.frames_to_win  = frames
                st.session_state.frames_won     = {"player":0,"cpu":0}
                st.session_state.balls          = create_balls()
                st.session_state.scores         = {"player":0,"cpu":0}
                st.session_state.target_color   = "red"
                st.session_state.reds_remaining = 15
                st.session_state.colors_phase   = False
                st.session_state.current_turn   = "player"
                st.session_state.message        = f"🎱 Match started! {name} breaks first. Pot a red ball."
                st.session_state.game_phase     = "playing"
                st.session_state.ai_shot        = None
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


# ── GAME OVER PHASE ─────────────────────────────────────────────────────────────
elif st.session_state.game_phase == "game_over":
    render_scoreboard()
    render_message()
    st.markdown("<br>", unsafe_allow_html=True)

    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        fp = st.session_state.frames_won["player"]
        fc = st.session_state.frames_won["cpu"]
        winner_text = "🎉 You won the match!" if fp > fc else "🤖 CPU wins the match!"
        st.markdown(f"""
        <div style="background:#0f2210; border:2px solid #4caf50; border-radius:14px;
                    padding:30px; text-align:center; color:#a5d6a7;">
          <div style="font-size:1.8rem; margin-bottom:10px;">{winner_text}</div>
          <div style="font-size:1.1rem; color:#fff;">{fp} – {fc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Play Again", use_container_width=True, type="primary"):
            st.session_state.game_phase  = "menu"
            st.session_state.frames_won  = {"player":0,"cpu":0}
            st.session_state.ai_shot     = None
            st.rerun()


# ── PLAYING PHASE ───────────────────────────────────────────────────────────────
elif st.session_state.game_phase == "playing":
    render_scoreboard()
    render_message()

    turn   = st.session_state.current_turn
    balls  = st.session_state.balls
    ai_sht = st.session_state.ai_shot

    # Render table
    table_html = render_table(
        balls,
        ai_shot_data=ai_sht if turn == "cpu" else None,
        mode=turn,
    )
    components.html(table_html, height=520, scrolling=False)

    # Control row
    col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])

    with col_a:
        if turn == "player":
            angle_deg = st.slider("🎯 Aim Angle (°)", 0, 359, 0, key="aim_angle_slider")
            power_pct = st.slider("💪 Power (%)", 5, 100, 50, key="power_slider")
            angle_rad = math.radians(angle_deg)
        else:
            st.markdown("""
            <div style="color:#ef9a9a; font-size:14px; padding:10px;">
              🤖 CPU is thinking…
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        if turn == "player":
            if st.button("🎱 Shoot!", use_container_width=True, type="primary"):
                potted, foul, msg = simulate_shot(angle_rad, power_pct/100, "player")
                apply_shot_result(potted, foul, msg, "player")
                st.session_state.ai_shot = None
                if st.session_state.current_turn == "cpu" and st.session_state.game_phase == "playing":
                    ai   = SnookerAI(st.session_state.difficulty)
                    shot = ai.choose_shot(get_game_state())
                    st.session_state.ai_shot = shot
                st.rerun()

    with col_c:
        if turn == "cpu" and ai_sht:
            if st.button("▶ Execute CPU Shot", use_container_width=True):
                shot = st.session_state.ai_shot
                potted, foul, msg = simulate_shot(shot["angle"], shot["power"], "cpu")
                apply_shot_result(potted, foul, msg, "cpu")
                st.session_state.ai_shot = None
                if st.session_state.current_turn == "cpu" and st.session_state.game_phase == "playing":
                    ai   = SnookerAI(st.session_state.difficulty)
                    nxt  = ai.choose_shot(get_game_state())
                    st.session_state.ai_shot = nxt
                st.rerun()
        elif turn == "cpu" and not ai_sht:
            if st.button("🤖 Generate CPU Shot", use_container_width=True):
                ai   = SnookerAI(st.session_state.difficulty)
                shot = ai.choose_shot(get_game_state())
                st.session_state.ai_shot = shot
                st.rerun()

    with col_d:
        if st.button("🔄 New Frame", use_container_width=True):
            new_frame()
            st.session_state.message = "New frame! Player breaks."
            st.session_state.ai_shot = None
            st.rerun()

    # Sidebar extras
    with st.sidebar:
        st.markdown("### 🎱 Snooker AI")
        st.markdown(f"**Match:** Best of {st.session_state.frames_to_win*2-1}")
        st.markdown(f"**Difficulty:** {st.session_state.difficulty.title()}")
        st.divider()
        st.markdown("**Ball Values:**")
        for name, val, col in [
            ("Red","1","🔴"),("Yellow","2","🟡"),("Green","3","🟢"),
            ("Brown","4","🟤"),("Blue","5","🔵"),("Pink","6","🩷"),("Black","7","⚫")
        ]:
            st.markdown(f"{col} {name}: **{val} pts**")
        st.divider()
        st.markdown("**Rules:**")
        st.markdown("- Pot red → then colour\n- Foul = opponent gets min 4 pts\n- Pot cue ball = foul")
        st.divider()
        if st.button("🏠 Main Menu"):
            st.session_state.game_phase = "menu"
            st.session_state.ai_shot    = None
            st.rerun()
