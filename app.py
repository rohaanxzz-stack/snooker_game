import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import tensorflow as tf
import time
import math

# Force TensorFlow to execute strictly in CPU mode to prevent threading crashes on Streamlit Cloud
tf.config.set_visible_devices([], 'GPU')

# Page Initialization
st.set_page_config(page_title="TF 2.13 Snooker Engine", layout="wide")

# Simulation Constants
TABLE_WIDTH = 800
TABLE_HEIGHT = 400
BALL_RADIUS = 10
FRICTION = 0.985
BOUNCE_DAMPING = 0.9

POCKETS = [
    (10, 10), (TABLE_WIDTH // 2, 10), (TABLE_WIDTH - 10, 10),
    (10, TABLE_HEIGHT - 10), (TABLE_WIDTH // 2, TABLE_HEIGHT - 10), (TABLE_WIDTH - 10, TABLE_HEIGHT - 10)
]
POCKET_RADIUS = 18

COLOR_MAP = {
    "cue": "#FFFFFF", "red": "#FF0000", "yellow": "#FFFF00", 
    "green": "#00FF00", "black": "#000000"
}

# --- State System Framework ---
if "balls" not in st.session_state:
    st.session_state.balls = []
    st.session_state.score = {"Player": 0, "AI": 0}
    st.session_state.turn = "Player"
    st.session_state.game_over = False
    st.session_state.training_episodes = 0
    
    # Position Elements
    st.session_state.balls.append({"id": "cue", "x": 200.0, "y": 200.0, "vx": 0.0, "vy": 0.0, "color": COLOR_MAP["cue"]})
    object_balls = [
        ("red", 550.0, 200.0), ("red", 570.0, 190.0), ("red", 570.0, 210.0),
        ("yellow", 520.0, 150.0), ("green", 520.0, 250.0), ("black", 650.0, 200.0)
    ]
    for b_type, x, y in object_balls:
        st.session_state.balls.append({"id": b_type, "x": x, "y": y, "vx": 0.0, "vy": 0.0, "color": COLOR_MAP[b_type]})

# --- Native TensorFlow Brain Architecture ---
class PureTFBrain(tf.Module):
    def __init__(self):
        super().__init__()
        self.W1 = tf.Variable(tf.random.normal([8, 64], stddev=0.1, dtype=tf.float32), name="W1")
        self.b1 = tf.Variable(tf.zeros([64], dtype=tf.float32), name="b1")
        self.W2 = tf.Variable(tf.random.normal([64, 16], stddev=0.1, dtype=tf.float32), name="W2")
        self.b2 = tf.Variable(tf.zeros([16], dtype=tf.float32), name="b2")
        self.optimizer = tf.optimizers.Adam(learning_rate=0.005)

    @tf.Module.with_name_scope
    def predict(self, state_tensor):
        layer_1 = tf.nn.relu(tf.matmul(state_tensor, self.W1) + self.b1)
        q_values = tf.matmul(layer_1, self.W2) + self.b2
        return q_values

    def train_step(self, state, action, target_val):
        with tf.GradientTape() as tape:
            q_values = self.predict(state)
            one_hot_mask = tf.one_hot(action, 16, dtype=tf.float32)
            predicted_q = tf.reduce_sum(q_values * one_hot_mask, axis=1)
            loss = tf.reduce_mean(tf.square(tf.stop_gradient(target_val) - predicted_q))
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

if "tf_brain" not in st.session_state:
    st.session_state.tf_brain = PureTFBrain()

# --- Vector Mechanics Engine ---
def resolve_collisions():
    balls = st.session_state.balls
    for b in balls:
        if b["x"] - BALL_RADIUS < 0:
            b["x"] = BALL_RADIUS
            b["vx"] *= -BOUNCE_DAMPING
        elif b["x"] + BALL_RADIUS > TABLE_WIDTH:
            b["x"] = TABLE_WIDTH - BALL_RADIUS
            b["vx"] *= -BOUNCE_DAMPING
        if b["y"] - BALL_RADIUS < 0:
            b["y"] = BALL_RADIUS
            b["vy"] *= -BOUNCE_DAMPING
        elif b["y"] + BALL_RADIUS > TABLE_HEIGHT:
            b["y"] = TABLE_HEIGHT - BALL_RADIUS
            b["vy"] *= -BOUNCE_DAMPING

    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            b1, b2 = balls[i], balls[j]
            dx, dy = b2["x"] - b1["x"], b2["y"] - b1["y"]
            dist = math.hypot(dx, dy)
            if dist < BALL_RADIUS * 2:
                overlap = (BALL_RADIUS * 2) - dist
                nx, ny = dx / (dist or 1), dy / (dist or 1)
                b1["x"] -= nx * overlap * 0.5
                b1["y"] -= ny * overlap * 0.5
                b2["x"] += nx * overlap * 0.5
                b2["y"] += ny * overlap * 0.5
                kx, ky = b1["vx"] - b2["vx"], b1["vy"] - b2["vy"]
                p = nx * kx + ny * ky
                if p > 0:
                    b1["vx"] -= nx * p
                    b1["vy"] -= ny * p
                    b2["vx"] += nx * p
                    b2["vy"] += ny * p

def run_physics_loop(rendering_element=None):
    balls = st.session_state.balls
    scored_this_turn = 0
    scratched = False
    
    for _ in range(150):
        moving = False
        for b in balls:
            b["x"] += b["vx"]
            b["y"] += b["vy"]
            b["vx"] *= FRICTION
            b["vy"] *= FRICTION
            if abs(b["vx"]) < 0.1: b["vx"] = 0.0
            if abs(b["vy"]) < 0.1: b["vy"] = 0.0
            if b["vx"] != 0 or b["vy"] != 0: moving = True
            
        resolve_collisions()
        
        active_balls = []
        for b in balls:
            is_pocketed = False
            for px, py in POCKETS:
                if math.hypot(b["x"] - px, b["y"] - py) < POCKET_RADIUS:
                    is_pocketed = True
                    break
            if is_pocketed:
                if b["id"] == "cue":
                    scratched = True
                else:
                    scored_this_turn += 1
                    st.session_state.score[st.session_state.turn] += 1
            else:
                active_balls.append(b)
                
        st.session_state.balls = active_balls
        balls = active_balls
        
        if rendering_element and moving:
            render_canvas(rendering_element)
            time.sleep(0.01)
            
        if not moving:
            break
            
    if scratched:
        st.session_state.balls.append({"id": "cue", "x": 200.0, "y": 200.0, "vx": 0.0, "vy": 0.0, "color": COLOR_MAP["cue"]})
    return scored_this_turn, scratched

# --- Feature Vector Extraction Matrix ---
def get_current_state_vector():
    balls = st.session_state.balls
    cue = next((b for b in balls if b["id"] == "cue"), {"x": 200.0, "y": 200.0})
    objs = [b for b in balls if b["id"] != "cue"]
    
    if objs:
        closest_obj = min(objs, key=lambda b: math.hypot(b["x"] - cue["x"], b["y"] - cue["y"]))
        ox, oy = closest_obj["x"], closest_obj["y"]
    else:
        ox, oy = TABLE_WIDTH / 2, TABLE_HEIGHT / 2
        
    closest_pocket = min(POCKETS, key=lambda p: math.hypot(p[0] - ox, p[1] - oy))
    
    state_array = np.array([[ 
        cue["x"] / TABLE_WIDTH, cue["y"] / TABLE_HEIGHT,
        ox / TABLE_WIDTH, oy / TABLE_HEIGHT,
        closest_pocket[0] / TABLE_WIDTH, closest_pocket[1] / TABLE_HEIGHT,
        st.session_state.score["AI"] / 10.0, st.session_state.score["Player"] / 10.0
    ]], dtype=np.float32)
    return tf.convert_to_tensor(state_array, dtype=tf.float32)

def execute_tf_ai_turn(visualize=True):
    balls = st.session_state.balls
    cue = next((b for b in balls if b["id"] == "cue"), None)
    if not cue: return
    
    state = get_current_state_vector()
    
    epsilon = max(0.1, 1.0 - (st.session_state.training_episodes * 0.05))
    if np.random.rand() < epsilon:
        action_idx = np.random.randint(0, 16)
    else:
        q_vals = st.session_state.tf_brain.predict(state)
        action_idx = int(tf.argmax(q_vals[0]).numpy())
        
    target_angle = (action_idx / 16.0) * 2 * math.pi
    cue["vx"] = math.cos(target_angle) * 15.0
    cue["vy"] = math.sin(target_angle) * 15.0
    
    scored, scratched = run_physics_loop(render_placeholder if visualize else None)
    
    reward = (scored * 5.0) - (1.0 if scratched else 0.0) + (0.1 if scored > 0 else -0.1)
    
    next_state = get_current_state_vector()
    next_q_vals = st.session_state.tf_brain.predict(next_state)
    max_next_q = tf.reduce_max(next_q_vals[0])
    
    target_q = tf.convert_to_tensor([reward + 0.95 * max_next_q.numpy()], dtype=tf.float32)
    st.session_state.tf_brain.train_step(state, tf.convert_to_tensor([action_idx], dtype=tf.int32), target_q)

# --- Version Safe Canvas Generation ---
def render_canvas(element_handle):
    balls_js = ", ".join([f'{{x: {b["x"]}, y: {b["y"]}, c: "{b["color"]}"}}' for b in st.session_state.balls])
    pockets_js = ", ".join([f'{{x: {p[0]}, y: {p[1]}}}' for p in POCKETS])
    
    html_data = f"""
    <div style="text-align: center; margin: 0; padding: 0;">
        <canvas id="snookerCanvas" width="{TABLE_WIDTH}" height="{TABLE_HEIGHT}" style="background-color:#1e5631; border:12px solid #4a2c11; border-radius:8px; box-shadow: 0px 4px 12px rgba(0,0,0,0.4);"></canvas>
    </div>
    <script>
        var canvas = document.getElementById('snookerCanvas');
        var ctx = canvas.getContext('2d');
        var balls = [{balls_js}];
        var pockets = [{pockets_js}];
        
        ctx.clearRect(0,0, {TABLE_WIDTH}, {TABLE_HEIGHT});
        
        pockets.forEach(p => {{
            ctx.beginPath();
            ctx.arc(p.x, p.y, {POCKET_RADIUS}, 0, 2*Math.PI);
            ctx.fillStyle = '#111';
            ctx.fill();
        }});
        
        balls.forEach(b => {{
            ctx.beginPath();
            ctx.arc(b.x, b.y, {BALL_RADIUS}, 0, 2*Math.PI);
            ctx.fillStyle = b.c;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }});
    </script>
    """
    with element_handle:
        components.html(html_data, height=440)

# --- Layout Configuration ---
st.title("🎱 TensorFlow DQN Snooker Engine")
st.markdown("Thread-isolated neural optimization processing inside Streamlit Cloud Containers.")

tab1, tab2 = st.tabs(["🎮 Active Play Deck", "🏋️ Train TensorFlow Weights"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("Match Tracker")
        st.write(f"**Turn:** `{st.session_state.turn}`")
        st.metric("Player Score", st.session_state.score["Player"])
        st.metric("AI Score", st.session_state.score["AI"])
        st.caption(f"Network Exploration Epsilon: {max(0.1, 1.0 - (st.session_state.training_episodes * 0.05)):.2f}")
        
    with col1:
        render_placeholder = st.container()
        render_canvas(render_placeholder)
        
        if st.session_state.turn == "Player":
            angle = st.slider("Striking Direction (Degrees)", 0.0, 360.0, 180.0, 1.0)
            power = st.slider("Force Magnitude", 1.0, 20.0, 12.0, 0.5)
            if st.button("💥 Strike Cue Ball", use_container_width=True):
                cue = next((b for b in st.session_state.balls if b["id"] == "cue"), None)
                if cue:
                    rad = math.radians(angle)
                    cue["vx"] = math.cos(rad) * power
                    cue["vy"] = math.sin(rad) * power
                    run_physics_loop(render_placeholder)
                    st.session_state.turn = "AI"
                    st.rerun()
        else:
            if st.button("🤖 Process AI Brain Calculation", type="primary", use_container_width=True):
                execute_tf_ai_turn(visualize=True)
                st.session_state.turn = "Player"
                st.rerun()

with tab2:
    st.subheader("Train TensorFlow Vector Profiles")
    episodes_to_run = st.number_input("Training Cycles", min_value=1, max_value=50, value=5)
    
    if st.button("🚀 Run Fast Training Runs"):
        progress_bar = st.progress(0.0)
        for ep in range(int(episodes_to_run)):
            st.session_state.training_episodes += 1
            st.session_state.balls = [{"id": "cue", "x": 200.0, "y": 200.0, "vx": 0.0, "vy": 0.0, "color": COLOR_MAP["cue"]}]
            for b_type, x, y in [("red", 550.0, 200.0), ("yellow", 520.0, 150.0), ("black", 650.0, 200.0)]:
                st.session_state.balls.append({"id": b_type, "x": x, "y": y, "vx": 0.0, "vy": 0.0, "color": COLOR_MAP[b_type]})
            
            for shot in range(5):
                execute_tf_ai_turn(visualize=False)
            progress_bar.progress((ep + 1) / episodes_to_run)
            
        st.success(f"TensorFlow network optimized! Current Training Benchmark: {st.session_state.training_episodes}")

if st.button("🔄 Full Engine System Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
