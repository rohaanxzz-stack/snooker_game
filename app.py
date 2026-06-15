import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import math

# Page Configuration
st.set_page_config(page_title="TF DQN Snooker Engine", layout="wide")

# Game Constants
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

# --- State Initialization ---
if "balls" not in st.session_state:
    st.session_state.balls = []
    st.session_state.score = {"Player": 0, "AI": 0}
    st.session_state.turn = "Player"
    st.session_state.game_over = False
    st.session_state.status_msg = "Your turn! Choose an angle and power to strike."
    st.session_state.training_episodes = 0
    
    # Initialize Core Ball Setup
    st.session_state.balls.append({"id": "cue", "x": 200.0, "y": 200.0, "vx": 0.0, "vy": 0.0, "color": COLOR_MAP["cue"]})
    object_balls = [
        ("red", 550.0, 200.0), ("red", 570.0, 190.0), ("red", 570.0, 210.0),
        ("yellow", 520.0, 150.0), ("green", 520.0, 250.0), ("black", 650.0, 200.0)
    ]
    for b_type, x, y in object_balls:
        st.session_state.balls.append({"id": b_type, "x": x, "y": y, "vx": 0.0, "vy": 0.0, "color": COLOR_MAP[b_type]})

# --- TensorFlow From-Scratch DQN Brain ---
# 8 Inputs: Cue X, Cue Y, Target closest X, Target closest Y, Closet pocket X, Pocket Y, Enemy Score, Your Score
# 16 Outputs: Categorized actions representing 16 distinct shooting angles around the circle
def build_dqn_model():
    model = models.Sequential([
        layers.Input(shape=(8,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(16, activation='linear') # 16 discrete firing vectors
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')
    return model

if "dqn_brain" not in st.session_state:
    st.session_state.dqn_brain = build_dqn_model()

# --- Physics Processing Pipeline ---
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
    
    for _ in range(150): # Limiting execution cycles per turn to prevent hangs
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
        
        # Track Pocketings
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

# --- Feature Extraction & TensorFlow Step ---
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
    
    return np.array([[ 
        cue["x"] / TABLE_WIDTH, cue["y"] / TABLE_HEIGHT,
        ox / TABLE_WIDTH, oy / TABLE_HEIGHT,
        closest_pocket[0] / TABLE_WIDTH, closest_pocket[1] / TABLE_HEIGHT,
        st.session_state.score["AI"] / 10.0, st.session_state.score["Player"] / 10.0
    ]], dtype=np.float32)

def execute_tf_ai_turn(visualize=True):
    balls = st.session_state.balls
    cue = next((b for b in balls if b["id"] == "cue"), None)
    if not cue: return 0, False
    
    state = get_current_state_vector()
    
    # Epsilon-greedy exploration vs exploitation
    epsilon = max(0.1, 1.0 - (st.session_state.training_episodes * 0.05))
    if np.random.rand() < epsilon:
        action_idx = np.random.randint(0, 16)
    else:
        q_values = st.session_state.dqn_brain.predict(state, verbose=0)
        action_idx = np.argmax(q_values[0])
        
    # Translate action integer into an operational vector angle
    target_angle = (action_idx / 16.0) * 2 * math.pi
    power = 15.0
    cue["vx"] = math.cos(target_angle) * power
    cue["vy"] = math.sin(target_angle) * power
    
    # Run dynamic system simulation
    scored, scratched = run_physics_loop(render_placeholder if visualize else None)
    
    # Reward engineering matrix
    reward = (scored * 5.0) - (0.5 if scratched else 0.0) + (0.1 if scored > 0 else -0.1)
    
    # Custom Gradient descent update via backpropagation
    next_state = get_current_state_vector()
    target_q = reward + 0.95 * np.max(st.session_state.dqn_brain.predict(next_state, verbose=0)[0])
    
    updated_q_profile = st.session_state.dqn_brain.predict(state, verbose=0)
    updated_q_profile[0][action_idx] = target_q
    
    # Train the model on the single step experience
    st.session_state.dqn_brain.fit(state, updated_q_profile, epochs=1, verbose=0)
    return scored, scratched

# --- Dynamic Canvas Rendering Markup ---
def render_canvas(element_handle):
    balls_js = ", ".join([f'{{x: {b["x"]}, y: {b["y"]}, c: "{b["color"]}"}}' for b in st.session_state.balls])
    pockets_js = ", ".join([f'{{x: {p[0]}, y: {p[1]}}}' for p in POCKETS])
    
    html_data = f"""
    <div style="text-align: center;">
        <canvas id="canvas" width="{TABLE_WIDTH}" height="{TABLE_HEIGHT}" style="background-color:#1e5631; border:10px solid #4a2c11; border-radius:5px;"></canvas>
    </div>
    <script>
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        var balls = [{balls_js}];
        var pockets = [{pockets_js}];
        
        ctx.clearRect(0,0, {TABLE_WIDTH}, {TABLE_HEIGHT});
        
        // Render Pockets
        pockets.forEach(p => {{
            ctx.beginPath();
            ctx.arc(p.x, p.y, {POCKET_RADIUS}, 0, 2*Math.PI);
            ctx.fillStyle = '#111';
            ctx.fill();
        }});
        
        // Render Balls
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
    element_handle.html(html_data, height=440)

# --- App UI Layout Framework ---
st.title("🧠 Self-Training TensorFlow DQN Snooker App")
st.markdown("This instance runs an independent deep reinforcement learning loop inside Python 3.11 without cloud APIs.")

# Main Interactive UI controls
tab1, tab2 = st.tabs(["🎮 Human vs AI Play Mode", "🏋️ Train TensorFlow Brain Live"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("Match Tracker")
        st.write(f"**Turn:** `{st.session_state.turn}`")
        st.metric("Player Score", st.session_state.score["Player"])
        st.metric("AI Score", st.session_state.score["AI"])
        st.caption(f"Exploration Rate (Epsilon): {max(0.1, 1.0 - (st.session_state.training_episodes * 0.05)):.2f}")
        
    with col1:
        render_placeholder = st.empty()
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
            if st.button("🤖 Run Neural Network AI Move", type="primary", use_container_width=True):
                st.session_state.status_msg = "TensorFlow checking state-space tensors..."
                execute_tf_ai_turn(visualize=True)
                st.session_state.turn = "Player"
                st.rerun()

with tab2:
    st.subheader("Train the Reinforcement Learning Agent")
    st.markdown("Let the TensorFlow agent play rapid self-play sessions to learn pocket alignment vectors.")
    episodes_to_run = st.number_input("Training Generations", min_value=1, max_value=50, value=5)
    
    if st.button("🚀 Start Self-Training Iteration Loop"):
        progress_bar = st.progress(0.0)
        status_box = st.empty()
        
        for ep in range(int(episodes_to_run)):
            st.session_state.training_episodes += 1
            # Reset game variables for training iteration
            st.session_state.balls = [{"id": "cue", "x": 200.0, "y": 200.0, "vx": 0.0, "vy": 0.0, "color": COLOR_MAP["cue"]}]
            for b_type, x, y in [("red", 550.0, 200.0), ("yellow", 520.0, 150.0), ("black", 650.0, 200.0)]:
                st.session_state.balls.append({"id": b_type, "x": x, "y": y, "vx": 0.0, "vy": 0.0, "color": COLOR_MAP[b_type]})
            
            # Execute 5 rapid sequential shots per training generation
            for shot in range(5):
                execute_tf_ai_turn(visualize=False)
                
            progress_bar.progress((ep + 1) / episodes_to_run)
            status_box.text(f"Completed Generation Match Series {ep+1}/{episodes_to_run} | Total Neural Training Count: {st.session_state.training_episodes}")
            
        st.success("TensorFlow weights optimized based on results! Return to 'Play Mode' to test your trained agent.")

if st.button("🔄 Full Engine System Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
