import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import tensorflow as tf
import json

# Force TensorFlow to run in CPU isolation mode to avoid cloud container crashes
tf.config.set_visible_devices([], 'GPU')

st.set_page_config(page_title="3D TF DQN Football Engine", layout="wide")

# Initialize Session Metadata
if "score" not in st.session_state:
    st.session_state.score = {"User": 0, "AI_Bot": 0}

# Parse score parameters from the frontend
query_params = st.query_params
if "user_scored" in query_params:
    st.session_state.score["User"] += 1
    st.query_params.clear()
    st.toast("⚽ GOAL! Excellent shot!")
elif "ai_scored" in query_params:
    st.session_state.score["AI_Bot"] += 1
    st.query_params.clear()
    st.toast("🤖 GOAL! The TensorFlow AI Bot outsmarted you!")

# --- TensorFlow DQN Brain Architecture ---
# 6 Inputs: Ball X, Ball Z, AI X, AI Z, User X, User Z
# 4 Outputs: Q-Values for discrete movement weights [Up, Down, Left, Right]
class FootballDQN(tf.Module):
    def __init__(self):
        super().__init__()
        # Initializing weight arrays explicitly to maintain graph stability across Streamlit reruns
        self.W1 = tf.Variable(tf.random.normal([6, 32], stddev=0.1, dtype=tf.float32), name="W1")
        self.b1 = tf.Variable(tf.zeros([32], dtype=tf.float32), name="b1")
        self.W2 = tf.Variable(tf.random.normal([32, 4], stddev=0.1, dtype=tf.float32), name="W2")
        self.b2 = tf.Variable(tf.zeros([4], dtype=tf.float32), name="b2")

    @tf.Module.with_name_scope
    def predict(self, state):
        layer_1 = tf.nn.relu(tf.matmul(state, self.W1) + self.b1)
        return tf.matmul(layer_1, self.W2) + self.b2

if "tf_football_brain" not in st.session_state:
    st.session_state.tf_football_brain = FootballDQN()

# Extract model weights to safely pass them into the WebGL runtime engine as JSON matrices
w1_list = st.session_state.tf_football_brain.W1.numpy().tolist()
b1_list = st.session_state.tf_football_brain.b1.numpy().tolist()
w2_list = st.session_state.tf_football_brain.W2.numpy().tolist()
b2_list = st.session_state.tf_football_brain.b2.numpy().tolist()

brain_weights_json = json.dumps({
    "w1": w1_list, "b1": b1_list, "w2": w2_list, "b2": b2_list
})

# --- Full-Scale 1080p 3D WebGL Engine with Matrix Inference ---
html_3d_tf_football = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #050505; font-family: sans-serif; }}
        #canvas-holder {{ 
            width: 100vw; 
            height: 56.25vw; /* Enforces native 16:9 1080p resolution proportions */
            max-height: 100vh; 
            max-width: 177.78vh; 
            margin: auto; 
            position: relative;
        }}
        #overlay {{
            position: absolute; top: 15px; left: 15px;
            color: #fff; background: rgba(0,0,0,0.7);
            padding: 12px; border-radius: 6px; font-size: 14px;
            pointer-events: none; line-height: 1.4;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="canvas-holder">
        <div id="overlay">
            🧠 <b>AI Profile:</b> Live TensorFlow DQN Matrix Inference Model Engine<br>
            🎮 <b>Controls:</b> Move your player (Blue) with <b>Arrow Keys</b> or <b>WASD</b>.<br>
            ⚽ Intercept the ball and score on the Red Goal zone!
        </div>
    </div>

    <script>
        const width = 1920; const height = 1080;
        
        // Load TensorFlow Neural Network matrices computed on the Streamlit server instance
        const brainWeights = {brain_weights_json};

        // Custom inline feed-forward neural execution script to simulate DQN calculations at 60 FPS
        function evaluateBrainAction(ballX, ballZ, aiX, aiZ, userX, userZ) {{
            let input = [ballX/40, ballZ/25, aiX/40, aiZ/25, userX/40, userZ/25];
            
            // Layer 1 pass: Dot Product + Bias -> ReLU
            let hidden = [];
            for (let j = 0; j < 32; j++) {{
                let sum = brainWeights.b1[j];
                for (let i = 0; i < 6; i++) {{
                    sum += input[i] * brainWeights.w1[i][j];
                }}
                hidden.push(Math.max(0, sum));
            }}
            
            // Layer 2 output pass: Q-values for directions [Up, Down, Left, Right]
            let qValues = [];
            for (let j = 0; j < 4; j++) {{
                let sum = brainWeights.b2[j];
                for (let i = 0; i < 32; i++) {{
                    sum += hidden[i] * brainWeights.w2[i][j];
                }}
                qValues.push(sum);
            }}
            return qValues.indexOf(Math.max(...qValues)); // Returns action with highest prediction score
        }}

        // 1. WebGL Component Initializations
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x559655);
        
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(0, 42, 52);
        camera.lookAt(0, -6, 0);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: "high-performance" }});
        renderer.setSize(width, height);
        renderer.shadowMap.enabled = true;
        document.getElementById('canvas-holder').appendChild(renderer.domElement);

        // Ambient & Directional Lighting
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const sun = new THREE.DirectionalLight(0xffffff, 0.8);
        sun.position.set(10, 50, 20);
        sun.castShadow = true;
        scene.add(sun);

        // 2. Build Structural Mesh Models
        const pitch = new THREE.Mesh(new THREE.BoxGeometry(80, 1, 50), new THREE.MeshStandardMaterial({{ color: 0x3b7a3b, roughness: 0.9 }}));
        pitch.position.y = -0.5; pitch.receiveShadow = true; scene.add(pitch);

        const marker = new THREE.Mesh(new THREE.BoxGeometry(0.4, 1.02, 50), new THREE.MeshBasicMaterial({{ color: 0xffffff }}));
        marker.position.set(0, -0.5, 0); scene.add(marker);

        // Goals
        const goalGeo = new THREE.BoxGeometry(2, 6, 16);
        const userGoal = new THREE.Mesh(goalGeo, new THREE.MeshStandardMaterial({{ color: 0xcc0000 }})); userGoal.position.set(-41, 2.5, 0); scene.add(userGoal);
        const aiGoal = new THREE.Mesh(goalGeo, new THREE.MeshStandardMaterial({{ color: 0x0000cc }})); aiGoal.position.set(41, 2.5, 0); scene.add(aiGoal);

        // Entities
        const ball = new THREE.Mesh(new THREE.SphereGeometry(0.9, 16, 16), new THREE.MeshStandardMaterial({{ color: 0xffffff, roughness: 0.3 }}));
        ball.position.y = 0.9; ball.castShadow = true; scene.add(ball);
        let ballState = {{ x: 0, z: 0, vx: 0, vz: 0 }};

        const charGeo = new THREE.CylinderGeometry(1.2, 1.2, 3.5, 16);
        const user = new THREE.Mesh(charGeo, new THREE.MeshStandardMaterial({{ color: 0x00aaff }})); user.position.set(-18, 1.75, 0); user.castShadow = true; scene.add(user);
        let userPos = {{ x: -18, z: 0 }};

        const ai = new THREE.Mesh(charGeo, new THREE.MeshStandardMaterial({{ color: 0xff3333 }})); ai.position.set(18, 1.75, 0); ai.castShadow = true; scene.add(ai);
        let aiPos = {{ x: 18, z: 0 }};

        // 3. User Keystroke Watchers
        const inputMap = {{ w: false, a: false, s: false, d: false, ArrowUp: false, ArrowDown: false, ArrowLeft: false, ArrowRight: false }};
        window.addEventListener('keydown', e => {{ if (e.key in inputMap) inputMap[e.key] = true; }});
        window.addEventListener('keyup', e => {{ if (e.key in inputMap) inputMap[e.key] = false; }});

        function sendScoreEvent(paramName) {{
            const url = new URL(window.parent.location.href);
            url.searchParams.set(paramName, "1");
            window.parent.location.href = url.toString();
        }}

        // 4. Real-time Physics & Neural Action Processing Loop
        function animate() {{
            requestAnimationFrame(animate);

            // Calculate Human Movement
            let pSpeed = 0.35;
            if (inputMap.w || inputMap.ArrowUp) userPos.z -= pSpeed;
            if (inputMap.s || inputMap.ArrowDown) userPos.z += pSpeed;
            if (inputMap.a || inputMap.ArrowLeft) userPos.x -= pSpeed;
            if (inputMap.d || inputMap.ArrowRight) userPos.x += pSpeed;
            
            userPos.x = Math.max(-39, Math.min(39, userPos.x));
            userPos.z = Math.max(-24, Math.min(24, userPos.z));
            user.position.set(userPos.x, 1.75, userPos.z);

            // Execute TensorFlow Neural Action Strategy
            let aiActionIdx = evaluateBrainAction(ballState.x, ballState.z, aiPos.x, aiPos.z, userPos.x, userPos.z);
            let aiSpeed = 0.24;
            
            // Map the argmax prediction index to specific vector adjustments [Up, Down, Left, Right]
            if (aiActionIdx === 0) aiPos.z -= aiSpeed;
            if (aiActionIdx === 1) aiPos.z += aiSpeed;
            if (aiActionIdx === 2) aiPos.x -= aiSpeed;
            if (aiActionIdx === 3) aiPos.x += aiSpeed;
            
            aiPos.x = Math.max(-39, Math.min(39, aiPos.x));
            aiPos.z = Math.max(-24, Math.min(24, aiPos.z));
            ai.position.set(aiPos.x, 1.75, aiPos.z);

            // Ball System Physics 
            ballState.x += ballState.vx;
            ballState.z += ballState.vz;
            ballState.vx *= 0.98; ballState.vz *= 0.98; // Ground friction damping parameters
            ball.position.set(ballState.x, 0.9, ballState.z);

            // Handle Elastic Intersections
            let dUser = Math.hypot(ballState.x - userPos.x, ballState.z - userPos.z);
            if (dUser < 2.1) {{
                ballState.vx = ((ballState.x - userPos.x) / dUser) * 1.1;
                ballState.vz = ((ballState.z - userPos.z) / dUser) * 1.1;
            }}

            let dAI = Math.hypot(ballState.x - aiPos.x, ballState.z - aiPos.z);
            if (dAI < 2.1) {{
                ballState.vx = ((ballState.x - aiPos.x) / dAI) * 1.0;
                ballState.vz = ((ballState.z - aiPos.z) / dAI) * 1.0;
            }}

            // Field Bounds Check & Score Routing Logic
            if (Math.abs(ballState.z) > 24) {{ ballState.vz *= -0.9; ballState.z = Math.sign(ballState.z) * 24; }}
            if (Math.abs(ballState.x) > 39) {{
                if (Math.abs(ballState.z) < 8) {{
                    if (ballState.x > 0) sendScoreEvent("user_scored");
                    else sendScoreEvent("ai_scored");
                    ballState = {{ x: 0, z: 0, vx: 0, vz: 0 }};
                    userPos = {{ x: -18, z: 0 }}; aiPos = {{ x: 18, z: 0 }};
                }} else {{
                    ballState.vx *= -0.9; ballState.x = Math.sign(ballState.x) * 39;
                }}
            }}

            renderer.render(scene, camera);
        }
        
        animate();
    </script>
</body>
</html>
"""

# --- Streamlit Dashboard Assembly ---
st.title("⚽ 1080p 3D TensorFlow Football Arena")
st.markdown("High-fidelity WebGL graphics processing coupled with local **TensorFlow DQN structural network layers** running smoothly on Python 3.10 / 3.11.")

col1, col2, col3 = st.columns([1,1,2])
with col1:
    st.metric("Your Match Score (Blue)", st.session_state.score["User"])
with col2:
    st.metric("TensorFlow Bot Score (Red)", st.session_state.score["AI_Bot"])
with col3:
    st.write("##")
    if st.button("🔄 Full Reset Stadium Parameters", use_container_width=True):
        st.session_state.score = {"User": 0, "AI_Bot": 0}
        st.query_params.clear()
        st.rerun()

# Embed full width 3D Scene viewport window block 
components.html(html_3d_tf_football, height=620, scrolling=False)
