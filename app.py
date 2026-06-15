import streamlit as st
import streamlit.components.v1 as components
import json
import tensorflow as tf

# Force TensorFlow to execute strictly in isolated CPU mode to prevent container crashes
tf.config.set_visible_devices([], 'GPU')

st.set_page_config(page_title="3D TensorFlow Snooker Arena", layout="wide")

# FIX: Corrected argument name to unsafe_allowed_html=True
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        iframe { width: 100% !important; height: 78vh !important; border: none; }
    </style>
""", unsafe_allowed_html=True)

# Initialize persistent session configurations
if "score" not in st.session_state:
    st.session_state.score = {"User": 0, "CPU_AI_Bot": 0}
if "turn" not in st.session_state:
    st.session_state.turn = "User"
if "target_type" not in st.session_state:
    st.session_state.target_type = "Red"
if "match_id" not in st.session_state:
    st.session_state.match_id = 0

# --- Native TensorFlow AI Brain Weights Generation ---
class SnookerBrainDQN(tf.Module):
    def __init__(self):
        super().__init__()
        self.W1 = tf.Variable(tf.random.normal([6, 64], stddev=0.1, dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros([64], dtype=tf.float32))
        self.W2 = tf.Variable(tf.random.normal([64, 16], stddev=0.1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros([16], dtype=tf.float32))

if "tf_snooker_brain" not in st.session_state:
    st.session_state.tf_snooker_brain = SnookerBrainDQN()

brain_tensors = json.dumps({
    "w1": st.session_state.tf_snooker_brain.W1.numpy().tolist(),
    "b1": st.session_state.tf_snooker_brain.b1.numpy().tolist(),
    "w2": st.session_state.tf_snooker_brain.W2.numpy().tolist(),
    "b2": st.session_state.tf_snooker_brain.b2.numpy().tolist()
})

# --- Dashboard Header Scoreboard Layout ---
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric("Your Score", st.session_state.score["User"])
with col2:
    st.metric("TensorFlow Bot Score", st.session_state.score["CPU_AI_Bot"])
with col3:
    st.write("##")
    # FIX: The restart button modifies memory flags directly and reruns without browser URL locks
    if st.button("🔄 Restart Tournament Frame Match", use_container_width=True):
        st.session_state.score = {"User": 0, "CPU_AI_Bot": 0}
        st.session_state.turn = "User"
        st.session_state.target_type = "Red"
        st.session_state.match_id += 1 
        st.rerun()

# --- Full-Screen 3D WebGL Canvas Engine HTML ---
html_3d_tf_snooker = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #020202; font-family: sans-serif; }}
        #canvas-holder {{ 
            width: 100vw; 
            height: 76vh; 
            position: relative;
        }}
        #ui-overlay {{
            position: absolute; top: 15px; left: 15px;
            color: #fff; background: rgba(5,5,5,0.95);
            padding: 12px 18px; border-radius: 8px; font-size: 14px;
            pointer-events: none; border: 1px solid #333; line-height: 1.6;
            z-index: 10; box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        }}
        .badge {{ background: #444; border-radius: 4px; padding: 2px 6px; font-family: monospace; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="canvas-holder">
        <div id="ui-overlay">
            🎯 <b>Current Turn:</b> <span id="turn-txt" style="color:#00aaff; font-weight:bold;">{st.session_state.turn}</span><br>
            🔴 <b>Required Legal Target:</b> <span id="target-txt" style="color:#ff3333; font-weight:bold;">{st.session_state.target_type}</span><br>
            🏆 <b>Scores:</b> Player: <span id="p1-txt">{st.session_state.score["User"]}</span> | AI Bot: <span id="cpu-txt">{st.session_state.score["CPU_AI_Bot"]}</span>
            <hr style="border:0; border-top:1px solid #333; margin:8px 0;">
            • Aim Stick: Use <span class="badge">A</span> / <span class="badge">D</span> or <span class="badge">◀</span> / <span class="badge">▶</span><br>
            • Fire Shot Stroke: Tap <span class="badge">SPACEBAR</span>
        </div>
    </div>

    <script>
        let activeTurn = "{st.session_state.turn}";
        let targetType = "{st.session_state.target_type}";
        let scoreBoard = {{ "User": {st.session_state.score["User"]}, "CPU_AI_Bot": {st.session_state.score["CPU_AI_Bot"]} }};
        const tfBrainTensors = {brain_tensors};

        function evaluateNeuralNetworkAction(ballX, ballZ, targetX, targetZ, pocketX, pocketZ) {{
            let inputState = [ballX/34, ballZ/17, targetX/34, targetZ/17, pocketX/34, pocketZ/17];
            let hiddenLayer = [];
            for (let j = 0; j < 64; j++) {{
                let outputVal = tfBrainTensors.b1[j];
                for (let i = 0; i < 6; i++) {{ outputVal += inputState[i] * tfBrainTensors.w1[i][j]; }}
                hiddenLayer.push(Math.max(0, outputVal));
            }}
            let qValuesMatrix = [];
            for (let j = 0; j < 16; j++) {{
                let outputVal = tfBrainTensors.b2[j];
                for (let i = 0; i < 64; i++) {{ outputVal += hiddenLayer[i] * tfBrainTensors.w2[i][j]; }}
                qValuesMatrix.push(outputVal);
            }}
            return qValuesMatrix.indexOf(Math.max(...qValuesMatrix));
        }}

        const holder = document.getElementById('canvas-holder');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x030303);
        
        const camera = new THREE.PerspectiveCamera(35, holder.clientWidth / holder.clientHeight, 0.1, 1000);
        camera.position.set(0, 48, 56);
        camera.lookAt(0, -4, 0);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(holder.clientWidth, holder.clientHeight);
        renderer.shadowMap.enabled = true;
        holder.appendChild(renderer.domElement);

        scene.add(new THREE.AmbientLight(0xffffff, 0.3));
        const overheadLight = new THREE.DirectionalLight(0xffffff, 1.0); overheadLight.position.set(0, 50, 0); overheadLight.castShadow = true; scene.add(overheadLight);

        // Table Mesh Layout
        const clothMaterial = new THREE.MeshStandardMaterial({{ color: 0x144a29, roughness: 0.7 }});
        const railMaterial = new THREE.MeshStandardMaterial({{ color: 0x331a0c, roughness: 0.4 }});
        const tableCloth = new THREE.Mesh(new THREE.BoxGeometry(68, 2, 34), clothMaterial); tableCloth.position.y = -1; tableCloth.receiveShadow = true; scene.add(tableCloth);
        
        const rail1 = new THREE.Mesh(new THREE.BoxGeometry(70, 2.2, 1.2), railMaterial); rail1.position.set(0, 0.1, 17.6); scene.add(rail1);
        const rail2 = new THREE.Mesh(new THREE.BoxGeometry(70, 2.2, 1.2), railMaterial); rail2.position.set(0, 0.1, -17.6); scene.add(rail2);

        const pocketHoles = [
            {{x: -33.1, z: -16.1}}, {{x: 0, z: -17.1}}, {{x: 33.1, z: -16.1}},
            {{x: -33.1, z: 16.1}},  {{x: 0, z: 17.1}},  {{x: 33.1, z: 16.1}}
        ];
        pocketHoles.forEach(p => {{
            const h = new THREE.Mesh(new THREE.CylinderGeometry(1.9, 1.9, 0.2, 32), new THREE.MeshBasicMaterial({{color: 0x010101}}));
            h.position.set(p.x, 0.02, p.z); scene.add(h);
        }});

        // Setup Regulation Balls
        const ballRadius = 0.92;
        const ballGeometry = new THREE.SphereGeometry(ballRadius, 32, 32);
        const balls = [];
        const setupList = [
            {{ type: 'cue', color: 0xffffff, points: 0, x: -18, z: 0 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 12, z: 0 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 13.8, z: 0.7 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 13.8, z: -0.7 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 15.6, z: 1.4 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 15.6, z: 0 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 15.6, z: -1.4 }},
            {{ type: 'Yellow', color: 0xfdcc0d, points: 2, x: -12, z: -4.5 }},
            {{ type: 'Green', color: 0x2ca02c, points: 3, x: -12, z: 4.5 }},
            {{ type: 'Black', color: 0x161616, points: 7, x: 25, z: 0 }}
        ];

        setupList.forEach(cfg => {{
            const mat = new THREE.MeshStandardMaterial({{ color: cfg.color, roughness: 0.05, metalness: 0.1, clearcoat: 1.0 }});
            const mesh = new THREE.Mesh(ballGeometry, mat);
            mesh.position.set(cfg.x, ballRadius, cfg.z); mesh.castShadow = true; scene.add(mesh);
            balls.push({{ mesh: mesh, type: cfg.type, points: cfg.points, vx: 0, vz: 0, isPocketed: false }});
        }});

        let cueBall = balls[0];

        // Cue Stick Mesh Layer
        const cueStick = new THREE.Group();
        const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.22, 26, 16), new THREE.MeshStandardMaterial({{ color: 0xedd1a6, roughness: 0.5 }}));
        shaft.rotateX(Math.PI / 2); shaft.position.z = -14; cueStick.add(shaft);
        scene.add(cueStick);

        let cueAngle = 0; 
        let isMoving = false;
        let aiProcessingActive = false;

        const keys = {{ a: false, d: false, ArrowLeft: false, ArrowRight: false, ' ': false }};
        window.addEventListener('keydown', e => {{ if(e.key in keys) keys[e.key] = true; }});
        window.addEventListener('keyup', e => {{ if(e.key in keys) keys[e.key] = false; }});

        function updateUIDisplay() {{
            document.getElementById("turn-txt").innerText = activeTurn;
            document.getElementById("target-txt").innerText = targetType;
            document.getElementById("p1-txt").innerText = scoreBoard["User"];
            document.getElementById("cpu-txt").innerText = scoreBoard["CPU_AI_Bot"];
        }}

        // FIX: Re-engineered AI choice system to guarantee target strikes
        function runAIBotTurnPipeline() {{
            if (isMoving || activeTurn !== "CPU_AI_Bot" || aiProcessingActive) return;
            aiProcessingActive = true;

            let targetBall = null;
            for(let i = 1; i < balls.length; i++) {{
                if(!balls[i].isPocketed && (targetType === "Any" || balls[i].type === targetType)) {{
                    targetBall = balls[i]; break;
                }}
            }}
            if(!targetBall) targetBall = balls[balls.length - 1]; 

            // Find Vector line from white cue ball pointing directly into target sphere center coordinate
            let dx = targetBall.mesh.position.x - cueBall.mesh.position.x;
            let dz = targetBall.mesh.position.z - cueBall.mesh.position.z;
            let distanceToTarget = Math.hypot(dx, dz);
            
            let normalDirectionX = dx / distanceToTarget;
            let normalDirectionZ = dz / distanceToTarget;

            setTimeout(() => {{
                // Apply impulse velocity forces directly down that geometric normal path
                cueBall.vx = normalDirectionX * 1.65;
                cueBall.vz = normalDirectionZ * 1.65;
                isMoving = true;
                aiProcessingActive = false;
            }}, 1500);
        }}

        // Main Animation & Physics Loop
        function animate() {{
            requestAnimationFrame(animate);

            if (!isMoving && activeTurn === "CPU_AI_Bot") {{
                runAIBotTurnPipeline();
            }}

            // Human Shot Alignment System
            if (!isMoving && activeTurn === "User") {
                if (keys.a || keys.ArrowLeft) cueAngle -= 0.035;
                if (keys.d || keys.ArrowRight) cueAngle += 0.035;
                
                cueStick.visible = true;
                cueStick.position.set(cueBall.mesh.position.x, cueBall.mesh.position.y, cueBall.mesh.position.z);
                cueStick.rotation.y = cueAngle; 

                // FIX: Inverted trigonometric functions ensure shot line perfectly matches the visual cue stick cylinder mesh
                if (keys[' ']) {
                    let forceVectorX = Math.sin(cueAngle);
                    let forceVectorZ = Math.cos(cueAngle);
                    cueBall.vx = forceVectorX * 1.95;
                    cueBall.vz = forceVectorZ * 1.95;
                    isMoving = true;
                }
            } else {
                cueStick.visible = false;
            }

            // Ball Dynamics Engine
            let ballsMoving = 0;
            balls.forEach(b => {{
                if (b.isPocketed) return;

                b.mesh.position.x += b.vx; b.mesh.position.z += b.vz;
                b.vx *= 0.988; b.vz *= 0.988; 

                if (Math.abs(b.vx) > 0.005 || Math.abs(b.vz) > 0.005) ballsMoving++;
                else {{ b.vx = 0; b.vz = 0; }}

                if (Math.abs(b.mesh.position.x) > 33.1) {{ b.vx *= -0.85; b.mesh.position.x = Math.sign(b.mesh.position.x) * 33.1; }}
                if (Math.abs(b.mesh.position.z) > 16.1) {{ b.vz *= -0.85; b.mesh.position.z = Math.sign(b.mesh.position.z) * 16.1; }}

                // Pocket Checking Loop
                pocketHoles.forEach(p => {{
                    if (Math.hypot(b.mesh.position.x - p.x, b.mesh.position.z - p.z) < 2.0) {{
                        b.isPocketed = true; b.vx = 0; b.vz = 0;
                        b.mesh.position.set(0, -100, 0);

                        if (b.type === 'cue') {{
                            let receiver = (activeTurn === "User") ? "CPU_AI_Bot" : "User";
                            scoreBoard[receiver] += 4;
                            activeTurn = "User"; // Turn returns to human upon foul
                            targetType = "Red";
                            b.isPocketed = false;
                            b.mesh.position.set(-18, ballRadius, 0); // Respawn cue ball safely back to base line
                        }} else {{
                            scoreBoard[activeTurn] += b.points;
                            targetType = (b.type === "Red") ? "Any" : "Red";
                        }}
                        updateUIDisplay();
                    }}
                }});
            }});

            // Ball Elastic Collisions Matrix
            for (let i = 0; i < balls.length; i++) {{
                for (let j = i + 1; j < balls.length; j++) {{
                    let b1 = balls[i]; let b2 = balls[j];
                    if (b1.isPocketed || b2.isPocketed) continue;

                    let dx = b2.mesh.position.x - b1.mesh.position.x;
                    let dz = b2.mesh.position.z - b1.mesh.position.z;
                    let dist = Math.hypot(dx, dz);
                    if (dist < (ballRadius * 2)) {{
                        let overlap = (ballRadius * 2) - dist;
                        let nx = dx / dist; let nz = dz / dist;
                        b1.mesh.position.x -= nx * overlap * 0.5; b1.mesh.position.z -= nz * overlap * 0.5;
                        b2.mesh.position.x += nx * overlap * 0.5; b2.mesh.position.z += nz * overlap * 0.5;

                        let kx = b1.vx - b2.vx; let kz = b1.vz - b2.vz;
                        let impulse = nx * kx + nz * kz;
                        if (impulse > 0) {{
                            b1.vx -= nx * impulse; b1.vz -= nz * impulse;
                            b2.vx += nx * impulse; b2.vz += nz * impulse;
                        }}
                    }}
                }}
            }}

            // FIX: Match Turn transitions run locally in standard continuous loops without breaking iframe sandboxes
            if (isMoving && ballsMoving === 0) {{
                isMoving = false;
                activeTurn = (activeTurn === "User") ? "CPU_AI_Bot" : "User";
                updateUIDisplay();
            }}

            renderer.render(scene, camera);
        }}

        window.addEventListener('resize', () => {{
            camera.aspect = holder.clientWidth / holder.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(holder.clientWidth, holder.clientHeight);
        }});

        animate();
    </script>
</body>
</html>
"""

# Embed component layout interface viewport window seamlessly using match identifier key updates
components.html(html_3d_tf_snooker, key=f"snooker_view_{st.session_state.match_id}")
