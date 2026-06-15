import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import tensorflow as tf
import json

# Force TensorFlow to execute strictly in isolated CPU mode to prevent multi-threading container crashes on Streamlit Cloud
tf.config.set_visible_devices([], 'GPU')

# Set full layout viewport
st.set_page_config(page_title="Production 3D TensorFlow Snooker", layout="wide")

# Ensure core game mechanics state memory maps across server refreshes
if "score" not in st.session_state:
    st.session_state.score = {"User": 0, "CPU_AI_Bot": 0}
if "turn" not in st.session_state:
    st.session_state.turn = "User"        # User executes the first break shot
if "target_type" not in st.session_state:
    st.session_state.target_type = "Red"  # Rules dictate opening with a Red ball

# Synchronize backend structural state via standard query parameters safely
try:
    query_params = st.query_params
    if "scored_points" in query_params:
        pts = int(query_params["scored_points"])
        current_turn = query_params.get("active_turn", "User")
        st.session_state.score[current_turn] += pts
        st.session_state.target_type = query_params.get("next_target", "Red")
        st.session_state.turn = query_params.get("switch_turn", current_turn)
        st.query_params.clear()
        st.toast(f"🎱 {current_turn} scored {pts} point(s)! Next required target: {st.session_state.target_type}")
    elif "foul_occured" in query_params:
        foul_by = query_params["foul_occured"]
        penalty_receiver = "CPU_AI_Bot" if foul_by == "User" else "User"
        st.session_state.score[penalty_receiver] += 4  # Standard minimum regulation foul penalty points
        st.session_state.turn = "User"                 # Hand match advantage back to human user
        st.session_state.target_type = "Red"
        st.query_params.clear()
        st.toast(f"⚠️ SCRATCH FOUL by {foul_by}! 4 Points awarded to {penalty_receiver}.")
except Exception:
    pass

# --- Native TensorFlow Brain Architecture ---
class SnookerBrainDQN(tf.Module):
    def __init__(self):
        super().__init__()
        self.W1 = tf.Variable(tf.random.normal([6, 64], stddev=0.1, dtype=tf.float32), name="W1")
        self.b1 = tf.Variable(tf.zeros([64], dtype=tf.float32), name="b1")
        self.W2 = tf.Variable(tf.random.normal([64, 16], stddev=0.1, dtype=tf.float32), name="W2")
        self.b2 = tf.Variable(tf.zeros([16], dtype=tf.float32), name="b2")

    @tf.Module.with_name_scope
    def predict(self, state_tensor):
        layer_1 = tf.nn.relu(tf.matmul(state_tensor, self.W1) + self.b1)
        return tf.matmul(layer_1, self.W2) + self.b2

if "tf_snooker_brain" not in st.session_state:
    st.session_state.tf_snooker_brain = SnookerBrainDQN()

w1_list = st.session_state.tf_snooker_brain.W1.numpy().tolist()
b1_list = st.session_state.tf_snooker_brain.b1.numpy().tolist()
w2_list = st.session_state.tf_snooker_brain.W2.numpy().tolist()
b2_list = st.session_state.tf_snooker_brain.b2.numpy().tolist()

brain_tensors_json = json.dumps({
    "w1": w1_list, "b1": b1_list, "w2": w2_list, "b2": b2_list
})

# --- Full-Screen 1080p 3D Three.js WebGL Graphics & Inference Engine ---
html_3d_tf_snooker = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #020202; font-family: sans-serif; }}
        #canvas-holder {{ 
            width: 100vw; 
            height: 56.25vw; 
            max-height: 100vh; 
            max-width: 177.78vh; 
            margin: auto; 
            position: relative;
            box-shadow: 0px 10px 40px rgba(0,0,0,0.8);
        }}
        #ui-overlay {{
            position: absolute; top: 20px; left: 20px;
            color: #fff; background: rgba(5,5,5,0.9);
            padding: 15px 20px; border-radius: 8px; font-size: 14px;
            pointer-events: none; border: 1px solid #222; line-height: 1.6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }}
        .key-badge {{
            background: #444; border-radius: 4px; padding: 2px 6px; font-family: monospace; font-weight: bold;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="canvas-holder">
        <div id="ui-overlay">
            🧠 <b>AI Pipeline:</b> Live TensorFlow DQN Matrix Model Active<br>
            🎯 <b>Match Turn:</b> <span id="turn-display" style="color:#00aaff; font-weight:bold;">{st.session_state.turn}</span><br>
            🔴 <b>Legal Target:</b> <span style="color:#ff3333; font-weight:bold;">{st.session_state.target_type}</span><br>
            <hr style="border:0; border-top:1px solid #333; margin:10px 0;">
            <b>User Strike Controls:</b><br>
            • Rotate Cue Stick: Move via <span class="key-badge">A</span> / <span class="key-badge">D</span> or <span class="key-badge">◀</span> / <span class="key-badge">▶</span><br>
            • Fire Cue Stroke: Tap <span class="key-badge">SPACEBAR</span>
        </div>
    </div>

    <script>
        const width = 1920; const height = 1080;
        let activeTurn = "{st.session_state.turn}";
        let targetType = "{st.session_state.target_type}";
        const tfBrainTensors = {brain_tensors_json};

        function evaluateNeuralNetworkAction(ballX, ballZ, targetX, targetZ, pocketX, pocketZ) {{
            let inputState = [ballX/34, ballZ/17, targetX/34, targetZ/17, pocketX/34, pocketZ/17];
            let hiddenLayer = [];
            for (let j = 0; j < 64; j++) {{
                let outputVal = tfBrainTensors.b1[j];
                for (let i = 0; i < 6; i++) {{
                    outputVal += inputState[i] * tfBrainTensors.w1[i][j];
                }}
                hiddenLayer.push(Math.max(0, outputVal));
            }}
            let qValuesMatrix = [];
            for (let j = 0; j < 16; j++) {{
                let outputVal = tfBrainTensors.b2[j];
                for (let i = 0; i < 64; i++) {{
                    outputVal += hiddenLayer[i] * tfBrainTensors.w2[i][j];
                }}
                qValuesMatrix.push(outputVal);
            }}
            return qValuesMatrix.indexOf(Math.max(...qValuesMatrix));
        }}

        // Scene Initialization
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x050505);
        
        const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 1000);
        camera.position.set(0, 46, 58);
        camera.lookAt(0, -3, 0);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: "high-performance" }});
        renderer.setSize(width, height);
        renderer.shadowMap.enabled = true;
        document.getElementById('canvas-holder').appendChild(renderer.domElement);

        scene.add(new THREE.AmbientLight(0xffffff, 0.2));
        const overheadKeyLight = new THREE.DirectionalLight(0xffffff, 0.9); overheadKeyLight.position.set(-15, 50, 10); overheadKeyLight.castShadow = true; scene.add(overheadKeyLight);
        
        // Table Construction
        const clothMaterial = new THREE.MeshStandardMaterial({{ color: 0x114223, roughness: 0.8 }});
        const mahoganyMaterial = new THREE.MeshStandardMaterial({{ color: 0x21120b, roughness: 0.3 }});
        const mainTableCloth = new THREE.Mesh(new THREE.BoxGeometry(68, 2, 34), clothMaterial);
        mainTableCloth.position.y = -1; mainTableCloth.receiveShadow = true; scene.add(mainTableCloth);

        const cushionBorder1 = new THREE.Mesh(new THREE.BoxGeometry(70, 2.4, 1.5), mahoganyMaterial); cushionBorder1.position.set(0, 0.2, 17.75); scene.add(cushionBorder1);
        const cushionBorder2 = new THREE.Mesh(new THREE.BoxGeometry(70, 2.4, 1.5), mahoganyMaterial); cushionBorder2.position.set(0, 0.2, -17.75); scene.add(cushionBorder2);

        const pocketHoles = [
            {{x: -33.2, z: -16.2}}, {{x: 0, z: -17.2}}, {{x: 33.2, z: -16.2}},
            {{x: -33.2, z: 16.2}},  {{x: 0, z: 17.2}},  {{x: 33.2, z: 16.2}}
        ];
        pocketHoles.forEach(p => {{
            const holeMesh = new THREE.Mesh(new THREE.CylinderGeometry(1.9, 1.9, 0.2, 32), new THREE.MeshBasicMaterial({{color: 0x0a0a0a}}));
            holeMesh.position.set(p.x, 0.02, p.z); scene.add(holeMesh);
        }});

        // Balls configuration
        const ballRadius = 0.92;
        const ballGeometry = new THREE.SphereGeometry(ballRadius, 32, 32);
        const balls = [];
        const configurationsMatrix = [
            {{ type: 'cue', color: 0xffffff, points: 0, x: -18, z: 0 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 12, z: 0 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 13.8, z: 0.75 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 13.8, z: -0.75 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 15.6, z: 1.5 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 15.6, z: 0 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 15.6, z: -1.5 }},
            {{ type: 'Yellow', color: 0xfdcc0d, points: 2, x: -12, z: -4.5 }},
            {{ type: 'Green', color: 0x2ca02c, points: 3, x: -12, z: 4.5 }},
            {{ type: 'Black', color: 0x161616, points: 7, x: 25, z: 0 }}
        ];

        configurationsMatrix.forEach(cfg => {{
            const ballMat = new THREE.MeshStandardMaterial({{ color: cfg.color, roughness: 0.03, metalness: 0.12, clearcoat: 1.0 }});
            const meshInstance = new THREE.Mesh(ballGeometry, ballMat);
            meshInstance.position.set(cfg.x, ballRadius, cfg.z); meshInstance.castShadow = true; scene.add(meshInstance);
            balls.push({{ mesh: meshInstance, type: cfg.type, points: cfg.points, vx: 0, vz: 0, isPocketed: false }});
        }});

        let cueBall = balls[0];

        // Cue Stick Setup
        const cueStickContainer = new THREE.Group();
        const cueShaftMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.26, 26, 16), new THREE.MeshStandardMaterial({{ color: 0xdfbe8c, roughness: 0.5 }}));
        cueShaftMesh.rotateX(Math.PI / 2); cueShaftMesh.position.z = -13; cueStickContainer.add(cueShaftMesh);
        scene.add(cueStickContainer);

        let cueAngle = 0; 
        let isPhysicsRunning = false;
        let turnToggledThisShot = false;

        const inputsStateMap = {{ a: false, d: false, ArrowLeft: false, ArrowRight: false, ' ': false }};
        window.addEventListener('keydown', e => {{ if(e.key in inputsStateMap) inputsStateMap[e.key] = true; }});
        window.addEventListener('keyup', e => {{ if(e.key in inputsStateMap) inputsStateMap[e.key] = false; }});

        function routeScoreState(paramRoute, parameterData) {{
            const url = new URL(window.parent.location.href);
            url.searchParams.set(paramRoute, parameterData.points || "0");
            if (parameterData.next_target) url.searchParams.set("next_target", parameterData.next_target);
            if (parameterData.active_turn) url.searchParams.set("active_turn", parameterData.active_turn);
            if (parameterData.switch_turn) url.searchParams.set("switch_turn", parameterData.switch_turn);
            window.parent.location.href = url.toString();
        }}

        function executeNeuralNetworkCPUTurn() {{
            if (isPhysicsRunning || activeTurn !== "CPU_AI_Bot") return;
            
            let selectedTargetBall = null;
            for(let i = 1; i < balls.length; i++) {{
                if(!balls[i].isPocketed && (targetType === "Any" || balls[i].type === targetType)) {{
                    selectedTargetBall = balls[i];
                    break;
                }}
            }}
            if(!selectedTargetBall) selectedTargetBall = balls[balls.length - 1];

            let targetPocket = pocketHoles[0];
            let shortestDist = Infinity;
            pocketHoles.forEach(p => {{
                let d = Math.hypot(selectedTargetBall.mesh.position.x - p.x, selectedTargetBall.mesh.position.z - p.z);
                if(d < shortestDist) {{ shortestDist = d; targetPocket = p; }}
            }});

            let optimalOutputActionSegment = evaluateNeuralNetworkAction(
                cueBall.mesh.position.x, cueBall.mesh.position.z,
                selectedTargetBall.mesh.position.x, selectedTargetBall.mesh.position.z,
                targetPocket.x, targetPocket.z
            );

            let targetRadialAngle = (optimalOutputActionSegment / 16) * Math.PI * 2;
            
            setTimeout(() => {{
                cueBall.vx = Math.cos(targetRadialAngle) * 1.65;
                cueBall.vz = Math.sin(targetRadialAngle) * 1.65;
                turnToggledThisShot = false;
                isPhysicsRunning = true;
            }}, 1500);
        }}

        function animate() {{
            requestAnimationFrame(animate);

            if(!isPhysicsRunning && activeTurn === "CPU_AI_Bot") {{
                executeNeuralNetworkCPUTurn();
            }}

            // User Turn Processing
            if (!isPhysicsRunning && activeTurn === "User") {{
                if (inputsStateMap.a || inputsStateMap.ArrowLeft) cueAngle -= 0.025;
                if (inputsStateMap.d || inputsStateMap.ArrowRight) cueAngle += 0.025;
                
                cueStickContainer.visible = true;
                cueStickContainer.position.set(cueBall.mesh.position.x, cueBall.mesh.position.y, cueBall.mesh.position.z);
                cueStickContainer.rotation.y = -cueAngle;

                // FIXED: Inverted the hit vector logic to match the physical orientation of the visual cue stick mesh
                if (inputsStateMap[' ']) {{
                    let forwardImpactAngle = Math.PI - cueAngle;
                    cueBall.vx = Math.cos(forwardImpactAngle) * 1.8;
                    cueBall.vz = Math.sin(forwardImpactAngle) * 1.8;
                    turnToggledThisShot = false;
                    isPhysicsRunning = true;
                }}
            }} else if (isPhysicsRunning) {{
                cueStickContainer.visible = false;
            }}

            let totalBallsMovingCount = 0;
            balls.forEach(b => {{
                if (b.isPocketed) return;

                b.mesh.position.x += b.vx;
                b.mesh.position.z += b.vz;
                b.vx *= 0.985; b.vz *= 0.985;

                if (Math.abs(b.vx) > 0.005 || Math.abs(b.vz) > 0.005) totalBallsMovingCount++;
                else {{ b.vx = 0; b.vz = 0; }}

                let limitsX = 33.1; let limitsZ = 16.1;
                if (Math.abs(b.mesh.position.x) > limitsX) {{ b.vx *= -0.82; b.mesh.position.x = Math.sign(b.mesh.position.x) * limitsX; }}
                if (Math.abs(b.mesh.position.z) > limitsZ) {{ b.vz *= -0.82; b.mesh.position.z = Math.sign(b.mesh.position.z) * limitsZ; }}

                // Pocket Checking
                pocketHoles.forEach(p => {{
                    if (Math.hypot(b.mesh.position.x - p.x, b.mesh.position.z - p.z) < 2.1) {{
                        b.isPocketed = true; b.vx = 0; b.vz = 0;
                        b.mesh.position.set(0, -100, 0);

                        if (b.type === 'cue') {{
                            routeScoreState("foul_occured", {{ points: "User" }});
                        }} else {{
                            let ruleStringTarget = (b.type === "Red") ? "Any" : "Red";
                            routeScoreState("scored_points", {{ 
                                points: b.points, next_target: ruleStringTarget, active_turn: activeTurn, switch_turn: activeTurn 
                            }});
                        }}
                    }}
                }});
            }});

            // Ball-to-Ball Collisions
            for (let i = 0; i < balls.length; i++) {{
                for (let j = i + 1; j < balls.length; j++) {{
                    let b1 = balls[i]; let b2 = balls[j];
                    if (b1.isPocketed || b2.isPocketed) continue;

                    let dx = b2.mesh.position.x - b1.mesh.position.x;
                    let dz = b2.mesh.position.z - b1.mesh.position.z;
                    let calculatedDistance = Math.hypot(dx, dz);
                    let maximumAllowedRadiusLimit = ballRadius * 2;

                    if (calculatedDistance < maximumAllowedRadiusLimit) {{
                        let overlappingGeometryCorrectionOffset = maximumAllowedRadiusLimit - calculatedDistance;
                        let normalVectorX = dx / calculatedDistance; let normalVectorZ = dz / calculatedDistance;
                        
                        b1.mesh.position.x -= normalVectorX * overlappingGeometryCorrectionOffset * 0.5;
                        b1.mesh.position.z -= normalVectorZ * overlappingGeometryCorrectionOffset * 0.5;
                        b2.mesh.position.x += normalVectorX * overlappingGeometryCorrectionOffset * 0.5;
                        b2.mesh.position.z += normalVectorZ * overlappingGeometryCorrectionOffset * 0.5;

                        let mechanicalKineticsVelocityX = b1.vx - b2.vx; 
                        let mechanicalKineticsVelocityZ = b1.vz - b2.vz;
                        let physicalImpulseScalarValue = normalVectorX * mechanicalKineticsVelocityX + normalVectorZ * mechanicalKineticsVelocityZ;
                        
                        if (physicalImpulseScalarValue > 0) {{
                            b1.vx -= normalVectorX * physicalImpulseScalarValue; 
                            b1.vz -= normalVectorZ * physicalImpulseScalarValue;
                            b2.vx += normalVectorX * physicalImpulseScalarValue; 
                            b2.vz += normalVectorZ * physicalImpulseScalarValue;
                        }}
                    }}
                }}
            }}

            // FIXED: Smooth, non-looping shift transitions to alternating opponent turn blocks 
            if (isPhysicsRunning && totalBallsMovingCount === 0 && !turnToggledThisShot) {{
                turnToggledThisShot = true;
                isPhysicsRunning = false;
                let nextAlternatingTurn = (activeTurn === "User") ? "CPU_AI_Bot" : "User";
                routeScoreState("scored_points", {{ 
                    points: 0, next_target: targetType, active_turn: activeTurn, switch_turn: nextAlternatingTurn 
                }});
            }}

            renderer.render(scene, camera);
        }}
        
        animate();
    </script>
</body>
</html>
"""

# --- Dashboard Layout Configuration Deck ---
st.title("🎱 Full-Screen 3D TensorFlow DQN Snooker Engine")
st.markdown("High-performance rendering running **Three.js WebGL graphics pipelines** coupled with isolated **TensorFlow Python layers**.")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.metric("Your Score", st.session_state.score["User"])
with col2:
    st.metric("TensorFlow Bot Score", st.session_state.score["CPU_AI_Bot"])
with col3:
    st.write("##")
    if st.button("🔄 Restart Tournament Frame Match Assembly", use_container_width=True):
        st.session_state.score = {"User": 0, "CPU_AI_Bot": 0}
        st.session_state.turn = "User"
        st.session_state.target_type = "Red"
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.rerun()

components.html(html_3d_tf_snooker, height=640, scrolling=False)
