import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Enhanced 3D Snooker Arena", layout="wide")

# Core Game Engine State Memory
if "score" not in st.session_state:
    st.session_state.score = {"User": 0, "CPU_AI": 0}
    st.session_state.turn = "User"  # User takes the first break shot
    st.session_state.target_type = "Red"  # Rules dictate starting with a Red ball

# Synchronize backend state via standard iframe query parameters
query_params = st.query_params
if "scored_points" in query_params:
    pts = int(query_params["scored_points"])
    current_turn = query_params.get("active_turn", "User")
    st.session_state.score[current_turn] += pts
    st.session_state.target_type = query_params.get("next_target", "Red")
    st.session_state.turn = query_params.get("switch_turn", current_turn)
    st.query_params.clear()
    st.toast(f"🎱 {current_turn} scored {pts} point(s)! Target is now: {st.session_state.target_type}")
elif "foul_occured" in query_params:
    foul_by = query_params["foul_occured"]
    penalty_receiver = "CPU_AI" if foul_by == "User" else "User"
    st.session_state.score[penalty_receiver] += 4  # Standard minimum snooker foul penalty
    st.session_state.turn = "User"  # Hand match play back to human user
    st.session_state.target_type = "Red"
    st.query_params.clear()
    st.toast(f"⚠️ FOUL by {foul_by}! 4 Points awarded to {penalty_receiver}.")

# --- Production 1080p 3D Three.js Snooker Matrix Engine ---
html_snooker_engine = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #030303; font-family: sans-serif; }}
        #canvas-holder {{ 
            width: 100vw; 
            height: 56.25vw; /* Native 1080p 16:9 Aspect Ratio */
            max-height: 100vh; 
            max-width: 177.78vh; 
            margin: auto; 
            position: relative;
        }}
        #ui-overlay {{
            position: absolute; top: 15px; left: 15px;
            color: #fff; background: rgba(0,0,0,0.85);
            padding: 12px 18px; border-radius: 8px; font-size: 14px;
            pointer-events: none; border: 1px solid #333; line-height: 1.5;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="canvas-holder">
        <div id="ui-overlay">
            🎯 <b>Current Turn:</b> {st.session_state.turn}<br>
            🔴 <b>Required Target Color:</b> {st.session_state.target_type}<br>
            🎮 <b>Controls (User Turn Only):</b> Use <b>A / D</b> or <b>Left / Right Arrows</b> to rotate Cue Stick.<br>
            💥 Press <b>SPACEBAR</b> to execute shot.
        </div>
    </div>

    <script>
        const width = 1920; const height = 1080;
        let activeTurn = "{st.session_state.turn}";
        let targetType = "{st.session_state.target_type}";

        // 1. Initial WebGL Canvas Setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x050505);
        
        const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 1000);
        camera.position.set(0, 48, 60);
        camera.lookAt(0, -2, 0);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: "high-performance" }});
        renderer.setSize(width, height);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.getElementById('canvas-holder').appendChild(renderer.domElement);

        // Professional Lighting Array
        scene.add(new THREE.AmbientLight(0xffffff, 0.25));
        const light1 = new THREE.DirectionalLight(0xffffff, 0.85); light1.position.set(-10, 45, 5); light1.castShadow = true; scene.add(light1);
        const light2 = new THREE.DirectionalLight(0xffffff, 0.45); light2.position.set(10, 45, -5); scene.add(light2);

        // 2. High-Quality Tournament Table Model Assembly
        const clothMat = new THREE.MeshStandardMaterial({{ color: 0x144c2a, roughness: 0.85, metalness: 0.05 }});
        const cushionMat = new THREE.MeshStandardMaterial({{ color: 0x2b1810, roughness: 0.3 }});
        
        const table = new THREE.Mesh(new THREE.BoxGeometry(68, 2, 34), clothMat);
        table.position.y = -1; table.receiveShadow = true; scene.add(table);

        const railL1 = new THREE.Mesh(new THREE.BoxGeometry(70, 2.5, 1.5), cushionMat); railL1.position.set(0, 0.25, 17.75); scene.add(railL1);
        const railL2 = new THREE.Mesh(new THREE.BoxGeometry(70, 2.5, 1.5), cushionMat); railL2.position.set(0, 0.25, -17.75); scene.add(railL2);

        // Standard Pockets Configuration Matrix
        const pockets = [
            {{x: -33, z: -16.5}}, {{x: 0, z: -17}}, {{x: 33, z: -16.5}},
            {{x: -33, z: 16.5}},  {{x: 0, z: 17}},  {{x: 33, z: 16.5}}
        ];
        pockets.forEach(p => {{
            const hole = new THREE.Mesh(new THREE.CylinderGeometry(2.0, 2.0, 0.2, 32), new THREE.MeshBasicMaterial({{color: 0x111111}}));
            hole.position.set(p.x, 0.02, p.z);
            scene.add(hole);
        }});

        // 3. Ultra-Smooth High Gloss Balls Initialization
        const ballRadius = 0.95;
        const ballGeo = new THREE.SphereGeometry(ballRadius, 32, 32);
        
        const balls = [];
        const configurations = [
            {{ type: 'cue', color: 0xffffff, points: 0, x: -18, z: 0 }},
            // Reds Cluster Array
            {{ type: 'Red', color: 0xd62728, points: 1, x: 12, z: 0 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 14, z: 0.8 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 14, z: -0.8 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 16, z: 1.6 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 16, z: 0 }},
            {{ type: 'Red', color: 0xd62728, points: 1, x: 16, z: -1.6 }},
            // High-Value Colors
            {{ type: 'Yellow', color: 0xfdcc0d, points: 2, x: -10, z: -5 }},
            {{ type: 'Green', color: 0x2ca02c, points: 3, x: -10, z: 5 }},
            {{ type: 'Black', color: 0x111111, points: 7, x: 26, z: 0 }}
        ];

        configurations.forEach(cfg => {{
            const mat = new THREE.MeshStandardMaterial({{ color: cfg.color, roughness: 0.05, metalness: 0.15 }});
            const mesh = new THREE.Mesh(ballGeo, mat);
            mesh.position.set(cfg.x, ballRadius, cfg.z);
            mesh.castShadow = true;
            scene.add(mesh);
            balls.push({{ mesh: mesh, type: cfg.type, points: cfg.points, vx: 0, vz: 0, isPocketed: false }});
        }});

        // Get safe access pointer to the Cue Ball object
        let cueBall = balls[0];

        // 4. Enhanced Render Model of Visual Aiming Cue Stick
        const cueStickGeo = new THREE.CylinderGeometry(0.15, 0.3, 24, 16);
        cueStickGeo.rotateX(Math.PI / 2); // Rotate to align horizontally
        const cueStickMat = new THREE.MeshStandardMaterial({{ color: 0xddba76, roughness: 0.4, metalness: 0.2 }});
        const cueStick = new THREE.Mesh(cueStickGeo, cueStickMat);
        scene.add(cueStick);

        let cueAngle = 0; // Measured in Radian dimensions
        let isMoving = false;

        // User Input Controls Keyboard Interface Loop
        const keys = {{ a: false, d: false, ArrowLeft: false, ArrowRight: false, ' ': false }};
        window.addEventListener('keydown', e => {{ if(e.key in keys) keys[e.key] = true; }});
        window.addEventListener('keyup', e => {{ if(e.key in keys) keys[e.key] = false; }});

        // Communication Bridge back to Streamlit app instance
        function notifyBackend(routeParam, args) {{
            const url = new URL(window.parent.location.href);
            url.searchParams.set(routeParam, args.points || "1");
            if (args.next_target) url.searchParams.set("next_target", args.next_target);
            if (args.active_turn) url.searchParams.set("active_turn", args.active_turn);
            if (args.switch_turn) url.searchParams.set("switch_turn", args.switch_turn);
            window.parent.location.href = url.toString();
        }}

        // 5. Native Automated CPU AI Routine Logic Engine
        function executeCPUTurn() {{
            if (isMoving || activeTurn !== "CPU_AI") return;
            
            // AI targets an object based on the active rules sequence
            let validTarget = null;
            for(let i=1; i<balls.length; i++) {{
                if(!balls[i].isPocketed && (targetType === "Any" || balls[i].type === targetType)) {{
                    validTarget = balls[i];
                    break;
                }}
            }}
            if(!validTarget) validTarget = balls[balls.length - 1]; // Default safe target backup

            // Calculate directional vector angles directly to target
            let dx = validTarget.mesh.position.x - cueBall.mesh.position.x;
            let dz = validTarget.mesh.position.z - cueBall.mesh.position.z;
            let baseAngle = Math.atan2(dz, dx);
            
            // Inject minor variance factor to mimic realistic human calculation limitations
            let aiError = (Math.random() - 0.5) * 0.08;
            let finalAngle = baseAngle + aiError;

            setTimeout(() => {{
                cueBall.vx = Math.cos(finalAngle) * 1.5;
                cueBall.vz = Math.sin(finalAngle) * 1.5;
                isMoving = true;
            }}, 1200); // 1.2s Artificial computation latency delay
        }}

        // 6. Vector Math Mechanical Physics Realtime Frame Core
        function animate() {{
            requestAnimationFrame(animate);

            // Turn Execution Watchdog Pipeline
            if(!isMoving && activeTurn === "CPU_AI") {{
                executeCPUTurn();
            }}

            // Handle manual human adjustment calculations during active player turn windows
            if (!isMoving && activeTurn === "User") {{
                if (keys.a || keys.ArrowLeft) cueAngle -= 0.025;
                if (keys.d || keys.ArrowRight) cueAngle += 0.025;
                
                // Keep the aiming cue stick pointing directly at the white cue ball
                cueStick.visible = true;
                let offsetDistance = 14.5; 
                cueStick.position.set(
                    cueBall.mesh.position.x - Math.cos(cueAngle) * offsetDistance,
                    cueBall.mesh.position.y + 0.3,
                    cueBall.mesh.position.z - Math.sin(cueAngle) * offsetDistance
                );
                cueStick.rotation.y = -cueAngle;

                // Fire Trigger Hook
                if (keys[' ']) {{
                    cueBall.vx = Math.cos(cueAngle) * 1.6;
                    cueBall.vz = Math.sin(cueAngle) * 1.6;
                    isMoving = true;
                }}
            }} else {{
                cueStick.visible = false; // Hide cue stick when balls are in motion
            }}

            // Process position displacements via speed components
            let rollingCount = 0;
            balls.forEach(b => {{
                if (b.isPocketed) return;

                b.mesh.position.x += b.vx;
                b.mesh.position.z += b.vz;
                
                // Continuous friction dissipation
                b.vx *= 0.988; b.vz *= 0.988;

                if (Math.abs(b.vx) > 0.005 || Math.abs(b.vz) > 0.005) rollingCount++;
                else {{ b.vx = 0; b.vz = 0; }}

                // Outer Cushion Boundary Elastic Impact Limits
                let borderX = 33; let borderZ = 16;
                if (Math.abs(b.mesh.position.x) > borderX) {{ b.vx *= -0.85; b.mesh.position.x = Math.sign(b.mesh.position.x) * borderX; }}
                if (Math.abs(b.mesh.position.z) > borderZ) {{ b.vz *= -0.85; b.mesh.position.z = Math.sign(b.mesh.position.z) * borderZ; }}

                // Check Pocket Intersections
                pockets.forEach(p => {{
                    if (Math.hypot(b.mesh.position.x - p.x, b.mesh.position.z - p.z) < 2.2) {{
                        b.isPocketed = true;
                        b.vx = 0; b.vz = 0;
                        b.mesh.position.set(0, -50, 0); // Remove from field coordinates safely

                        // Rule Enforcement Sequence evaluation logic
                        if (b.type === 'cue') {{
                            notifyBackend("foul_occured", {{ points: "User" }});
                        }} else {{
                            // Legitimate score tracking logic rules parsing
                            let nextRuleTarget = (b.type === "Red") ? "Any" : "Red";
                            let nextTurnPlayer = activeTurn; // Maintain turn sequence on successful scores
                            notifyBackend("scored_points", {{ 
                                points: b.points, 
                                next_target: nextRuleTarget,
                                active_turn: activeTurn,
                                switch_turn: nextTurnPlayer
                            }});
                        }}
                    }}
                }});
            }});

            // Ball-to-Ball Elastic Collisions Loop Setup
            for (let i = 0; i < balls.length; i++) {{
                for (let j = i + 1; j < balls.length; j++) {{
                    let b1 = balls[i]; let b2 = balls[j];
                    if (b1.isPocketed || b2.isPocketed) continue;

                    let dx = b2.mesh.position.x - b1.mesh.position.x;
                    let dz = b2.mesh.position.z - b1.mesh.position.z;
                    let dist = Math.hypot(dx, dz);
                    let minDist = ballRadius * 2;

                    if (dist < minDist) {{
                        // Separate overlapping geometry positions immediately
                        let overlap = minDist - dist;
                        let nx = dx / dist; let nz = dz / dist;
                        b1.mesh.position.x -= nx * overlap * 0.5;
                        b1.mesh.position.z -= nz * overlap * 0.5;
                        b2.mesh.position.x += nx * overlap * 0.5;
                        b2.mesh.position.z += nz * overlap * 0.5;

                        // Calculate elastic vector speed exchanges
                        let kx = b1.vx - b2.vx; let kz = b1.vz - b2.vz;
                        let impulse = nx * kx + nz * kz;
                        if (impulse > 0) {{
                            b1.vx -= nx * impulse; b1.vz -= nz * impulse;
                            b2.vx += nx * impulse; b2.vz += nz * impulse;
                        }}
                    }}
                }}
            }}

            // Transition turn rules if all entities have come to complete rest without scores
            if (isMoving && rollingCount === 0) {{
                isMoving = false;
                let alternatingTurn = (activeTurn === "User") ? "CPU_AI" : "User";
                notifyBackend("scored_points", {{ 
                    points: 0, 
                    next_target: targetType, 
                    active_turn: activeTurn, 
                    switch_turn: alternatingTurn 
                }});
            }}

            renderer.render(scene, camera);
        }}
        
        animate();
    </script>
</body>
</html>
"""

# --- Dashboard View Interface Layout Configuration ---
st.title("🎱 Professional 1080p 3D Snooker Match Engine")
st.markdown("Equipped with **Interactive Cue Rotation Mechanics**, Official Alternate Scoring Protocols, and a Local CPU Opponent Pipeline.")

col1, col2, col3 = st.columns([1,1,2])
with col1:
    st.metric("Your Match Score", st.session_state.score["User"])
with col2:
    st.metric("CPU AI Score", st.session_state.score["CPU_AI"])
with col3:
    st.write("##")
    if st.button("🔄 Restart Tournament Frame Match", use_container_width=True):
        st.session_state.score = {"User": 0, "CPU_AI": 0}
        st.session_state.turn = "User"
        st.session_state.target_type = "Red"
        st.query_params.clear()
        st.rerun()

# Inject fully isolated 3D view interface frame logic execution block inside dashboard view
components.html(html_snooker_engine, height=620, scrolling=False)
