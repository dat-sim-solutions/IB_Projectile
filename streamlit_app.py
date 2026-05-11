import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from app.models import run_physics_simulation
import streamlit.components.v1 as components
import json

st.set_page_config(page_title="Physics Uncertainty Lab", layout="wide") 
st.title("🪂 Advanced Projectile 3D Lab: Fluid Dynamics & Error Propagation")
#st.markdown("#### Developed by: **Avalos Carrión** | 2026")
#st.divider()
# --- THEORY SECTION ---
with st.expander("📘 View Mathematical & Numerical Model"):
    st.write("The simulation models a projectile subject to a constant gravitational field and a velocity-dependent drag force (quadratic drag).")
    
    # Using LaTeX for professional formatting
    st.latex(r"m \frac{dv}{dt} = mg - \frac{1}{2} \rho C_d A v^2")
    
    st.write("Rearranging to find the instantaneous acceleration:")
    st.latex(r"a(t) = g - \frac{k}{m}v(t)^2 \quad \text{where} \quad k = \frac{1}{2} \rho C_d A")

    st.write("Analytical Solutions:")
    st.latex(r"v(t)=v_{t}\tanh \left(\frac{gt}{v_{t}}\right) \quad \text{where terminal velocity} \quad v_t = \sqrt{\frac{mg}{k}}")
    st.latex(r" a(t)= g\>\mathrm{sech}\>{}^{2}\left(\frac{gt}{v_{t}}\right) \quad \text{alternatively} \quad \>\mathrm{sech}\>{}^{2}(u) =1 - \tanh^2(u) ")
    st.latex(r"y(t)=\frac{v_{t}^{2}}{g}\ln \left[\cosh \left(\frac{gt}{v_{t}}\right)\right] ")
    
    st.info("""
    **Numerical Method:** The simulation uses the **Euler Method** for integration:
    - $v_{n+1} = v_n + a_n \Delta t$
    - $y_{n+1} = y_n + v_n \Delta t$
    """)
    
    st.write(" ")
    st.latex(r" ")
    st.write("Absolute Uncertainty in Velocity:")
    st.write("First for terminal velocity. (from Theory of Power Law)")
    st.latex(r"\frac{\Delta v_t}{v_t} = \frac{1}{2} (\frac{\Delta m}{m} + \frac{\Delta A}{A}) \text{ is calculated}")
    st.latex(r"\Delta v_t = \text{number} \times v_t")
    st.latex(r"\text{For a value } v_t \text{, the Theoretical Range is } v_t \pm \Delta v_t ")
    st.write("For the velocity as a function of terminal velocity")
    st.latex(r"\Delta v(v_t)= \left| \frac{\partial v}{\partial v_t} \right| \Delta v_t")
    st.latex(r"\frac{dv}{dv_{t}}=\tanh \left(\frac{gt}{v_{t}}\right)-\frac{gt}{v_{t}}\text{sech}^{2}\left(\frac{gt}{v_{t}}\right)")
    st.latex(r"\text{For a value } v \text{, the Theoretical Range is } v \pm \Delta v ")

    st.write("Absolute Uncertainty in acceleration (from Uncertainty of f(x))")
    st.latex(r"\Delta a = \left| \frac{da}{dv} \right| \Delta v = \left( \frac{2kv}{m} \right) \Delta v \text{ with } a = g - \frac{k}{m}v^2")

    st.info("""
    General Law of Uncertainty Propagation for $f(x,y)$
    $$ \Delta f = \sqrt{\left(\frac{\partial f}{\partial x} \Delta x\right)^2 + \left(\frac{\partial f}{\partial y} \Delta y\right)^2} $$
    - $v_{n+1} = v_n + a_n \Delta t$
    - $y_{n+1} = y_n + v_n \Delta t$
    """)
    
    st.write("Theory of Power Laws:")
    st.write("If Q")
    st.latex(r"Q = x^a y^b z^c")
    st.write("Its Fractional Uncertainty")
    st.latex(r"\frac{\Delta Q}{Q} = |a| \frac{\Delta x}{x} + |b| \frac{\Delta y}{y} + |c| \frac{\Delta z}{z}")
    

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("🎯 Physical Parameters")
mass = st.sidebar.slider("Mass (kg)", 0.1, 2.0, 0.5)
area = st.sidebar.slider("Cross-Area (m^2)", 0.001, 0.05, 0.01)

st.sidebar.header("📏 Instrument Precision (Absolute Uncertainties)")

# FIX: Explicitly set format and use floats for all arguments to prevent input locking
d_m = st.sidebar.number_input(
    "Scale (Δm in kg)", 
    min_value=0.000, 
    max_value=0.100, 
    value=0.005, 
    step=0.001, 
    format="%.3f",
    key="scale_input"
)

d_d = st.sidebar.number_input(
    "Ruler (Δd in m)", 
    min_value=0.0001, 
    max_value=0.0500, 
    value=0.0010, 
    step=0.0005, 
    format="%.4f",
    key="ruler_input"
)

u_y = st.sidebar.slider("Pos. Sensor Δy (m)", 0.05, 1.0, 0.40)
u_v = st.sidebar.slider("Vel. Sensor Δv (m/s)", 0.1, 2.0, 0.60)

t_max_limit = 10.0  # Define this globally so the slider can see it

st.sidebar.divider()
st.sidebar.header("🔍 Plot View Controls")
# This creates a double-ended slider for the time range
time_range = st.sidebar.slider(
    "Select Time Zoom (s)", 
    0.0, t_max_limit, (0.0, t_max_limit), 
    step=0.1
)
auto_scale = st.sidebar.checkbox("Auto-scale Y-axis on Zoom", value=False)


# --- RUN SIMULATION ---
data = run_physics_simulation(
    m=mass, 
    A=area, 
    delta_m=d_m, 
    delta_d=d_d, 
    y_uncert_abs=u_y, 
    v_uncert_abs=u_v,
    t_max=t_max_limit
)

# --- SECTION 1: MOTION PROFILES ---
st.subheader("1. Primary Motion Analysis")
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))

# Velocity with Dual Envelopes
ax1.fill_between(data["t"], data["v_ana"] - data["v_uncert_abs"], data["v_ana"] + data["v_uncert_abs"], 
                 color='blue', alpha=0.2, label='Sensor Precision Envelope')
ax1.fill_between(data["t"], data["v_ana"] - data["v_t_theory_err"], data["v_ana"] + data["v_t_theory_err"], 
                 color='none', edgecolor='orange', hatch='//', alpha=0.4, label='Setup Propagation (m, A)')
ax1.plot(data["t"], data["v_ana"], 'k', label='Theory (Analytical)')
ax1.plot(data["t"], data["v_num"], 'g--', label='Numerical (Euler)')
ax1.scatter(data["t"], data["v_noisy"], color='red', s=8, alpha=0.5, label='Sensor Data')
ax1.set_ylabel("Velocity (m/s)"); ax1.legend(loc='lower right'); ax1.grid(True, alpha=0.3)

# Acceleration
ax2.fill_between(data["t"], data["a_ana"] - data["a_uncert_abs"], data["a_ana"] + data["a_uncert_abs"], color='blue', alpha=0.15)
ax2.plot(data["t"], data["a_ana"], 'k')
ax2.plot(data["t"], data["a_num"], 'g--')
ax2.scatter(data["t"], data["a_noisy"], color='red', s=8, alpha=0.5)
ax2.set_ylabel("Acceleration (m/s^2)"); ax2.grid(True, alpha=0.3)

# Position
ax3.fill_between(data["t"], data["y_ana"] - data["y_uncert_abs"], data["y_ana"] + data["y_uncert_abs"], color='blue', alpha=0.15)
ax3.plot(data["t"], data["y_ana"], 'k')
ax3.plot(data["t"], data["y_num"], 'g--')
ax3.scatter(data["t"], data["y_noisy"], color='red', s=8, alpha=0.5)
ax3.set_ylabel("Position (m)"); ax3.set_xlabel("Time (s)"); ax3.grid(True, alpha=0.3)

# --- APPLY GLOBAL VIEW SETTINGS (The "Camera" Work) ---
# Apply the X-axis zoom to all
ax1.set_xlim(time_range)
ax2.set_xlim(time_range)
ax3.set_xlim(time_range)

if auto_scale:
    # 1. Create a mask to find data within the time_range
    mask = (data["t"] >= time_range[0]) & (data["t"] <= time_range[1])
    
    # 2. Zoom AX1 (Velocity)
    v_visible = data["v_noisy"][mask]
    if len(v_visible) > 0:
        ax1.set_ylim(np.min(v_visible) - 0.5, np.max(v_visible) + 0.5)

    # 3. Zoom AX2 (Acceleration)
    a_visible = data["a_ana"][mask] # Using analytical to avoid outlier spikes
    if len(a_visible) > 0:
        ax2.set_ylim(np.min(a_visible) - 1.0, np.max(a_visible) + 1.0)

    # 4. Zoom AX3 (Position)
    y_visible = data["y_noisy"][mask]
    if len(y_visible) > 0:
        ax3.set_ylim(np.min(y_visible) - 1.0, np.max(y_visible) + 1.0)

st.pyplot(fig1)

# --- SECTION 2: STATISTICAL ERROR ANALYSIS ---
st.divider()
st.subheader("2. Statistical Verification of Sensor Noise, for v")
res = data["residuals"]
fig2, (ax_res, ax_box, ax_vio) = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1, 1]})

# Residuals with Envelopes
ax_res.axhline(0, color='black', linestyle='--')
ax_res.fill_between(data["t"], -data["v_t_theory_err"], data["v_t_theory_err"], color='orange', alpha=0.1, label='Setup Limit')
ax_res.fill_between(data["t"], -data["v_uncert_abs"], data["v_uncert_abs"], color='gray', alpha=0.2, label='Sensor Limit')
ax_res.scatter(data["t"], res, color='red', s=8, alpha=0.4)
ax_res.set_ylabel("Error (m/s)"); ax_res.set_title("Residuals vs. Time")
ax_res.set_xlim(time_range)

# Box Plot
ax_box.boxplot(res, vert=True, patch_artist=True, boxprops=dict(facecolor='tomato', alpha=0.4))
ax_box.set_ylim(ax_res.get_ylim()); ax_box.set_xticks([1]); ax_box.set_xticklabels(['Data'])

# Violin + Normal Curve Overlay
v_plot = ax_vio.violinplot(res, showmedians=True)
for pc in v_plot['bodies']: pc.set_facecolor('gray'), pc.set_alpha(0.5)
mu, std = np.mean(res), np.std(res)
y_range = np.linspace(ax_res.get_ylim()[0], ax_res.get_ylim()[1], 100)
p = norm.pdf(y_range, mu, std)
p_scaled = (p / max(p)) * 0.4
ax_vio.plot(1 + p_scaled, y_range, 'k--', label='Normal Dist.')
ax_vio.plot(1 - p_scaled, y_range, 'k--')
ax_vio.set_ylim(ax_res.get_ylim()); ax_vio.set_xticks([1]); ax_vio.set_xticklabels(['Density'])

st.pyplot(fig2)

# --- SECTION 3: MICRO-ANALYSIS WINDOW ---
st.divider()
st.subheader("3. Local Interval Inspection (4.0s - 5.0s)")
mask = (data["t"] >= 4.0) & (data["t"] <= 5.0)
fig3, ax_zoom = plt.subplots(figsize=(10, 4))
ax_zoom.plot(data["t"][mask], data["v_ana"][mask], 'k', label='Theory')
ax_zoom.errorbar(data["t"][mask], data["v_noisy"][mask], xerr=0.05, yerr=data["v_uncert_abs"], 
                 fmt='o', color='red', ecolor='gray', capsize=3, label='Sensor + Sync Error Boxes')
ax_zoom.set_xlabel("Time (s)"); ax_zoom.set_ylabel("Velocity (m/s)"); ax_zoom.grid(alpha=0.3); ax_zoom.legend()
st.pyplot(fig3)

# ---- 3D -----

def render_3d_simulation(data):    
    # Ensure data is clean for JS
    y_vals = [float(y) for y in data["y_ana"]]
    v_vals = [float(v) for v in data["v_ana"]]
    t_vals = [float(t) for t in data["t"]]
    
    y_data_js = json.dumps(y_vals)
    v_data_js = json.dumps(v_vals)
    t_data_js = json.dumps(t_vals)
    
    html_code = f"""
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <style>
            body {{ margin: 0; background-color: #f0f2f6; overflow: hidden; font-family: sans-serif; }}
            #canvas-container {{ 
                width: 100vw; 
                height: 500px; 
                cursor: pointer; 
            }}
            #ui-hint {{ 
                position: absolute; top: 10px; left: 10px; 
                background: rgba(255,255,255,0.8); padding: 8px 12px; 
                border-radius: 8px; font-size: 13px; font-weight: bold;
                pointer-events: none; z-index: 100;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                color: #1c2b42;
            }}
        </style>
    </head>
    <body>
        <div id="ui-hint">🖱️ Click 3D view to change the Reference Frame from Rotating to Stationary.</div>
        <div id="canvas-container"></div>
        
        <script>
            // Wrap in an IIFE to avoid global scope issues
            (function() {{
                try {{
                    if (typeof THREE === 'undefined') {{
                        document.getElementById('canvas-container').innerHTML = "<p style='padding:20px; color:red;'>Error: Three.js library not loaded.</p>";
                        return;
                    }}

                    const yData = {y_data_js};
                    const vData = {v_data_js};
                    const tData = {t_data_js};
                    const scale = 0.1; 
                    const sceneHeight = 300 * scale; 
                    let frame = 0;
                    let angle = 0;
                    const radius = 50; 
                    let isRotating = true;

                    // --- SCENE SETUP ---
                    const scene = new THREE.Scene();
                    scene.background = new THREE.Color(0xf0f2f6);

                    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / 500, 0.1, 1000);
                    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                    renderer.setSize(window.innerWidth, 500);
                    const container = document.getElementById('canvas-container');
                    container.appendChild(renderer.domElement);
                    
                    // --- CLICK TOGGLE ---
                    container.addEventListener('click', () => {{
                        isRotating = !isRotating;
                    }});
                    
                    // --- NEW: PARTICLE SYSTEM (SNOW/DUST) ---
                    const particleCount = 2000;
                    const particlesGeo = new THREE.BufferGeometry();
                    const positions = new Float32Array(particleCount * 3);
                    
                    for (let i = 0; i < particleCount * 3; i += 3) {{
                        positions[i] = (Math.random() - 0.5) * 200;     // X
                        positions[i+1] = Math.random() * 100;           // Y (Height)
                        positions[i+2] = (Math.random() - 0.5) * 200;   // Z
                    }}
                    particlesGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    
                    const pMaterial = new THREE.PointsMaterial({{
                        color: 0xffffff,
                        size: 0.4,
                        transparent: true,
                        opacity: 0.6
                    }});
                    
                    const particleSystem = new THREE.Points(particlesGeo, pMaterial);
                    scene.add(particleSystem);
                    
                    // --- NEW: PROCEDURAL MOUNTAINS ---
                    const mtnGeo = new THREE.PlaneGeometry(400, 400, 50, 50);
                    const verts = mtnGeo.attributes.position.array;
                    
                    for (let i = 0; i < verts.length; i += 3) {{
                        const x = verts[i];
                        const y = verts[i+1];
                        // Using Sine sums to simulate Perlin Noise topography
                        const d = Math.sqrt(x*x + y*y);
                        if (d > 50) {{ // Only raise mountains far away from the building
                            verts[i+2] = (Math.sin(x * 0.1) * Math.cos(y * 0.1) * 10) + 
                                         (Math.sin(x * 0.05) * 5);
                        }}
                    }}
                    mtnGeo.computeVertexNormals();
                    const mtnMat = new THREE.MeshPhongMaterial({{ color: 0x3d4a35, flatShading: true }});
                    const mountains = new THREE.Mesh(mtnGeo, mtnMat);
                    mountains.rotation.x = -Math.PI / 2; // Lay flat
                    mountains.position.y = -0.5; // Slightly below grid
                    scene.add(mountains);
                    
                    // --- NEW: TREE CONSTRUCTOR ---
                    function addTree(x, z) {{
                        const trunkGeo = new THREE.CylinderGeometry(0.5, 0.7, 4, 8);
                        const trunkMat = new THREE.MeshPhongMaterial({{ color: 0x5D4037 }});
                        const trunk = new THREE.Mesh(trunkGeo, trunkMat);
                        trunk.position.set(x, 2, z);
                        scene.add(trunk);

                        const leavesGeo = new THREE.ConeGeometry(3, 6, 8);
                        const leavesMat = new THREE.MeshPhongMaterial({{ color: 0x2E7D32 }});
                        const leaves = new THREE.Mesh(leavesGeo, leavesMat);
                        leaves.position.set(x, 6, z);
                        scene.add(leaves);
                    }}

                    // Place a few trees around the building
                    addTree(15, 15); addTree(-20, 10); addTree(10, -25); addTree(10, -18);

                    // --- HELPER: TEXT SPRITE ---
                    function createText(text, color="black") {{
                        const canvas = document.createElement('canvas');
                        const ctx = canvas.getContext('2d');
                        canvas.width = 512; canvas.height = 128; // Higher resolution
                        ctx.fillStyle = color;
                        ctx.font = 'Bold 60px Arial';
                        ctx.fillText(text, 10, 80);
                        const texture = new THREE.CanvasTexture(canvas);
                        const spriteMat = new THREE.SpriteMaterial({{ map: texture }});
                        const sprite = new THREE.Sprite(spriteMat);
                        sprite.scale.set(15, 4, 1);
                        return sprite;
                    }}

                    // --- 1. FLOOR & RULER ---
                    scene.add(new THREE.GridHelper(200, 20, 0x888888, 0xcccccc));

                    for (let m = 0; m <= 300; m += 50) {{
                        const yPos = sceneHeight - (m * scale);
                        const line = new THREE.Mesh(
                            new THREE.BoxGeometry(12, 0.1, 0.1), 
                            new THREE.MeshBasicMaterial({{color: 0x999999}})
                        );
                        line.position.set(6, yPos, 0);
                        scene.add(line);

                        const label = createText(m + "m", "#111111");
                        label.position.set(8, yPos, 0);
                        scene.add(label);
                    }}

                    // --- 2. BUILDING ---
                    const bldMat = new THREE.MeshPhongMaterial({{ 
                        color: 0x1c2b42, transparent: true, opacity: 0.3, shininess: 50 
                    }});
                    const building = new THREE.Mesh(new THREE.BoxGeometry(6, sceneHeight, 6), bldMat);
                    building.position.y = sceneHeight / 2;
                    scene.add(building);

                    // --- 3. MAIN SPHERE ---
                    const sphere = new THREE.Mesh(
                        new THREE.SphereGeometry(1.2, 32, 32), 
                        new THREE.MeshPhongMaterial({{ color: 0xff4b4b, shininess: 100 }})
                    );
                    scene.add(sphere);

                    // --- 4. GHOSTS ---
                    const ghostMat = new THREE.MeshPhongMaterial({{ color: 0xff4b4b, transparent: true, opacity: 0.5 }});
                    const segments = 5;
                    for (let i = 1; i < segments; i++) {{
                        const idx = Math.floor((i / segments) * yData.length);
                        const yPos = sceneHeight - (yData[idx] * scale);
                        const ghost = new THREE.Mesh(new THREE.SphereGeometry(1.2, 32, 32), ghostMat);
                        ghost.position.set(-6, yPos, 0);
                        scene.add(ghost);

                        const info = "t:" + tData[idx].toFixed(1) + "s; v:" + vData[idx].toFixed(1)+ " m/s";
                        const label = createText(info, "#ff1111");
                        label.position.set(-12, yPos, 0);
                        scene.add(label);
                    }}

                    // --- LIGHTS ---
                    scene.add(new THREE.AmbientLight(0x404040, 2));
                    const light = new THREE.DirectionalLight(0xffffff, 1);
                    light.position.set(10, 50, 10);
                    scene.add(light);
                    
                    function animate() {{
                        requestAnimationFrame(animate);
                        
                        if (isRotating) {{
                            angle += 0.005; 
                        }}
                        camera.position.x = Math.cos(angle) * radius;
                        camera.position.z = Math.sin(angle) * radius;
                        camera.position.y = 30;
                        camera.lookAt(0, 15, 0);
                                
                        if (frame < yData.length) {{
                            sphere.position.y = sceneHeight - (yData[frame] * scale);
                            frame++;
                        }} else {{ frame = 0; }}
                        renderer.render(scene, camera);
                    }}
                    animate();

                }} catch (err) {{
                    document.getElementById('canvas-container').innerHTML = "<p style='color:red;'>Script Error: " + err.message + "</p>";
                }}
            }})();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=520)

st.divider()
st.subheader("4. 3D Digital Twin: Real-Time Visualization")
st.write("Watch the physics in action. The building is scaled to **300m** (approx. 80 stories).")
render_3d_simulation(data)

# Status Footer --------
status_color = "green" if 0.05 < data['cfl_limit'] else "red"
st.sidebar.markdown(f"**Stability (CFL):** :{status_color}[{data['cfl_limit']:.4f}s]")

# --- FOOTER / DEVELOPER INFO ---
st.sidebar.divider()
st.sidebar.markdown(f"**Developer:** Juan Avalos Carrión")
st.sidebar.markdown(f"Numerical Modeling + AI | Geophysics Data Engineer")
st.sidebar.markdown(f"www.linkedin.com/in/juan-a-c-02a51420b")
st.sidebar.caption(" 2026. Physics Simulation Framework v1.0")
