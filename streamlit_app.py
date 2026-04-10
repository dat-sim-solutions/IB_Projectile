import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from app.models import run_physics_simulation
import streamlit.components.v1 as components
import json

st.set_page_config(page_title="Physics Uncertainty Lab", layout="wide") 
st.title("🪂 Advanced Projectile Lab: Fluid Dynamics & Error Propagation")
#st.markdown("#### Developed by: **Avalos Carrión** | 2026")
#st.divider()
# --- THEORY SECTION ---
with st.expander("📘 View Mathematical & Numerical Model"):
    st.write("The simulation models a projectile subject to a constant gravitational field and a velocity-dependent drag force (quadratic drag).")
    
    # Using LaTeX for professional formatting
    st.latex(r"m \frac{dv}{dt} = mg - \frac{1}{2} \rho C_d A v^2")
    
    st.write("Rearranging to find the instantaneous acceleration:")
    st.latex(r"a(t) = g - \frac{k}{m}v(t)^2 \quad \text{where} \quad k = \frac{1}{2} \rho C_d A")
    
    st.info("""
    **Numerical Method:** The simulation uses the **Euler Method** for integration:
    - $v_{n+1} = v_n + a_n \Delta t$
    - $y_{n+1} = y_n + v_n \Delta t$
    """)

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
st.subheader("2. Statistical Verification of Sensor Noise")
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
            body {{ margin: 0; background-color: #f0f2f6; overflow: hidden; }}
            #canvas-container {{ width: 100%; height: 500px; }}
        </style>
    </head>
    <body>
        <div id="canvas-container"></div>
        <script>
            try {{
                const yData = {y_data_js};
                const vData = {v_data_js};
                const tData = {t_data_js};
                const scale = 0.1; 
                const sceneHeight = 300 * scale; 
                let frame = 0;

                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf0f2f6);

                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / 500, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, 500);
                document.getElementById('canvas-container').appendChild(renderer.domElement);

                // --- HELPER: TEXT SPRITE ---
                function createText(text, color="black") {{
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    canvas.width = 256; canvas.height = 64;
                    ctx.fillStyle = color;
                    ctx.font = 'Bold 44px Arial';
                    ctx.fillText(text, 0, 45);
                    const texture = new THREE.CanvasTexture(canvas);
                    const spriteMat = new THREE.SpriteMaterial({{ map: texture }});
                    const sprite = new THREE.Sprite(spriteMat);
                    // sprite.scale.set(Width, Height, Depth)
                    sprite.scale.set(15, 5, 2);
                    return sprite;
                }}

                // --- 1. THE GRID (The 3D Floor) ---
                const grid = new THREE.GridHelper(200, 20, 0x888888, 0xcccccc);
                scene.add(grid);

                // --- 1. THE RULER (Every 50m) ---
                for (let m = 0; m <= 300; m += 50) {{
                    const yPos = sceneHeight - (m * scale);
                    // Line
                    const lineGeo = new THREE.BoxGeometry(10, 0.1, 0.1);
                    const line = new THREE.Mesh(lineGeo, new THREE.MeshBasicMaterial({{color: 0x999999}}));
                    line.position.set(5, yPos, 0);
                    scene.add(line);
                    // Label
                    const label = createText(m + "m", "#555555");
                    label.position.set(12, yPos, 0);
                    scene.add(label);
                }}

                // --- 2. BUILDING ---
                const bldMat = new THREE.MeshPhongMaterial({{ color: 0x1c2b42, transparent: true, opacity: 0.3, shininess: 100 }});
                const building = new THREE.Mesh(new THREE.BoxGeometry(6, sceneHeight, 6), bldMat);
                building.position.y = sceneHeight / 2;
                scene.add(building);

                // --- 3. MAIN SPHERE ---
                const sphere = new THREE.Mesh(new THREE.SphereGeometry(1.2, 32, 32), new THREE.MeshPhongMaterial({{ color: 0xff4b4b, shininess: 100 }}));
                scene.add(sphere);

                // --- 4. GHOSTS WITH DATA LABELS ---
                const ghostMat = new THREE.MeshPhongMaterial({{ color: 0xff4b4b, transparent: true, opacity: 0.3 }});
                const segments = 5;
                for (let i = 1; i < segments; i++) {{
                    const idx = Math.floor((i / segments) * yData.length);
                    // Ghost
                    const ghost = new THREE.Mesh(new THREE.SphereGeometry(1.2, 32, 32), ghostMat);
                    const yPos = sceneHeight - (yData[idx] * scale);
                    ghost.position.set(-5, yPos, 0);
                    scene.add(ghost);

                    // Label: "t=2s, v=15m/s"
                    const info = "t:" + tData[idx].toFixed(1) + "s v:" + vData[idx].toFixed(1);
                    const label = createText(info, "#ff4b4b");
                    label.position.set(-15, yPos, 0);
                    scene.add(label);
                }}

                // --- LIGHTS & CAMERA ---
                scene.add(new THREE.AmbientLight(0x404040, 2));
                const light = new THREE.DirectionalLight(0xffffff, 1);
                light.position.set(10, 20, 10);
                scene.add(light);

                camera.position.set(60, 20, 60);
                camera.lookAt(0, 15, 0);

                function animate() {{
                    requestAnimationFrame(animate);
                    if (frame < yData.length) {{
                        sphere.position.y = sceneHeight - (yData[frame] * scale);
                        frame++;
                    }} else {{ frame = 0; }}
                    renderer.render(scene, camera);
                }}
                animate();
            }} catch (e) {{ console.error(e); }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=520)

st.divider()
st.subheader("4. 3D Digital Twin: Real-Time Visualization")
st.write("Watch the physics in action. The building is scaled to **300m** (approx. 80 stories).")
# render_3d_simulation(data["y_ana"])
render_3d_simulation(data)

# Status Footer --------
status_color = "green" if 0.05 < data['cfl_limit'] else "red"
st.sidebar.markdown(f"**Stability (CFL):** :{status_color}[{data['cfl_limit']:.4f}s]")

# --- FOOTER / DEVELOPER INFO ---
st.sidebar.divider()
st.sidebar.markdown(f"**Developer:** Juan Avalos Carrión")
st.sidebar.markdown(f"Numerical Modeling & Data Science Specialist")
st.sidebar.markdown(f"www.linkedin.com/in/juan-a-c-02a51420b")
st.sidebar.caption(" 2026. Physics Simulation Framework v1.0")
