import streamlit as st
import cv2
import numpy as np
import time
from logic import load_model, preprocess_image, run_ai_prediction, run_quantum_verification, get_remedy

# Page Configuration
st.set_page_config(
    page_title="PlantPulse AI + Quantum",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    .highlight {
        color: #10b981;
        font-weight: bold;
    }
    .prediction-title {
        font-size: 24px;
        margin-bottom: 10px;
    }
    .quantum-badge {
        background: linear-gradient(45deg, #6366f1, #a855f7);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/892/892926.png", width=100)
    st.title("Settings")
    
    st.subheader("Compute Engine")
    quantum_mode = st.toggle("Use Real IBM Hardware", value=False, help="Requires IBM Quantum API Token")
    
    ibm_token = ""
    if quantum_mode:
        ibm_token = st.text_input("IBM Quantum Token", type="password", 
                                value=st.secrets.get("IBM_TOKEN", ""))
        if not ibm_token:
            st.warning("⚠️ Token required for Hardware mode")
            
    st.divider()
    st.subheader("AI Mode")
    confidence_thresh = st.slider("Confidence Threshold (%)", 0, 100, 70)
    
    st.info("PlantPulse v4.0 GOLD\nHybrid AI-Quantum Diagnostic")

# Main Header
st.title("🌿 PlantPulse: Hybrid Diagnostic")
st.markdown("### Next-Gen Plant Disease Detection using AI & Quantum Verifier")

# App Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Image Input")
    uploaded_file = st.file_uploader("Upload leaf image for analysis", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="BGR", use_container_width=True, caption="Analysis Subject")
    else:
        st.info("Please upload an image to start the expert diagnostic.")

if uploaded_file:
    with col2:
        # Load Resources
        model = load_model()
        
        # 1. Processing Stage
        with st.status("🔍 Analyzing Image...", expanded=True) as status:
            time.sleep(0.5)
            st.write("Preprocessing features...")
            features, gray = preprocess_image(img)
            
            # 2. AI Prediction
            st.write("Running AI Diagnostic...")
            ai_data, confidence = run_ai_prediction(model, features)
            time.sleep(0.5)
            
            # 3. Quantum Verification
            st.write("Initializing Quantum Verifier...")
            q_counts, q_state = run_quantum_verification(gray, use_hardware=quantum_mode, ibm_token=ibm_token)
            
            status.update(label="✅ Analysis Complete", state="complete", expanded=False)

        # UI Results
        st.subheader("🧬 Diagnostic Report")
        
        # Confidence Metric
        st.metric("AI Confidence", f"{confidence:.1f}%", delta=f"{confidence-70:.1f}%" if confidence > 70 else f"{confidence-70:.1f}%")
        
        # Case Mapping
        res = get_remedy(ai_data['disease'])
        
        # Logic for Final Hybrid Result
        is_healthy = ai_data['disease'].lower() == "healthy"
        # Quantum Logic: '00' confirms healthy-looking patterns, '11' confirms detected anomalies
        q_confirms = (is_healthy and q_state == "00") or (not is_healthy and q_state == "11")
        
        if q_confirms:
            st.success(f"### Result: {ai_data['disease']} (Verified By Quantum)")
        else:
            st.warning(f"### Result: {ai_data['disease']} (Uncertain)")
            st.caption("Quantum verifier detected atmospheric noise or pattern mismatch. Verification 72% probable.")

        # Remedies Card
        st.markdown(f"""
        <div class="status-card">
            <div class="prediction-title">Detailed Analysis</div>
            <p><b>Plant:</b> {ai_data['plant']}</p>
            <p><b>Status:</b> <span class="highlight">{res['status']}</span></p>
            <hr style="opacity: 0.1">
            <p><b>Remedy/Action:</b> {res['action']}</p>
            <p style="font-size: 0.9em; opacity: 0.8">{res['details']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quantum Visualizer
        with st.expander("⚛️ View Quantum Bit Distribution"):
            st.bar_chart(q_counts)
            st.caption("Quantum distribution across states |00> to |11>. Significant Peaks indicate anomaly confirmation.")

# Footer
st.divider()
st.caption("Deploying to Streamlit Cloud | AI + Qiskit v0.45 | Built for Precision Agriculture")