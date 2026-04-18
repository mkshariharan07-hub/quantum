import cv2
import numpy as np
import joblib
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer import AerSimulator

# Remedies Database
REMEDIES = {
    "Healthy": {
        "status": "Healthy",
        "action": "Maintain optimal growing conditions.",
        "details": "The plant shows no signs of disease. Continue regular watering and monitoring."
    },
    "Bacterial_spot": {
        "status": "At Risk",
        "action": "Use copper-based fungicides.",
        "details": "Remove affected leaves. Avoid overhead watering as bacteria spread through splashes."
    },
    "Early_blight": {
        "status": "Infected",
        "action": "Improve air circulation and use fungicides.",
        "details": "Caused by fungus. Remove lower leaves to prevent spread from soil splash."
    },
    "Late_blight": {
        "status": "Critical",
        "action": "Destroy infected plants immediately.",
        "details": "Highly contagious. Use protective fungicides and ensure low humidity."
    },
    "Leaf_Mold": {
        "status": "Infected",
        "action": "Reduce humidity and improve airflow.",
        "details": "Common in greenhouses. Keep foliage dry and use calcium-rich fertilizers."
    },
    "Septoria_leaf_spot": {
        "status": "Infected",
        "action": "Remove debris and use fungicides.",
        "details": "Avoid overhead irrigation. Clean garden tools to prevent spread."
    },
    "Spider_mites Two-spotted_spider_mite": {
        "status": "Infected",
        "action": "Use insecticidal soaps or neem oil.",
        "details": "Increase humidity as mites thrive in dry conditions. Introduce natural predators."
    },
    "Target_Spot": {
        "status": "Infected",
        "action": "Apply appropriate fungicides.",
        "details": "Prune lower leaves to improve ventilation."
    },
    "Yellow_Leaf_Curl_Virus": {
        "status": "Critical",
        "action": "Control whitefly population.",
        "details": "Transmitted by whiteflies. Remove infected plants and use reflective mulches."
    },
    "Mosaic_virus": {
        "status": "Critical",
        "action": "Remove infected plants immediately.",
        "details": "Spread by aphids or handling. Wash hands and tools frequently."
    },
    "Powdery_mildew": {
        "status": "Infected",
        "action": "Apply sulfur or potassium bicarbonate sprays.",
        "details": "White powdery spots. Ensure plants have plenty of sun and air."
    },
    "Rust": {
        "status": "Infected",
        "action": "Remove infected leaves and apply sulfur.",
        "details": "Fungal disease. Avoid wetting leaves during watering."
    }
}

@st.cache_resource
def load_model(model_path="plant_model.pkl"):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img, target_size=(128, 128)):
    # Resize and prepare features for AI (RandomForest expects flat array)
    img_resized = cv2.resize(img, target_size)
    features = img_resized.flatten().reshape(1, -1)
    
    # Grayscale version for Quantum Logic
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return features, gray

def run_ai_prediction(model, features):
    if model is None:
        return None, 0
    
    prediction = model.predict(features)[0]
    # Handle both single label and "Plant___Disease" format
    if "___" in prediction:
        plant, disease = prediction.split("___")
    else:
        plant, disease = "Unknown", prediction
        
    confidence = np.max(model.predict_proba(features)) * 100
    return {"plant": plant, "disease": disease}, confidence

def run_quantum_verification(gray_image, use_hardware=False, ibm_token=None):
    # Normalize grayscale to create a quantum input (0 to 1)
    # Using the standard deviation and mean to capture texture
    mean_val = np.mean(gray_image) / 255.0
    std_val = np.std(gray_image) / 255.0
    
    # Simple Plant-Health Quantum Circuit
    # We use 3 qubits to represent a small state space of healthy/unhealthy patterns
    qc = QuantumCircuit(3, 3)
    
    # Amplitude Encoding (Simplified): 
    # Use rotation gates to map image properties to qubit states
    qc.ry(mean_val * np.pi, 0) # Phase based on brightness
    qc.ry(std_val * np.pi, 1)  # Phase based on texture
    
    # Entanglement to check for correlated anomalies
    qc.h(2)
    qc.cx(0, 2)
    qc.cx(1, 2)
    
    # Measurement
    qc.measure([0, 1, 2], [0, 1, 2])
    
    if use_hardware and ibm_token:
        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=ibm_token)
            backend = service.least_busy(simulator=False)
            qc_transpiled = transpile(qc, backend)
            sampler = Sampler(backend)
            job = sampler.run([qc_transpiled], shots=1024)
            result = job.result()
            counts = result[0].data.c.get_counts()
        except Exception as e:
            st.warning(f"Hardware error: {e}. Falling back to simulator.")
            counts = _run_simulator(qc)
    else:
        counts = _run_simulator(qc)
        
    dominant_state = max(counts, key=counts.get)
    return counts, dominant_state

def _run_simulator(qc):
    backend = AerSimulator()
    qc_transpiled = transpile(qc, backend)
    result = backend.run(qc_transpiled, shots=1024).result()
    return result.get_counts()

def get_remedy(disease):
    # Search for partial match in REMEDIES
    for key in REMEDIES:
        if key.lower() in disease.lower():
            return REMEDIES[key]
    return {
        "status": "Unknown",
        "action": "Consult an agricultural expert.",
        "details": f"No specific remedy found for {disease} in our database."
    }
