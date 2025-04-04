import streamlit as st
import numpy as np
import pickle

# Load the model, scaler, and label encoder
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Set custom CSS style
st.markdown("""
    <style>
    body {
        background-color: blue;
    }
    .main {
        background-color: skyblue;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #2e7d32;
        text-align: center;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .success-message {
        color: #0d47a1;
        font-weight: bold;
        font-size: 24px;
        text-align: center;
    }
            
            
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("🌾 Crop Recommendation System")

st.markdown("""
    Enter the soil and weather conditions to get the best crop recommendation for your land.
""")
st.markdown("""
    <style>
    /* Change the color of input field labels */
    div[data-testid="stNumberInput"] label {
        color: brown !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Input fields
N = st.number_input("Enter Nitrogen content (N)", 0, 140, 50)
P = st.number_input("Enter Phosphorous content (P)", 5, 145, 50)
k = st.number_input("Enter Potassium content (K)", 5, 205, 50)
temperature = st.number_input("Enter Temperature (°C)", 10.0, 50.0, 25.0)
humidity = st.number_input("Enter Humidity (%)", 10.0, 100.0, 65.0)
ph = st.number_input("Enter pH value", 3.5, 9.5, 6.5)
rainfall = st.number_input("Enter Rainfall (mm)", 20.0, 300.0, 100.0)

if st.button("Predict Crop"):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    transformed = scaler.transform(features)
    prediction = model.predict(transformed)
    crop_name = le.inverse_transform(prediction)[0]
    st.markdown(f"<div class='success-message'>✅ Recommended Crop: <strong>{crop_name.upper()}</strong></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
