import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Page Config
st.set_page_config(page_title="VinoAnalytics Luxury", page_icon="🍷", layout="centered")

# 2. Professional Luxury CSS
# Using a high-resolution wine cellar background from Unsplash
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(30, 0, 0, 0.85), rgba(30, 0, 0, 0.85)), 
                    url("https://images.unsplash.com/photo-1510850402484-49521f1e2a97?q=80&w=2070");
        background-size: cover;
        background-position: center;
        color: #f1faee;
    }

    /* Glassmorphism Input Containers */
    div[data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Luxury Gold Button */
    div.stButton > button {
        background: linear-gradient(45deg, #d4af37, #f9e27d);
        color: #1a0a0a !important;
        border: none;
        font-weight: bold;
        padding: 15px 30px;
        width: 100%;
        border-radius: 5px;
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        box-shadow: 0px 0px 20px rgba(212, 175, 55, 0.6);
        transform: scale(1.02);
    }

    /* Result Metric Styling */
    .result-box {
        background: rgba(255, 255, 255, 0.1);
        border-left: 10px solid #d4af37;
        padding: 25px;
        border-radius: 5px;
        text-align: center;
        margin-top: 30px;
    }

    h1, h2, h3 {
        color: #f1faee !important;
        font-family: 'Serif';
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Assets
try:
    with open('wine_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('wine_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Diagnostic Assets Missing. Please run train.py.")
    st.stop()

# 4. Header Section
st.markdown("<h1 style='text-align: center;'>CHÂTEAU ANALYTICS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic; opacity: 0.8;'>Predictive Enology & Wine Quality Assessment</p>", unsafe_allow_html=True)
st.write("---")

# 5. Grouped Inputs for Better UX
st.subheader("🍷 Vintage Chemical Profile")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity (g/dm³)", 4.0, 16.0, 8.3)
    volatile_acidity = st.number_input("Volatile Acidity (g/dm³)", 0.1, 1.6, 0.5)
    citric_acid = st.number_input("Citric Acid (g/dm³)", 0.0, 1.0, 0.27)
    res_sugar = st.number_input("Residual Sugar (g/dm³)", 0.9, 15.5, 2.5)
    chlorides = st.number_input("Chlorides (g/dm³)", 0.01, 0.6, 0.08)

with col2:
    f_sulfur = st.number_input("Free Sulfur Dioxide", 1, 72, 15)
    t_sulfur = st.number_input("Total Sulfur Dioxide", 6, 289, 46)
    density = st.number_input("Density (g/cm³)", 0.9900, 1.0040, 0.9960, format="%.4f")
    ph = st.number_input("pH Level", 2.7, 4.0, 3.3)
    sulphates = st.number_input("Sulphates (g/dm³)", 0.3, 2.0, 0.65)
    alcohol = st.number_input("Alcohol Volume %", 8.0, 15.0, 10.5)

# 6. Predict Button
st.write("")
if st.button("RUN VINTAGE ASSESSMENT"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, res_sugar, 
                          chlorides, f_sulfur, t_sulfur, density, ph, sulphates, alcohol]])
    
    # Preprocessing
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    # 7. Attractive Results
    st.markdown(f"""
        <div class="result-box">
            <h2 style='margin:0; color: #d4af37;'>Quality Rating Score</h2>
            <h1 style='font-size: 60px; margin: 10px 0;'>{prediction:.2f} <span style='font-size: 20px;'>/ 10</span></h1>
            <p style='font-size: 1.1rem; opacity: 0.8;'>Validated via 5-Fold Cross Validation</p>
        </div>
    """, unsafe_allow_html=True)
    
    if prediction >= 7:
        st.balloons()
        st.success("Congratulations! This vintage is classified as **Premium Quality**.")

# Footer
st.markdown("<br><p style='text-align: center; opacity: 0.5;'>© 2026 Global Enology Solutions | Data Driven Viticulture</p>", unsafe_allow_html=True)