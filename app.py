# import streamlit as st
# import re
# import os
# import gdown
# import pickle

# # Google Drive direct download links
# model_url = 'https://drive.google.com/uc?id=1_dFRVVt6RQyNNCtcMKOiWfqIjEo1rwal'
# vectorizer_url = 'https://drive.google.com/uc?id=1vvd_j6v-TNhoHhzyNYP9aY-NG_V6jdEi'

# # Filenames
# model_file = 'spam_model.pkl'
# vectorizer_file = 'vectorizer.pkl'

# # Download if not already present
# if not os.path.exists(model_file):
#     gdown.download(model_url, model_file, quiet=False)
# if not os.path.exists(vectorizer_file):
#     gdown.download(vectorizer_url, vectorizer_file, quiet=False)

# # Load model and vectorizer
# with open(model_file, 'rb') as f:
#     model = pickle.load(f)
# with open(vectorizer_file, 'rb') as f:
#     vectorizer = pickle.load(f)

# # Simple text preprocessing (no nltk)
# def transform_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     text = text.split()
#     return ' '.join(text)

# # --- Sidebar ---
# st.sidebar.title("📘 About This App")
# st.sidebar.markdown("""
# This mini-project uses a **Machine Learning model** to detect whether a message is **Spam** or **Ham (Not Spam)**.  
# **Technologies used**:  
# - Streamlit  
# - Scikit-learn  
# - TF-IDF Vectorizer  
# - Naive Bayes Classifier  

# 👨‍💻 Created by: *Your Name Here*
# """, unsafe_allow_html=True)

# st.sidebar.markdown("---")
# st.sidebar.subheader("💬 Try Sample Messages:")
# sample_msgs = {
#     "Congratulations! You've won a $1000 Walmart gift card.": "Spam",
#     "Your meeting is scheduled at 3 PM today.": "Ham",
#     "Limited offer! Buy now and get 50% off.": "Spam"
# }
# for msg, label in sample_msgs.items():
#     st.sidebar.write(f"**{label}:** \"{msg}\"")

# # --- Main Page ---
# st.markdown("""
#     <h1 style='text-align: center; color: #4A90E2;'>📩 Spam Message Detector</h1>
#     <p style='text-align: center; font-size: 18px;'>Paste a message below to check if it's <b>Spam</b> or <b>Ham</b>.</p>
#     <hr style="border:1px solid #f0f0f0;">
# """, unsafe_allow_html=True)

# # Input box
# input_sms = st.text_area("🖊️ Enter your message here:", height=150, max_chars=500, key="input_sms")

# # Predict button with style
# button_style = """
#     <style>
#     .css-1emrehy.edgvbvh3 {
#         background-color: #4CAF50;
#         color: white;
#         font-weight: bold;
#         font-size: 16px;
#         padding: 10px;
#         border-radius: 5px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
#     </style>
# """
# st.markdown(button_style, unsafe_allow_html=True)

# if st.button("🔍 Predict"):
#     if input_sms.strip() == "":
#         st.warning("⚠️ Please enter some text before predicting.")
#     else:
#         transformed_sms = transform_text(input_sms)
#         vector_input = vectorizer.transform([transformed_sms])
#         prediction = model.predict(vector_input)[0]

#         st.markdown("---")
#         if prediction == "spam":
#             st.error("🚫 **This message is SPAM** – Looks suspicious or promotional.", icon="🚫")
#         else:
#             st.success("✅ **This message is HAM** – It seems safe and legitimate.", icon="✅")

#         # Display processed text for user review
#         st.markdown("### 🔁 Processed Text:")
#         st.code(transformed_sms)

# # Footer with custom style
# st.markdown("""
#     <hr style="border:1px solid #f0f0f0;">
#     <p style='text-align: center; font-size: 14px; color: #4A90E2;'>Built with ❤️ using Streamlit & Scikit-learn</p>
# """, unsafe_allow_html=True)

import streamlit as st
from datetime import datetime

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RailSense Lite — Crowd & Seat Predictor",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL STYLES ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0d12 !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8eaf0;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1200px; }
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem; font-weight: 800;
    letter-spacing: -1px; line-height: 1.1;
    background: linear-gradient(135deg, #e8eaf0 30%, #5ce1e6 70%, #7b61ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.25rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace; font-size: 0.78rem;
    color: #5ce1e6; letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 2rem;
}
.section-head {
    font-family: 'Syne', sans-serif; font-size: 0.68rem;
    font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: #5a6480;
    margin: 1.6rem 0 0.7rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2330;
}
.rail-card {
    background: #0d1117; border: 1px solid #1e2330;
    border-radius: 14px; padding: 1.5rem;
    margin-bottom: 1rem; transition: border-color 0.2s;
}
.rail-card:hover { border-color: #2a3450; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 100px; font-family: 'DM Mono', monospace; font-size: 0.62rem; font-weight: 500; letter-spacing: 1.5px; text-transform: uppercase; }
.badge-green  { background: #0d2e1f; color: #4ade80; border: 1px solid #1a5e3a; }
.badge-yellow { background: #2e2000; color: #f9a825; border: 1px solid #5e4000; }
.badge-red    { background: #2e0d0d; color: #ff6f61; border: 1px solid #5e1a1a; }
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background: #0d1117 !important; border: 1px solid #1e2330 !important;
    border-radius: 8px !important; color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button {
    background: linear-gradient(135deg, #5ce1e6, #7b61ff) !important;
    color: #0a0d12 !important; border: none !important;
    border-radius: 8px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    padding: 0.55rem 1.5rem !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
label { color: #8896b3 !important; font-size: 0.83rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem;">🚆 RailSense Lite</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Rule‑Based Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-head">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="rail-card" style="font-size:0.85rem;color:#8896b3;line-height:2;">
        This version uses <b style="color:#e8eaf0;">no ML model</b>.<br>
        Crowds and seat availability are estimated using simple <b style="color:#5ce1e6;">if‑elif‑else rules</b> on your input.<br>
        You can submit this as a concept project.
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN HEADER ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">RailSense Lite</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Crowd & Seat Intelligence (Rule‑Based)</div>', unsafe_allow_html=True)

# ─── INPUT FORM ──────────────────────────────────────────────────────────────
col_form, col_result = st.columns([1.1, 1], gap="large")

with col_form:
    st.markdown('<div class="section-head">Journey Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        journey_type = st.selectbox("Journey Type", ["Express", "Superfast", "Local", "Mail", "Intercity", "Rajdhani"])
    with c2:
        distance = st.number_input("Distance (km)", 1.0, 5000.0, 450.0, 10.0)

    c3, c4 = st.columns(2)
    with c3:
        month = st.selectbox("Month", list(range(1, 13)),
                             format_func=lambda m: datetime(2024, m, 1).strftime("%B"),
                             index=5)
    with c4:
        holiday_type = st.selectbox("Holiday Type", [0,1,2], format_func=lambda x: {0:"None",1:"Minor",2:"Major"}[x])

    is_weekend = st.checkbox("Weekend Journey", value=False)

    st.markdown('<div class="section-head">General Coach Seats</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        sl_capacity = st.number_input("SL Capacity", 0, 2000, 600, 50)
        sl_booked   = st.number_input("SL Booked", 0, 2000, 420, 10)
    with c6:
        ac3_capacity = st.number_input("AC3 Capacity", 0, 1000, 300, 25)
        ac3_booked   = st.number_input("AC3 Booked", 0, 1000, 210, 10)

    c7, c8 = st.columns(2)
    with c7:
        ac2_capacity = st.number_input("AC2 Capacity", 0, 500, 100, 10)
        ac2_booked   = st.number_input("AC2 Booked", 0, 500, 80, 5)
    with c8:
        ac1_capacity = st.number_input("AC1 Capacity", 0, 200, 24, 4)
        ac1_booked   = st.number_input("AC1 Booked", 0, 200, 18, 1)

    predict_btn = st.button("⚡ Run Prediction", use_container_width=True)

# ─── RULE-BASED PREDICTION ───────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-head">Prediction Result</div>', unsafe_allow_html=True)
    
    if not predict_btn:
        st.markdown("""
        <div class="rail-card" style="border-style:dashed;text-align:center;padding:3rem 1.5rem;border-color:#1e2330;">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">⚡</div>
            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#3a4560;margin-bottom:0.4rem;">Ready to predict</div>
            <div style="font-size:0.8rem;color:#2a3450;">Fill the form and click Run Prediction</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Compute total capacity and booked
        total_cap    = sl_capacity + ac3_capacity + ac2_capacity + ac1_capacity
        total_booked = sl_booked + ac3_booked + ac2_booked + ac1_booked
        occ_ratio = (total_booked / total_cap) if total_cap > 0 else 0
        occ_pct = occ_ratio * 100

        # ── Crowd level
        if occ_ratio <= 0.5:
            crowd_level = "low"
        elif occ_ratio <= 0.8:
            crowd_level = "medium"
        else:
            crowd_level = "high"

        # ── Seat availability (SL + AC3)
        gen_seats = sl_capacity + ac3_capacity
        gen_booked = sl_booked + ac3_booked
        gen_ratio = (gen_booked / gen_seats) if gen_seats > 0 else 0
        if gen_ratio <= 0.6:
            seat_avail = "available"
        elif gen_ratio <= 0.85:
            seat_avail = "limited"
        else:
            seat_avail = "unavailable"

        # ── Confidence (UI only)
        crowd_conf = 0.75 + 0.25 * min(occ_ratio, 1.0)
        seat_conf  = 0.70 + 0.30 * (1.0 - gen_ratio)

        # ── Badges
        badge_crowd = lambda x: f'<span class="badge badge-{"green" if x=="low" else "yellow" if x=="medium" else "red"}">{x.title()}</span>'
        badge_seat  = lambda x: f'<span class="badge badge-{"green" if x=="available" else "yellow" if x=="limited" else "red"}">{x.title()}</span>'

        # ── Show result cards
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.7rem;margin-bottom:1.2rem;">
            <div class="rail-card" style="padding:1rem;text-align:center;">
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#5a6480;letter-spacing:2px;text-transform:uppercase;">Occupancy</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:#5ce1e6;">{occ_pct:.0f}%</div>
            </div>
            <div class="rail-card" style="padding:1rem;text-align:center;">
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#5a6480;letter-spacing:2px;text-transform:uppercase;">Crowd</div>
                <div style="margin-top:4px;">{badge_crowd(crowd_level)}</div>
            </div>
            <div class="rail-card" style="padding:1rem;text-align:center;">
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#5a6480;letter-spacing:2px;text-transform:uppercase;">Seat Availability</div>
                <div style="margin-top:4px;">{badge_seat(seat_avail)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
