import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Skinalize", layout="wide", page_icon="🧴")

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/skin_disease_classification_model.h5')

model = load_model()
class_names = ['Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Vitiligo']

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    /* 1. App background */
    [data-testid="stAppViewContainer"] {
        background-color: #eef2f7;
    }
    /* 2. Navbar styling */
    .navbar {
        background: #2b2d42;             /* solid dark slate */
        position: fixed;
        top: 0; left: 0; right: 0;
        padding: 12px 30px;
        display: flex;
        align-items: center;
        gap: 20px;
        z-index: 999;
    }
    .navbar-logo {
        height: 40px;
    }
    .navbar-title {
        color: #edf2f4;
        font-size: 1.4em;
        font-weight: 600;
        flex-grow: 1;
    }
    .navbar-button {
        background-color: #8d99ae20;
        color: #edf2f4;
        border: none;
        padding: 8px 18px;
        border-radius: 6px;
        font-size: 1em;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .navbar-button:hover {
        background-color: #8d99ae40;
        transform: scale(1.05);
    }
    /* 3. Push content below navbar */
    .spacer { height: 72px; }
    </style>
""", unsafe_allow_html=True)

# ─── NAVBAR ───────────────────────────────────────────────────────────────────
def render_navbar():
    st.markdown("""
        <div class="navbar">
            <img src="https://i.imgur.com/your_logo.png" class="navbar-logo" alt="Skinalize Logo">
            <div class="navbar-title">Welcome to Skinalize</div>
            <button class="navbar-button" onclick="window.location.href='?page=home'">🏠 Home</button>
            <button class="navbar-button" onclick="window.location.href='?page=features'">✨ Features</button>
            <button class="navbar-button" onclick="window.location.href='?page=faq'">❓ FAQ</button>
            <button class="navbar-button" onclick="window.location.href='?page=contact'">📬 Contact</button>
        </div>
        <div class="spacer"></div>
    """, unsafe_allow_html=True)

render_navbar()

# ─── ROUTING ───────────────────────────────────────────────────────────────────
page = st.query_params.get("page", "home")

# ─── HOME ──────────────────────────────────────────────────────────────────────
if page == "home":
    st.markdown("<h1 style='text-align:center; color:#3b3b98;'>🧴 Skinalize</h1>", unsafe_allow_html=True)
    st.write("### Upload a skin image and let AI help with diagnosis suggestions.")
    f = st.file_uploader("📷 Upload an Image", type=["jpg","jpeg","png"])
    if f:
        img = Image.open(f).convert("RGB")
        st.image(img, use_column_width=True)
        with st.spinner("Analyzing..."):
            x = np.array(img.resize((224,224)))/255.0
            pred = model.predict(x[np.newaxis,...])[0]
            idx = pred.argmax(); conf = pred[idx]
        st.success(f"✅ **{class_names[idx]}** — {conf*100:.2f}%")
        st.bar_chart(pred)

# ─── FEATURES ─────────────────────────────────────────────────────────────────
elif page == "features":
    st.markdown("## ✨ Features")
    st.write("""
    - 🤖 **AI‑Powered Diagnosis**  
    - 📷 **One‑click Image Upload**  
    - ⚡ **Real‑time Results**  
    - 🔒 **Privacy‑first: No images stored**  
    - 📱 **Mobile & Desktop Friendly**  
    """)

# ─── FAQ ───────────────────────────────────────────────────────────────────────
elif page == "faq":
    st.markdown("## ❓ Frequently Asked Questions")
    with st.expander("What conditions can it detect?"):
        st.write("Acne, Eczema, Psoriasis, Rosacea, Vitiligo.")
    with st.expander("Is it a medical diagnosis?"):
        st.write("No—always consult a professional.")
    with st.expander("Is it free?"):
        st.write("Yes, totally free and anonymous.")

# ─── CONTACT ───────────────────────────────────────────────────────────────────
elif page == "contact":
    st.markdown("## 📬 Contact Us")
    st.write("📧 skinalize.help@gmail.com  |  🧑‍💻 Shivani")

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center;'>Made with ❤️ using Streamlit & TensorFlow</div>", unsafe_allow_html=True)
