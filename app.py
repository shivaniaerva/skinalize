import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skinalize",
    layout="wide",
    page_icon="🧴",
    initial_sidebar_state="collapsed"
)

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/skin_disease_classification_model.h5')

model = load_model()
class_names = ['Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Vitiligo']

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4f0fb 50%, #d5e8ff 100%);
    }
    .main-container {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        margin: 1rem auto;
        max-width: 1200px;
    }
    .navbar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        position: sticky;
        top: 0;
        padding: 12px 30px;
        display: flex;
        align-items: center;
        gap: 20px;
        z-index: 999;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    .navbar-title {
        color: #ffffff;
        font-size: 1.4em;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .nav-links {
        display: flex;
        gap: 15px;
        margin-left: auto;
    }
    .nav-link {
        color: white !important;
        text-decoration: none !important;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .nav-link:hover {
        background-color: rgba(255, 255, 255, 0.3);
        transform: scale(1.05);
    }
    .spacer { height: 72px; }
    .upload-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(240,248,255,0.9) 100%);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        border: 1px dashed #667eea;
    }
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .stFileUploader > div > div {
        border: 2px dashed #667eea !important;
        background-color: rgba(102, 126, 234, 0.05) !important;
        border-radius: 15px !important;
    }
    h1, h2, h3 {
        color: #4a4a4a;
    }
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ─── NAVBAR ───────────────────────────────────────────────────────────────────
def render_navbar():
    st.markdown("""
        <div class="navbar">
            <div class="navbar-title">Skinalize</div>
            <div class="nav-links">
                <a href="?page=home" class="nav-link">🏠 Home</a>
                <a href="?page=features" class="nav-link">✨ Features</a>
                <a href="?page=faq" class="nav-link">❓ FAQ</a>
                <a href="?page=contact" class="nav-link">📬 Contact</a>
            </div>
        </div>
        <div class="spacer"></div>
    """, unsafe_allow_html=True)

# ─── PAGE FUNCTIONS ───────────────────────────────────────────────────────────
def show_home():
    st.markdown("<h1>🧴 Skinalize AI Dermatology Assistant</h1>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="upload-container">
            <h3 style="color: #667eea; text-align: center;">📷 Upload a skin image for analysis</h3>
            <p style="text-align: center; color: #666;">Our AI will analyze your skin condition and provide suggestions</p>
        </div>
        """, unsafe_allow_html=True)
        
        f = st.file_uploader("Drag and drop file here or click to browse", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if f:
            col1, col2 = st.columns([1, 2])
            with col1:
                img = Image.open(f).convert("RGB")
                st.image(img, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                with st.spinner("🔍 Analyzing your skin image..."):
                    x = np.array(img.resize((224, 224))) / 255.0
                    pred = model.predict(x[np.newaxis, ...])[0]
                    idx = pred.argmax()
                    conf = pred[idx]
                
                st.markdown(f"""
                <div class="result-card">
                    <h3 style="color: #667eea;">Analysis Results</h3>
                    <p style="font-size: 1.2em;">Most likely condition: <strong style="color: #764ba2;">{class_names[idx]}</strong></p>
                    <p style="font-size: 1.2em;">Confidence level: <strong style="color: #764ba2;">{conf * 100:.2f}%</strong></p>
                    <p style="color: #666; font-size: 0.9em;">Remember: This is not a medical diagnosis. Please consult a dermatologist for professional advice.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Prepare data for chart
                df = pd.DataFrame({
                    'Class': class_names,
                    'Confidence': pred
                }).sort_values('Confidence', ascending=False)
                
                # Confidence chart
                chart = alt.Chart(df).mark_bar(
                    cornerRadiusTopLeft=5,
                    cornerRadiusTopRight=5
                ).encode(
                    x=alt.X('Class:N', title="Skin Condition", sort='-y'),
                    y=alt.Y('Confidence:Q', title="Confidence Level", axis=alt.Axis(format='%')),
                    color=alt.Color('Confidence:Q', scale=alt.Scale(scheme='purpleblue'), legend=None),
                    tooltip=['Class', alt.Tooltip('Confidence:Q', format='.2%')]
                ).properties(
                    width=600,
                    height=400,
                    title="Confidence Distribution Across Conditions"
                )
                
                st.altair_chart(chart, use_container_width=True)

def show_features():
    st.markdown("## ✨ Key Features")
    
    features = [
        {"icon": "🤖", "title": "AI-Powered Analysis", "desc": "Advanced deep learning model for accurate skin condition assessments"},
        {"icon": "⚡", "title": "Instant Results", "desc": "Get analysis in seconds with our optimized processing"},
        {"icon": "🔒", "title": "Privacy Focused", "desc": "Your images are processed securely and never stored"},
        {"icon": "📱", "title": "Mobile Friendly", "desc": "Works perfectly on all devices from smartphones to desktops"},
        {"icon": "💡", "title": "Educational Insights", "desc": "Learn about different skin conditions and characteristics"},
        {"icon": "🌐", "title": "Global Access", "desc": "Available worldwide with multilingual support coming soon"}
    ]
    
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="feature-card">
                <h3 style="color: #667eea;">{feature['icon']} {feature['title']}</h3>
                <p>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def show_faq():
    st.markdown("## ❓ Frequently Asked Questions")
    
    faqs = [
        {
            "question": "What skin conditions can Skinalize detect?",
            "answer": "Skinalize can identify several common skin conditions including Acne, Eczema, Psoriasis, Rosacea, and Vitiligo."
        },
        {
            "question": "Is Skinalize's analysis a medical diagnosis?",
            "answer": "No, Skinalize provides informational suggestions only. Always consult a dermatologist for professional diagnosis."
        },
        {
            "question": "How accurate is Skinalize?",
            "answer": "Our AI model achieves high accuracy on test datasets, but real-world performance may vary depending on image quality."
        },
        {
            "question": "Is my image data stored or shared?",
            "answer": "No, we prioritize your privacy. All image processing happens in memory and images are never stored."
        }
    ]
    
    for faq in faqs:
        with st.expander(f"❔ {faq['question']}", expanded=False):
            st.write(faq['answer'])

def show_contact():
    st.markdown("## 📬 Contact Us")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; border-radius: 15px; padding: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
            <h3 style="color: #667eea;">Get in Touch</h3>
            <p>📧 <strong>Email:</strong> skinalize.help@gmail.com</p>
            <p>🧑‍💻 <strong>Developer:</strong> Team Skinalize</p>
        </div>
        """, unsafe_allow_html=True)
    
    

# ─── MAIN APP ────────────────────────────────────────────────────────────────
# ─── MAIN APP ────────────────────────────────────────────────────────────────
def main():
    # Get current page from query params
    query_params = st.query_params
    page = query_params.get("page", "home")
    
    # Clear any previous content
    st.empty()
    
    # Render navbar (must be first component)
    render_navbar()
    
    # Show the appropriate page
    if page == "home":
        show_home()
    elif page == "features":
        show_features()
    elif page == "faq":
        show_faq()
    elif page == "contact":
        show_contact()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; color: #666;">
        <p>Made with ❤️ using Streamlit & TensorFlow | © 2023 Skinalize</p>
        <div style="margin-top: 0.5rem;">
            <a href="/?page=privacy" style="margin: 0 10px; color: #667eea; text-decoration: none;">Privacy Policy</a>
            <a href="/?page=terms" style="margin: 0 10px; color: #667eea; text-decoration: none;">Terms of Service</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
