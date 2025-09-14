import streamlit as st
import pickle
from pathlib import Path

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Smart Spam Detector",
    page_icon="✉️",
    layout="centered",
    initial_sidebar_state="auto"
)

# ---------------------------
# Load Saved Artifacts
# ---------------------------
# Use a consistent path for assets
BASE_DIR = Path(__file__).resolve().parent
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
MODEL_PATH = BASE_DIR / "model.pkl"
BANNER_PATH = BASE_DIR / "assets" / "banner.jpg" # Path to your local banner
HAM_ICON_PATH = "https://cdn-icons-png.flaticon.com/512/190/190411.png" # Using stable CDN link
SPAM_ICON_PATH = "https://cdn-icons-png.flaticon.com/512/595/595067.png" # Using stable CDN link

@st.cache_resource(show_spinner="Loading model assets...")
def load_artifacts():
    """Loads the ML model and vectorizer from disk."""
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        st.error(
            "Vectorizer or model not found. Make sure 'vectorizer.pkl' and 'model.pkl' are in the root folder."
        )
        st.stop()
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
    model = pickle.load(open(MODEL_PATH, "rb"))
    return vectorizer, model

vectorizer, model = load_artifacts()

# ---------------------------
# Theme-Aware Styling
# ---------------------------
st.markdown("""
<style>
    /* General body styling */
    .stApp {
        background-color: var(--background-color);
    }
    /* Custom classes for titles */
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--header-text-color);
        padding-bottom: 0.5rem;
    }
    .subtitle {
        color: var(--text-color);
        margin-top: -0.5rem;
        padding-bottom: 1rem;
    }
    /* Result card styling */
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        display: flex;
        align-items: center;
        border-width: 1px;
        border-style: solid;
    }
    .result-card img {
        width: 50px;
        margin-right: 15px;
    }
    .result-text h3 {
        margin: 0;
        font-size: 1.25rem;
    }
    .result-text p {
        margin: 0.25rem 0 0 0;
        color: var(--text-color);
    }
    /* Specific styles for spam/ham cards */
    .scam-card {
        background-color: #fff7f5;
        border-color: #ffd6d0;
    }
    .legit-card {
        background-color: #f6ffed;
        border-color: #d1f7c4;
    }
    /* Tips card styling */
    .tips-card {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--gray-300);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helper Functions
# ---------------------------
LABEL_MAP = {0: "Ham (Not Scam)", 1: "Spam / Scam"}

def predict(text: str):
    """Vectorizes text and predicts using the loaded model."""
    if not text.strip():
        return None, None, None
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    try:
        proba = model.predict_proba(X)[0]
        confidence = proba[prediction]
    except AttributeError: # Some models don't have predict_proba
        confidence = None
    return int(prediction), LABEL_MAP[int(prediction)], confidence

def display_result(prediction, confidence):
    """Displays the prediction result in a styled card."""
    if prediction == 1: # Spam
        st.markdown(f"""
        <div class='result-card scam-card'>
            <img src='{SPAM_ICON_PATH}'>
            <div class='result-text'>
                <h3>🚨 This looks like a scam!</h3>
                <p>Be cautious and do not click any links or provide personal info.</p>
                {"<b>Confidence:</b> {:.2f}%".format(confidence * 100) if confidence else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else: # Ham
        st.markdown(f"""
        <div class='result-card legit-card'>
            <img src='{HAM_ICON_PATH}'>
            <div class='result-text'>
                <h3>✅ This appears to be a legitimate message.</h3>
                {"<b>Confidence:</b> {:.2f}%".format(confidence * 100) if confidence else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# --- Main Application UI ---
# ---------------------------

# --- Header ---
if BANNER_PATH.exists():
    st.image(str(BANNER_PATH), use_container_width=True)
st.markdown('<div class="title">Smart Spam Detector ✉️</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste an email or SMS below to check if it is a scam. Stay alert! ⚠️</div>', unsafe_allow_html=True)

# --- Input and Prediction ---
email_text = st.text_area(
    "Enter the email / SMS text here:",
    height=200,
    placeholder="Type or paste your message..."
)

if st.button("Check for Scam", type="primary"):
    with st.spinner("Analyzing..."):
        pred_idx, _, conf = predict(email_text)

    if pred_idx is None:
        st.warning("Please enter a message to analyze.", icon="✍️")
    else:
        display_result(pred_idx, conf)

# --- Awareness Tips ---
st.markdown("""
<div class='tips-card'>
    💡 <b>Tips to Avoid Scams:</b>
    <ul>
        <li>Never click on suspicious links or download unknown attachments.</li>
        <li>Verify the sender's identity through a trusted channel before sharing sensitive info.</li>
        <li>Use strong, unique passwords and enable Two-Factor Authentication (2FA).</li>
        <li>Report suspicious messages to your email or messaging provider.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("Built with Streamlit. Always verify suspicious messages through trusted channels.")
st.markdown("<div style='text-align:center; color:var(--gray-600); padding-top:8px;'>Made with ❤ — Spam Detector App</div>", unsafe_allow_html=True)