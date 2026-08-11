"""
Gender Classification from Eye Images — Streamlit Demo (public deploy)
Model: Vision Transformer (ViT-Base/16), fine-tuned (checkpoint D0)
Model weights: loaded from Hugging Face Hub (free model repo)
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from transformers import ViTForImageClassification
from PIL import Image
import numpy as np
import cv2
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Gender Classification - Eye Images",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

MODEL_REPO = "Hanura22/vit-eye-gender-classification"  # weights on HF Hub (free)
NUM_CLASSES = 2
CLASS_NAMES = ["Female", "Male"]  # class 0 = Female, class 1 = Male (per README)
IMAGE_SIZE = 224

# Eye detector (Haar cascade) for auto-crop
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# ══════════════════════════════════════════════════════════════
# AUTO EYE CROP (same pipeline as the Gradio research app)
# ══════════════════════════════════════════════════════════════

def auto_crop_eye(image):
    """Detect and crop the eye region automatically. Returns (image, cropped_bool)."""
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(eyes) == 0:
        return image, False

    # Largest detected eye
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)
    x, y, w, h = eyes[0]
    pad = 20
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)

    return Image.fromarray(img[y1:y2, x1:x2]), True


# ══════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ══════════════════════════════════════════════════════════════
# MODEL LOADING (from HF Hub, cached across sessions)
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(MODEL_REPO)
    model.to(device)
    model.eval()
    return model, device


# ══════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════

def predict_gender(image, model, device, transform):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    cropped, was_cropped = auto_crop_eye(image)
    input_tensor = transform(cropped).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

    pred_class = int(np.argmax(probs))
    return pred_class, float(probs[pred_class]), probs, cropped, was_cropped


# ══════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════

def create_confidence_chart(probs, class_names):
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probs * 100,
            text=[f"{p:.2f}%" for p in probs * 100],
            textposition="auto",
            marker=dict(
                color=["#FF6B6B" if i == 0 else "#4ECDC4" for i in range(len(probs))],
                line=dict(color="white", width=2),
            ),
        )
    ])
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Gender",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False,
        template="plotly_white",
    )
    return fig


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════

def main():
    st.markdown('<h1 class="main-header">👁️ Gender Classification from Eye Images</h1>', unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; margin-bottom: 2rem;'>"
        "Upload an eye image and let AI predict the gender based on periocular features. "
        "The system auto-detects and crops the eye region.</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model... (first load downloads ~340MB from Hugging Face Hub)"):
        model, device = load_model()
        transform = get_transform()

    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.markdown(
            f"""
            **Architecture:** Vision Transformer (ViT-Base/16)  
            **Preprocessing:** Grayscale + auto eye-crop  
            **Classes:** Female (0), Male (1)  
            **Device:** {device}  
            **Checkpoint:** D0
            """
        )
        st.header("📊 Model Stats")
        st.metric("Test Accuracy", "95.7%")
        st.metric("Trainable Params", "42M")
        st.metric("Layers Unfrozen", "6")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an eye image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of an eye region",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Predict Gender", type="primary"):
                with st.spinner("Analyzing image..."):
                    pred_class, confidence, probs, cropped, was_cropped = predict_gender(
                        image, model, device, transform
                    )
                    st.session_state.prediction = {
                        "class": pred_class,
                        "confidence": confidence,
                        "probs": probs,
                        "cropped": cropped,
                        "was_cropped": was_cropped,
                    }

    with col2:
        st.subheader("🎯 Prediction Results")
        if "prediction" in st.session_state:
            pred = st.session_state.prediction
            gender = CLASS_NAMES[pred["class"]]
            conf = pred["confidence"]
            color = "#FF6B6B" if pred["class"] == 0 else "#4ECDC4"

            st.markdown(
                f"""
                <div class="prediction-box" style="border-left: 5px solid {color};">
                    <h2 style="margin: 0; color: {color};">{gender}</h2>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                        Confidence: <strong>{conf:.2%}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if pred["was_cropped"]:
                st.image(pred["cropped"], caption="Detected eye region (auto-cropped)", use_container_width=True)
            else:
                st.warning("No eye detected — prediction used the full image.")

            fig = create_confidence_chart(pred["probs"], CLASS_NAMES)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Upload an image and click 'Predict Gender' to see results")

    st.markdown("---")
    st.markdown(
        """
        ### 📚 About This Research
        **Thesis:** Klasifikasi Jenis Kelamin Berbasis Citra Mata Manusia dengan Pendekatan Vision Transformer (ViT)  
        **Objective:** Develop an interpretable AI model for gender classification from periocular images  
        **Key Innovation:** Bias-free preprocessing to ensure the model learns from anatomical features, not artifacts

        **Note:** Research prototype for educational/demonstration purposes.
        """
    )


if __name__ == "__main__":
    main()
