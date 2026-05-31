import streamlit as st
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manufacturing Defect Detector",
    page_icon="🔬",
    layout="wide"
)

# ── Load model and metadata ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('defect_detection_model.keras')

@st.cache_resource
def load_metadata():
    with open('model_metadata.pkl', 'rb') as f:
        return pickle.load(f)

model    = load_model()
metadata = load_metadata()
CLASS_NAMES = metadata['class_names']   # ['dent', 'normal', 'scratch', 'stain']
IMG_SIZE    = metadata['img_size']       # 64

# ── Class config ───────────────────────────────────────────────────────────────
CLASS_INFO = {
    'normal' : {'emoji': '✅', 'color': '#2E7D32', 'bg': '#E8F5E9', 'action': 'Route to packaging — No defect detected.'},
    'scratch': {'emoji': '⚠️', 'color': '#E65100', 'bg': '#FFF3E0', 'action': 'Route to rework — Linear scratch detected.'},
    'dent'   : {'emoji': '🔴', 'color': '#B71C1C', 'bg': '#FFEBEE', 'action': 'Route to scrap — Dent depression detected.'},
    'stain'  : {'emoji': '🟡', 'color': '#F57F17', 'bg': '#FFFDE7', 'action': 'Route to rework — Surface stain detected.'},
}

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#1a237e;'>🔬 Manufacturing Defect Detector</h1>
    <p style='text-align:center; color:#555; font-size:16px;'>
        CNN · TensorFlow/Keras · Test Accuracy 91.67% · 4 Defect Classes
    </p>
    <hr>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 Upload Product Surface Image")
    st.markdown("Supported formats: **PNG, JPG, JPEG** · Model input: 64×64 RGB")

    uploaded = st.file_uploader(
        "Drop your image here or click to browse",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_array   = np.array(img_resized, dtype=np.float32) / 255.0
        img_input   = np.expand_dims(img_array, axis=0)  # (1, 64, 64, 3)

        # Predict
        with st.spinner("Analysing surface..."):
            preds      = model.predict(img_input, verbose=0)[0]
            pred_idx   = np.argmax(preds)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = preds[pred_idx]

        info = CLASS_INFO[pred_class]

        with col_right:
            st.markdown("### 🧠 Prediction Result")

            # Main result card
            st.markdown(f"""
                <div style='
                    background: {info["bg"]};
                    border-left: 6px solid {info["color"]};
                    border-radius: 12px;
                    padding: 28px 32px;
                    margin-bottom: 20px;
                '>
                    <div style='font-size:48px; margin-bottom:8px;'>{info["emoji"]}</div>
                    <div style='color:{info["color"]}; font-size:32px; font-weight:800; 
                                text-transform:uppercase; letter-spacing:2px;'>
                        {pred_class}
                    </div>
                    <div style='color:#555; font-size:18px; margin-top:6px;'>
                        Confidence: <strong style="color:{info["color"]}">{confidence:.1%}</strong>
                    </div>
                    <div style='color:#444; font-size:15px; margin-top:12px; 
                                background:white; padding:10px 14px; border-radius:8px;'>
                        🏭 <strong>Action:</strong> {info["action"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Confidence bar chart for all classes
            st.markdown("#### 📊 Confidence Across All Classes")
            for i, cls in enumerate(CLASS_NAMES):
                pct   = preds[i]
                color = CLASS_INFO[cls]['color'] if cls == pred_class else '#90A4AE'
                label = f"**{cls}**" if cls == pred_class else cls
                st.markdown(f"{label}")
                st.progress(float(pcts := pct), text=f"{pct:.1%}")

            # Model info
            st.markdown("---")
            st.markdown("""
                <div style='background:#F5F5F5; border-radius:10px; padding:16px;'>
                    <p style='margin:0; font-size:13px; color:#555;'>
                        <strong>Model:</strong> Custom CNN (3 Conv Blocks + Dense Head)<br>
                        <strong>Parameters:</strong> ~2.2M<br>
                        <strong>Test Accuracy:</strong> 91.67% &nbsp;|&nbsp; <strong>Macro F1:</strong> 0.95<br>
                        <strong>Inference:</strong> &lt;50ms per image
                    </p>
                </div>
            """, unsafe_allow_html=True)

    else:
        with col_right:
            st.markdown("### 📋 Defect Classes")
            for cls, info in CLASS_INFO.items():
                st.markdown(f"""
                    <div style='background:{info["bg"]}; border-left:4px solid {info["color"]};
                                border-radius:8px; padding:12px 16px; margin-bottom:10px;'>
                        <strong style='color:{info["color"]};'>{info["emoji"]} {cls.upper()}</strong><br>
                        <span style='color:#555; font-size:13px;'>{info["action"]}</span>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("""
                <br>
                <div style='background:#E3F2FD; border-radius:10px; padding:16px; text-align:center;'>
                    <p style='margin:0; color:#1565C0; font-size:14px;'>
                        ⬆️ Upload a product surface image on the left to get started
                    </p>
                </div>
            """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align:center; color:#aaa; font-size:13px;'>
        Built by <b>Shivam Singh</b> · M.Sc. Data Science & AI, BITS Pilani ·
        <a href='https://github.com/ShivamSingh3406' target='_blank'>GitHub</a>
    </p>
""", unsafe_allow_html=True)
