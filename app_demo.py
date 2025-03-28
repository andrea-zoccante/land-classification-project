import streamlit as st
from PIL import Image
import os
from customCLIP import customCLIP
from config import CLASSES, MOD_CLASSES

# -------------------- CONFIG --------------------
st.set_page_config(page_title="EuroSAT CLIP Classifier", layout="centered")
st.title("🛰️ EuroSAT CLIP Image Classifier")
st.markdown("""
Upload a satellite image and classify it into one of the 10 EuroSAT land classes using different trained CLIP models.
""")
st.markdown(
    """
    <style>
        div.stButton > button {
            border-radius: 8px;
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            transition: 0.3s;
            border: none;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        }
        div.stButton > button:hover {
            background-color: #45a049;
            color: white;
            transform: scale(1.05);
            box-shadow: 3px 3px 8px rgba(0,0,0,0.3);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- PLACEHOLDERS --------------------
model_options = {
    "Zero-Shot CLIP": "zeroshot",
    "Linear Probe": "linear_probe",
    "MLP Probe": "mlp_probe",
    "Logistic Regression": "logreg_probe",
    "CoOp Prompt Learner": "coop"
}

class_names = CLASSES
modified_names = MOD_CLASSES

if "clip_model" not in st.session_state:
    st.session_state.clip_model = customCLIP(
        model_name="openai/clip-vit-base-patch32"
    )
clip_model = st.session_state.clip_model


# -------------------- SIDEBAR --------------------
st.sidebar.header("🔧 Settings")

selected_model_name = st.sidebar.selectbox("CLIP Model Type", list(model_options.keys()))
selected_mode = model_options[selected_model_name]

# Conditional: Prompt (only for non-CoOp)
prompt_prefix = "A satellite image of"
use_modified = True
if selected_mode == "zeroshot":
    prompt_prefix = st.sidebar.text_input("Prompt Template", value="A satellite image of")

    # Label style
    label_type = st.sidebar.selectbox("Label Style", ["Class Names", "Modified Names"])
    use_modified = label_type == "Modified Names"

# Conditional: Few-shot models (only for non-zeroshot)
few_shot = None
if selected_mode != "zeroshot":
    few_shot = st.sidebar.selectbox("Few-Shot Setting", [16, 8, 4, 2, 1], index=1)

    loading_placeholder = st.sidebar.empty()

    if st.sidebar.button("✅ Load Model"):
        loading_placeholder.text("⏳ Loading...")
        # Init model
        clip_model.set_modify(use_modified)
        clip_model.set_prompt_template(prompt_prefix)

        # Fix for ValueError: map probes to 'probe' mode
        testing_mode = "probe" if selected_mode in ["linear_probe", "mlp_probe", "logreg_probe"] else selected_mode
        clip_model.set_testing_mode(testing_mode)

        # Load few-shot model (if applicable)
        if selected_mode == "coop":
            model_path = f"models/prompt_learners/coop/{few_shot}_shot.pth"
        else:
            ext = "pkl" if selected_mode == "logreg_probe" else "pth"
            model_path = f"models/classifiers/{selected_mode}/{few_shot}-shot.{ext}"
        clip_model.load_model(model_path, mode=selected_mode)
        loading_placeholder.text("")  # Clear the message once loading is done
        st.sidebar.success(f"Model Loaded! 🎉 from {model_path}")
else:
    clip_model.set_testing_mode("zeroshot")

# -------------------- MAIN APP --------------------
uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Classify"):
        with st.spinner("Classifying..."):
            # Save uploaded image temporarily
            temp_path = "temp_upload.jpg"
            image.save(temp_path)

            # Classify image and get full probs
            pred_idx, top_prob, all_probs = clip_model.classify_images_clip([temp_path])
            pred_label = clip_model.class_labels[pred_idx.item()]
            confidence = top_prob.item() * 100
            
            st.success(f"**Prediction:** {pred_label} ({confidence:.2f}%)")

            # ---- EXPANDER: Show All Probabilities ----
            with st.expander("Show all class probabilities"):
                sorted_probs = sorted(
                    zip(clip_model.class_labels, all_probs.squeeze().tolist()),
                    key=lambda x: x[1],
                    reverse=True
                )
                for label, prob in sorted_probs:
                    st.write(f"{label}: {prob * 100:.2f}%")

            os.remove(temp_path)
else:
    st.info("⬆️ Drag and drop an image to get started.")

# -------------------- STYLE --------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
    }
    .stFileUploader > div {
        border: 2px dashed #ccc;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)
