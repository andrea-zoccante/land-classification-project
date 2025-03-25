import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os

# -------------------- CONFIG --------------------
st.set_page_config(page_title="EuroSAT CLIP Zero-Shot Classifier", layout="centered")
st.title("🛰️ EuroSAT Zero-Shot Image Classifier with CLIP")
st.markdown("""
Upload an image and classify it into one of the 10 EuroSAT land classes using different versions of CLIP.
""")

# -------------------- PLACEHOLDERS --------------------
class_names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
modified_names = ["Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", "Industrial", "Pasture", "Permanent Crop", "Residential", "River", "Sea Lake"]

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_hf_clip_model(model_name):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

available_models = ["Zero Shot Learning"]

# -------------------- SIDEBAR --------------------
st.sidebar.header("🔧 Settings")
label_type = st.sidebar.selectbox("Label Style", ["Class Names", "Modified Names"])
selected_model_label = st.sidebar.selectbox("Zero-Shot CLIP Model", available_models)
selected_model_name = "openai/clip-vit-base-patch32" if "Zero Shot Learning" in selected_model_label else selected_model_label
prompt_prefix = st.sidebar.text_input("Prompt Template (e.g. 'A satellite image of')", value="")

labels = class_names if label_type == "Class Names" else modified_names

model, processor = load_hf_clip_model(selected_model_name)

# -------------------- MAIN APP --------------------
uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Classify"):
        with st.spinner("Classifying..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            text_inputs = [f"{prompt_prefix} {label}".strip() for label in labels]
            inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).squeeze()

            top_prob, top_class = torch.topk(probs, 1)
            st.success(f"**Prediction:** {labels[top_class.item()]} ({top_prob.item()*100:.2f}%)")

            with st.expander("Show all class probabilities"):
                for label, prob in zip(labels, probs):
                    st.write(f"{label}: {prob.item()*100:.2f}%")
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
