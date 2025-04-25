import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from facenet_pytorch import MTCNN
import urllib.request
import os

# --- Download model if not exists
MODEL_URL = "https://huggingface.co/LinkersGraham/age-model/resolve/main/age_model_resnet18_utkface.pth"
MODEL_PATH = "age_model_resnet18_utkface.pth"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# --- Load Face Detector and Age Estimator
def load_model_and_detector():
    mtcnn = MTCNN(image_size=160, margin=0)
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 101)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return mtcnn, model

mtcnn, model = load_model_and_detector()

# --- Streamlit App
st.title("WhatsApp Profile Photo Age Estimator")

uploaded_files = st.file_uploader("Upload profile pictures", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_container_width=True)

        # Face detection
        face = mtcnn(image)

        if face is not None:
            face = face.unsqueeze(0)
            with torch.no_grad():
                age_pred = model(face)
                predicted_age = torch.argmax(age_pred, 1).item()
            st.success(f"Estimated Age: {predicted_age} years")
        else:
            st.warning("No face detected in this photo.")
