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

# --- Load
