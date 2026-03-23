import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128*4*4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ✅ Cherche model.pth dans plusieurs emplacements
candidate_paths = [
    "model.pth",                                                          # Docker / local
    os.path.join(os.path.dirname(__file__), "model.pth"),                 # même dossier que app.py
    "/kaggle/working/model.pth",                                          # Kaggle working
    "/kaggle/input/datasets/aymanebadia/cifar-model/model.pth",          # Kaggle input
]

model_path = None
for path in candidate_paths:
    if os.path.exists(path):
        model_path = path
        break

model = CNN(num_classes=10)
if model_path:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
else:
    st.error("⚠️ model.pth introuvable. Vérifiez que le fichier est présent.")
    st.stop()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

st.title("🚀 CIFAR-10 Classifier")
file = st.file_uploader("Choisissez une image...", type=["jpg","png","jpeg"])
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)
    if st.button("Lancer la prédiction"):
        img = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)
        st.success(f"Résultat : **{classes[pred.item()].upper()}**")