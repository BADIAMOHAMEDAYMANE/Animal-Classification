import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F
import os

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

VALID_IMAGENET = {
    'airplane': [404, 405, 895],
    'automobile': [407,408,409,410,411,412,436,468,511,627,654,656,705,757,817],
    'bird': list(range(7, 24)) + list(range(80, 101)),
    'cat': [281, 282, 283, 284, 285],
    'deer': [351, 352, 353, 354, 355],
    'dog': list(range(151, 269)),
    'frog': [30, 31, 32],
    'horse': [339, 340],
    'ship': [510, 576, 833, 894],
    'truck': [555, 569, 656, 675, 717, 734, 757, 864, 867]
}
ALL_VALID_IDS = set(idx for ids in VALID_IMAGENET.values() for idx in ids)

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

# -- Load CIFAR-10 model --
candidate_paths = [
    "model.pth",
    os.path.join(os.path.dirname(__file__), "model.pth"),
    "/kaggle/working/model.pth",
    "/kaggle/input/datasets/aymanebadia/cifar-model/model.pth",
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
    st.error("model.pth not found. Please make sure the file is present.")
    st.stop()

# -- Load MobileNetV2 pre-filter --
@st.cache_resource
def load_mobilenet():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m.eval()
    return m

mobilenet = load_mobilenet()

# -- Transforms --
transform_cifar = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

transform_mobilenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -- Pre-filter using MobileNet --
def prefilter(image):
    img = transform_mobilenet(image).unsqueeze(0)
    with torch.no_grad():
        out = mobilenet(img)
        probs = F.softmax(out, dim=1)
        top_ids = torch.argsort(probs[0], descending=True)[:5].tolist()
    for idx in top_ids:
        if idx in ALL_VALID_IDS:
            return True
    return False

# -- OOD detection --
def entropy(probs):
    log_p = torch.log(probs + 1e-9)
    return float(-torch.sum(probs * log_p).item())

MAX_ENTROPY = float(-torch.log(torch.tensor(1.0 / 10)))

def is_ood(probs, confidence_threshold=0.80, entropy_threshold=0.60):
    confidence, _ = torch.max(probs, 1)
    conf_val = confidence.item()
    ent_val = entropy(probs)
    rel_entropy = ent_val / MAX_ENTROPY
    if conf_val < confidence_threshold:
        return True, "low confidence (" + f"{conf_val:.1%})"
    if rel_entropy > entropy_threshold:
        return True, "high uncertainty (entropy " + f"{rel_entropy:.0%} of max)"
    return False, "confidence " + f"{conf_val:.1%}"

# -- UI --
st.title("CIFAR-10 Classifier")
st.caption("Classes: airplane - automobile - bird - cat - deer - dog - frog - horse - ship - truck")

file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Loaded image", use_column_width=True)
    if st.button("Run prediction"):
        with st.spinner("Analyzing..."):
            valid = prefilter(image)
            if not valid:
                st.warning("UNKNOWN - image does not belong to any CIFAR-10 category")
                st.info("Rejected: not airplane, car, bird, cat, deer, dog, frog, horse, ship or truck")
            else:
                img = transform_cifar(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(img)
                    probs = F.softmax(output, dim=1)
                    _, pred = torch.max(probs, 1)
                unknown, reason = is_ood(probs)
                if unknown:
                    st.warning("UNKNOWN - image outside CIFAR-10 classes | Reason: " + reason)
                else:
                    predicted_class = classes[pred.item()].upper()
                    st.success("Result: " + predicted_class + " - " + reason)
                st.divider()
                st.subheader("Probability distribution")
                prob_dict = {cls: float(p) for cls, p in zip(classes, probs[0])}
                st.bar_chart(prob_dict)
```
