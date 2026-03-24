import subprocess
subprocess.run(["pip", "install", "pyngrok", "-q"], check=True)
subprocess.run(["pip", "install", "streamlit", "-q"], check=True)

import os, time
from pyngrok import ngrok, conf

PYTHON = "/usr/local/bin/python"
NGROK_TOKEN = "3BJIfbZohuOTp30RUfTEoruFv1Z_yvtLcWM4JS9CBBgxuUTK"

app_code = (
    "import streamlit as st\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torchvision.transforms as transforms\n"
    "import torchvision.models as models\n"
    "from PIL import Image\n"
    "import torch.nn.functional as F\n"
    "import os\n"
    "\n"
    "classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']\n"
    "\n"
    "VALID_IMAGENET = {\n"
    "    'airplane': [404, 405, 895],\n"
    "    'automobile': [407,408,409,410,411,412,436,468,511,627,654,656,705,757,817],\n"
    "    'bird': list(range(7, 24)) + list(range(80, 101)),\n"
    "    'cat': [281, 282, 283, 284, 285],\n"
    "    'deer': [351, 352, 353, 354, 355],\n"
    "    'dog': list(range(151, 269)),\n"
    "    'frog': [30, 31, 32],\n"
    "    'horse': [339, 340],\n"
    "    'ship': [510, 576, 833, 894],\n"
    "    'truck': [555, 569, 656, 675, 717, 734, 757, 864, 867]\n"
    "}\n"
    "ALL_VALID_IDS = set(idx for ids in VALID_IMAGENET.values() for idx in ids)\n"
    "\n"
    "class CNN(nn.Module):\n"
    "    def __init__(self, num_classes=10):\n"
    "        super(CNN, self).__init__()\n"
    "        self.features = nn.Sequential(\n"
    "            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),\n"
    "            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),\n"
    "            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),\n"
    "        )\n"
    "        self.classifier = nn.Sequential(\n"
    "            nn.Dropout(0.4),\n"
    "            nn.Linear(128*4*4, 256),\n"
    "            nn.ReLU(inplace=True),\n"
    "            nn.Linear(256, num_classes)\n"
    "        )\n"
    "    def forward(self, x):\n"
    "        x = self.features(x)\n"
    "        x = x.view(x.size(0), -1)\n"
    "        return self.classifier(x)\n"
    "\n"
    "model = CNN(num_classes=10)\n"
    "model_path = '/kaggle/working/model.pth'\n"
    "if not os.path.exists(model_path):\n"
    "    model_path = '/kaggle/input/datasets/aymanebadia/cifar-model/model.pth'\n"
    "if os.path.exists(model_path):\n"
    "    model.load_state_dict(torch.load(model_path, map_location='cpu'))\n"
    "    model.eval()\n"
    "else:\n"
    "    st.error('Model not found at: ' + model_path)\n"
    "\n"
    "@st.cache_resource\n"
    "def load_mobilenet():\n"
    "    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)\n"
    "    m.eval()\n"
    "    return m\n"
    "\n"
    "mobilenet = load_mobilenet()\n"
    "\n"
    "transform_cifar = transforms.Compose([\n"
    "    transforms.Resize((32, 32)),\n"
    "    transforms.ToTensor(),\n"
    "    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))\n"
    "])\n"
    "\n"
    "transform_mobilenet = transforms.Compose([\n"
    "    transforms.Resize((224, 224)),\n"
    "    transforms.ToTensor(),\n"
    "    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])\n"
    "])\n"
    "\n"
    "def prefilter(image):\n"
    "    img = transform_mobilenet(image).unsqueeze(0)\n"
    "    with torch.no_grad():\n"
    "        out = mobilenet(img)\n"
    "        probs = F.softmax(out, dim=1)\n"
    "        top_ids = torch.argsort(probs[0], descending=True)[:5].tolist()\n"
    "    for idx in top_ids:\n"
    "        if idx in ALL_VALID_IDS:\n"
    "            return True\n"
    "    return False\n"
    "\n"
    "def entropy(probs):\n"
    "    log_p = torch.log(probs + 1e-9)\n"
    "    return float(-torch.sum(probs * log_p).item())\n"
    "\n"
    "MAX_ENTROPY = float(-torch.log(torch.tensor(1.0 / 10)))\n"
    "\n"
    "def is_ood(probs, confidence_threshold=0.80, entropy_threshold=0.60):\n"
    "    confidence, _ = torch.max(probs, 1)\n"
    "    conf_val = confidence.item()\n"
    "    ent_val = entropy(probs)\n"
    "    rel_entropy = ent_val / MAX_ENTROPY\n"
    "    if conf_val < confidence_threshold:\n"
    "        return True, 'low confidence (' + f'{conf_val:.1%})'\n"
    "    if rel_entropy > entropy_threshold:\n"
    "        return True, 'high uncertainty (entropy ' + f'{rel_entropy:.0%} of max)'\n"
    "    return False, 'confidence ' + f'{conf_val:.1%}'\n"
    "\n"
    "st.title('CIFAR-10 Classifier')\n"
    "st.caption('Classes: airplane - automobile - bird - cat - deer - dog - frog - horse - ship - truck')\n"
    "\n"
    "file = st.file_uploader('Choose an image...', type=['jpg','png','jpeg'])\n"
    "\n"
    "if file:\n"
    "    image = Image.open(file).convert('RGB')\n"
    "    st.image(image, caption='Loaded image', use_column_width=True)\n"
    "    if st.button('Run prediction'):\n"
    "        with st.spinner('Analyzing...'):\n"
    "            valid = prefilter(image)\n"
    "            if not valid:\n"
    "                st.warning('UNKNOWN - image does not belong to any CIFAR-10 category')\n"
    "                st.info('Rejected by pre-filter: not airplane, car, bird, cat, deer, dog, frog, horse, ship or truck')\n"
    "            else:\n"
    "                img = transform_cifar(image).unsqueeze(0)\n"
    "                with torch.no_grad():\n"
    "                    output = model(img)\n"
    "                    probs = F.softmax(output, dim=1)\n"
    "                    _, pred = torch.max(probs, 1)\n"
    "                unknown, reason = is_ood(probs)\n"
    "                if unknown:\n"
    "                    st.warning('UNKNOWN - image outside CIFAR-10 classes | Reason: ' + reason)\n"
    "                else:\n"
    "                    predicted_class = classes[pred.item()].upper()\n"
    "                    st.success('Result: ' + predicted_class + ' - ' + reason)\n"
    "                st.divider()\n"
    "                st.subheader('Probability distribution')\n"
    "                prob_dict = {cls: float(p) for cls, p in zip(classes, probs[0])}\n"
    "                st.bar_chart(prob_dict)\n"
)

with open("/kaggle/working/app.py", "w") as f:
    f.write(app_code)

print("Cleaning old processes...")
subprocess.run(["pkill", "-f", "ngrok"], capture_output=True)
subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
time.sleep(2)

print("Starting Streamlit in background...")
with open("/kaggle/working/streamlit.log", "w") as sl:
    subprocess.Popen(
        [PYTHON, "-m", "streamlit", "run", "/kaggle/working/app.py",
         "--server.port=8501", "--server.headless=true"],
        stdout=sl, stderr=sl, start_new_session=True
    )
time.sleep(8)

conf.get_default().auth_token = NGROK_TOKEN
try:
    tunnels = ngrok.get_tunnels()
    for t in tunnels:
        ngrok.disconnect(t.public_url)
    print("Connecting to static domain...")
    public_url = ngrok.connect(8501, domain="jolie-intromissive-ken.ngrok-free.dev")
    print("SUCCESS! Your app is live at: " + str(public_url.public_url))
except Exception as e:
    print("Static domain error: " + str(e))
    try:
        ngrok.kill()
        time.sleep(2)
        public_url = ngrok.connect(8501)
        print("SUCCESS (random URL): " + str(public_url.public_url))
    except Exception as e2:
        print("Critical Ngrok failure: " + str(e2))
