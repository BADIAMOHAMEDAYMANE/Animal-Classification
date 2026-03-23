# 🧠 CIFAR-10 CNN Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU%20T4-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/Licence-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Statut-En%20cours-blue?style=for-the-badge)

![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~78%25-brightgreen?style=flat-square)
![Epochs](https://img.shields.io/badge/Epochs-15-blue?style=flat-square)
![Batch Size](https://img.shields.io/badge/Batch%20Size-128-blue?style=flat-square)
![Optimizer](https://img.shields.io/badge/Optimizer-Adam-orange?style=flat-square)
![LR](https://img.shields.io/badge/Learning%20Rate-0.001-orange?style=flat-square)
![Dropout](https://img.shields.io/badge/Dropout-0.4-yellow?style=flat-square)
![Params](https://img.shields.io/badge/Parameters-~500K-lightgrey?style=flat-square)
![Hardware](https://img.shields.io/badge/Hardware-Tesla%20T4%20GPU-76B900?style=flat-square&logo=nvidia&logoColor=white)

Implémentation d'un réseau de neurones convolutif (CNN) pour la classification d'images sur le dataset **CIFAR-10**, entraîné sur GPU avec PyTorch dans Google Colab et Kaggle. Inclut une application de démo interactive avec **Streamlit** et un déploiement **Dockerisé**.

---

## 🌐 Live Demo

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://animal-classification-2gd59hwka4ppwvsjsnp7cp.streamlit.app/)

> 🚀 **[Accéder à l'application déployée](https://animal-classification-2gd59hwka4ppwvsjsnp7cp.streamlit.app/)** — Testez le modèle directement dans votre navigateur, sans installation.

---

## 🖼️ Demo — Streamlit App

The interactive Streamlit app lets you upload any image and instantly get a prediction from the trained CNN model. Below are real inference examples across three different CIFAR-10 classes:

| ✈️ Airplane | 🐱 Cat | 🐶 Dog |
|:-----------:|:------:|:------:|
| ![Airplane prediction](plane.png) | ![Cat prediction](cat.png) | ![Dog prediction](dog.png) |
| **Input:** Air Transat Airbus A310 | **Input:** Grey fluffy cat close-up | **Input:** Beagle portrait |
| ✅ **Prediction: airplane** | ✅ **Prediction: cat** | ✅ **Prediction: dog** |

> The model correctly classified all three images despite them being high-resolution real-world photos — very different from the 32×32 training data.

---

## 📊 Dataset

![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-orange?style=flat-square)
![Classes](https://img.shields.io/badge/Classes-10-blue?style=flat-square)
![Images](https://img.shields.io/badge/Images-60%2C000-lightgrey?style=flat-square)
![Résolution](https://img.shields.io/badge/Résolution-32×32-lightgrey?style=flat-square)

Le dataset CIFAR-10 contient 60 000 images couleur (32×32 px) réparties en 10 classes :

| Classe | Classe | Classe | Classe | Classe |
|--------|--------|--------|--------|--------|
| ✈️ airplane | 🚗 automobile | 🐦 bird | 🐱 cat | 🦌 deer |
| 🐶 dog | 🐸 frog | 🐴 horse | 🚢 ship | 🚚 truck |

- **Entraînement** : 50 000 images
- **Test** : 10 000 images

---

## 🏗️ Architecture

```
Input (3×32×32)
    │
    ▼
[Bloc 1] Conv2d(3→32, 3×3) → BatchNorm2d → ReLU → MaxPool(2×2)
    │                                               32×32 → 16×16
    ▼
[Bloc 2] Conv2d(32→64, 3×3) → BatchNorm2d → ReLU → MaxPool(2×2)
    │                                               16×16 → 8×8
    ▼
[Bloc 3] Conv2d(64→128, 3×3) → BatchNorm2d → ReLU → MaxPool(2×2)
    │                                               8×8 → 4×4
    ▼
Flatten → Dropout(0.4)
FC(128×4×4 → 256) → ReLU
FC(256 → 10)
    │
    ▼
Output (10 classes)
```

---

## ⚙️ Configuration d'entraînement

| Hyperparamètre | Valeur |
|----------------|--------|
| Optimiseur | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 15 |
| Scheduler | StepLR (step_size=5, gamma=0.5) |
| Dropout | 0.4 |
| Normalisation | (0.5, 0.5, 0.5) / (0.5, 0.5, 0.5) |

---

## 🚀 Installation & Utilisation

### Prérequis

```bash
pip install torch torchvision matplotlib streamlit pillow numpy
```

### Lancer sur Google Colab

1. Ouvrir le notebook dans Google Colab
2. Activer le GPU : `Exécution → Modifier le type d'exécution → GPU (T4)`
3. Exécuter toutes les cellules dans l'ordre : `Ctrl + F9`

### Vérifier le GPU

```python
import torch
print(torch.cuda.is_available())        # True
print(torch.cuda.get_device_name(0))    # Tesla T4
```

### Lancer l'application Streamlit

Après avoir exécuté toutes les cellules du notebook (le fichier `model.pth` doit exister) :

```bash
streamlit run app.py
```

Ou directement via le déploiement en ligne : **https://animal-classification-2gd59hwka4ppwvsjsnp7cp.streamlit.app/**

### Structure du notebook

```
📓 cifar10_cnn.ipynb
├── 🔧 1.  Installation des dépendances
├── 📦 2.  Imports
├── 🗃️  3.  Chargement du dataset CIFAR-10
├── 🖼️  4.  Visualisation des données
├── 🧠 5.  Architecture du modèle CNN
├── 🚀 6.  Entraînement
├── 🔍 7.  Évaluation
├── 📈 8.  Courbe de loss
├── 💾 9.  Sauvegarde du modèle
├── 🔮 10. Fonction de prédiction
├── 🧪 11. Test sur une image
└── 🌐 12. Application Streamlit (app.py)
```

---

## 🐳 Docker

L'application est entièrement **dockerisée** — aucune installation Python requise sur votre machine.

### Prérequis

Installer [Docker Desktop](https://www.docker.com/products/docker-desktop)

### Structure des fichiers

```
cifar10-cnn/
├── app.py
├── model.pth
├── Dockerfile
└── requirements.txt
```

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model.pth .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.address=0.0.0.0"]
```

### Builder l'image

```bash
docker build -t cifar10-cnn .
```

### Lancer le conteneur

```bash
docker run -p 8501:8501 cifar10-cnn
```

Puis ouvrez **http://localhost:8501** dans votre navigateur. 🎉

### Vérifier l'image

```bash
docker images
```

```
REPOSITORY     TAG       IMAGE ID       CREATED         SIZE
cifar10-cnn    latest    1a2ad8b6d609   5 minutes ago   ~1.5GB
```

### Publier sur Docker Hub (optionnel)

```bash
# Se connecter
docker login

# Tagger l'image
docker tag cifar10-cnn votre-username/cifar10-cnn:latest

# Pusher
docker push votre-username/cifar10-cnn:latest

# N'importe qui peut ensuite lancer l'app avec :
docker pull votre-username/cifar10-cnn
docker run -p 8501:8501 votre-username/cifar10-cnn
```

---

## 📈 Résultats

![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~78%25-brightgreen?style=flat-square)
![Loss](https://img.shields.io/badge/Final%20Loss-~0.65-yellow?style=flat-square)
![Training Time](https://img.shields.io/badge/Training%20Time-~10%20min-blue?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-76B900?style=flat-square&logo=nvidia&logoColor=white)

| Modèle | Précision |
|--------|-----------|
| CNN 3 blocs + BatchNorm + Dropout (ce projet) | ~75-80% |
| ResNet-18 (transfer learning) | ~94% |

---

## 🔧 Caractéristiques techniques

- ✅ **3 blocs convolutifs** : 32 → 64 → 128 filtres avec BatchNorm
- ✅ **Batch Normalization** : après chaque couche Conv2d
- ✅ **Dropout** : p=0.4 avant les couches fully-connected
- ✅ **DataLoader optimisé** : `pin_memory=True`, `num_workers=2`
- ✅ **Learning Rate Scheduler** : StepLR (divisé par 2 tous les 5 epochs)
- ✅ **Inférence flexible** : accepte PIL Image ou Tensor
- ✅ **App Streamlit** : démo interactive avec upload d'image
- ✅ **Mode évaluation** : `model.eval()` + `torch.no_grad()` à l'inférence
- ✅ **Dockerisé** : déploiement en une commande sur n'importe quelle machine

---

## 🌐 Application Streamlit

L'application `app.py` (générée automatiquement par la cellule 12 du notebook) permet de tester le modèle en uploadant n'importe quelle image :

```
1. Uploader une image JPG ou PNG
2. Cliquer sur "Predict"
3. Le modèle retourne la classe prédite parmi les 10 classes CIFAR-10
```

🔗 **Application déployée** : https://animal-classification-2gd59hwka4ppwvsjsnp7cp.streamlit.app/

> ⚠️ Pour une exécution locale, le fichier `model.pth` doit être présent dans le même dossier que `app.py`.

---

## 📁 Structure du projet

```
cifar10-cnn/
├── cifar10_cnn.ipynb       # Notebook principal (12 cellules)
├── app.py                  # Application Streamlit (généré par cellule 12)
├── model.pth               # Poids du modèle sauvegardés (généré à l'entraînement)
├── Dockerfile              # Configuration Docker
├── requirements.txt        # Dépendances Python
├── data/                   # Dataset CIFAR-10 (téléchargé automatiquement)
└── README.md
```

---

## 🧪 Exemple d'inférence

```python
from PIL import Image
image = Image.open("mon_image.jpg")
prediction = predict_image(image)
print(f"Classe prédite : {prediction}")
```

---

## 📄 Licence

![License](https://img.shields.io/badge/Licence-MIT-green?style=flat-square)

Ce projet est sous licence MIT.
