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

## 🛡️ Robustesse de la démo

Un CNN entraîné sur CIFAR-10 répond toujours l'une des 10 classes, même pour une image qui n'a rien à voir (un visage, un document…). L'application ajoute donc deux garde-fous :

1. **Filtre hors domaine** : un **MobileNetV2** pré-entraîné sur ImageNet analyse d'abord l'image (224×224). Si aucune de ses 5 meilleures classes ne correspond à une catégorie proche de CIFAR-10, l'image est classée **UNKNOWN**.
2. **Rejet des prédictions peu sûres** : la prédiction du CNN est aussi rejetée (**UNKNOWN**) si la confiance maximale est **inférieure à 0,80** ou si l'**entropie normalisée** des probabilités (entropie / ln 10) **dépasse 0,60**.

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

### Notebooks

- `ImageClassification.ipynb` : version Google Colab (GPU T4)
- `AnimalClassification.ipynb` : version Kaggle (GPU T4, précision mixte avec `torch.amp.autocast`)

Étapes suivies :

```
📓 Notebook
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

### Fichiers Docker

- `Dockerfile` : image `python:3.10-slim`, roues PyTorch **CPU** (image plus légère), poids MobileNetV2 téléchargés au build
- `docker-compose.yml` : service `cifar10-app` exposé sur le port 8501
- `.dockerignore` : exclut notebooks, images de démo et dossier `.git`

### Lancer avec Docker Compose (recommandé)

```bash
docker compose up -d --build
```

Pour arrêter : `docker compose down`

### Ou avec Docker seul

```bash
docker build -t cifar10-cnn .
docker run -p 8501:8501 cifar10-cnn
```

Puis ouvrez **http://localhost:8501** dans votre navigateur. 🎉

### Vérifier l'image

```bash
docker images
```

```
REPOSITORY     TAG       IMAGE ID       CREATED         SIZE
cifar10-cnn    latest    1a2ad8b6d609   5 minutes ago   1.76GB
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

| Modèle | Accuracy (jeu de test) |
|--------|-----------|
| CNN 3 blocs + BatchNorm + Dropout (**ce projet**) | **~78 %** |
| ResNet-18 en transfer learning (*valeur de référence de la littérature, non entraîné dans ce projet*) | ~94 % |

> Piste d'amélioration : data augmentation (crop aléatoire, flip horizontal) et transfer learning pour se rapprocher des ~94 %.

---

## 🔧 Caractéristiques techniques

- ✅ **3 blocs convolutifs** : 32 → 64 → 128 filtres avec BatchNorm
- ✅ **Batch Normalization** : après chaque couche Conv2d
- ✅ **Dropout** : p=0.4 avant les couches fully-connected
- ✅ **DataLoader optimisé** : `pin_memory=True`, `num_workers=2`
- ✅ **Learning Rate Scheduler** : StepLR (divisé par 2 tous les 5 epochs)
- ✅ **Inférence flexible** : accepte PIL Image ou Tensor
- ✅ **App Streamlit** : démo interactive avec upload d'image
- ✅ **Filtre hors domaine** : MobileNetV2 + rejet si confiance < 0,80 ou entropie normalisée > 0,60
- ✅ **Mode évaluation** : `model.eval()` + `torch.no_grad()` à l'inférence
- ✅ **Dockerisé** : `docker compose up -d` sur n'importe quelle machine

---

## 🌐 Application Streamlit

L'application `app.py` permet de tester le modèle en uploadant n'importe quelle image :

```
1. Uploader une image JPG ou PNG
2. Cliquer sur "Predict"
3. Le modèle retourne la classe prédite parmi les 10 classes CIFAR-10 (ou UNKNOWN si l'image est hors domaine ou la prédiction peu sûre), avec le graphique des probabilités
```

🔗 **Application déployée** : https://animal-classification-2gd59hwka4ppwvsjsnp7cp.streamlit.app/

> ⚠️ Pour une exécution locale, le fichier `model.pth` doit être présent dans le même dossier que `app.py`.

---

## 📁 Structure du projet

```
Animal-Classification/
├── ImageClassification.ipynb    # Notebook Google Colab
├── AnimalClassification.ipynb   # Notebook Kaggle (précision mixte)
├── app.py                       # Application Streamlit (CNN + filtre MobileNetV2)
├── model.pth                    # Poids du modèle entraîné
├── requirements.txt             # Dépendances (PyTorch CPU)
├── Dockerfile                   # Image Docker de l'application
├── docker-compose.yml           # Lancement en une commande
├── .dockerignore
├── .devcontainer/               # Environnement de développement (Codespaces)
├── plane.png, cat.png, dog.png  # Captures de démonstration
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
