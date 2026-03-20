# 🧠 CIFAR-10 CNN Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU%20T4-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/Licence-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Statut-En%20cours-blue?style=for-the-badge)

Implémentation d'un réseau de neurones convolutif (CNN) pour la classification d'images sur le dataset **CIFAR-10**, entraîné sur GPU avec PyTorch dans Google Colab.

---

## 📊 Dataset

![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-orange?style=flat-square)
![Classes](https://img.shields.io/badge/Classes-10-blue?style=flat-square)
![Images](https://img.shields.io/badge/Images-60%2C000-lightgrey?style=flat-square)
![Résolution](https://img.shields.io/badge/Résolution-32×32-lightgrey?style=flat-square)

Le dataset CIFAR-10 contient 60 000 images couleur (32×32 px) réparties en 10 classes :

| Classe | Classe | Classe | Classe | Classe |
|--------|--------|--------|--------|--------|
| ✈️ plane | 🚗 car | 🐦 bird | 🐱 cat | 🦌 deer |
| 🐶 dog | 🐸 frog | 🐴 horse | 🚢 ship | 🚚 truck |

- **Entraînement** : 50 000 images
- **Test** : 10 000 images

---

## 🏗️ Architecture

```
Input (3×32×32)
    │
    ▼
Conv2d(3→64, 3×3) → BatchNorm → ReLU
Conv2d(64→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    │
    ▼
Conv2d(64→128, 3×3) → BatchNorm → ReLU
Conv2d(128→128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    │
    ▼
Flatten → Dropout(0.5)
FC(128×8×8 → 512) → ReLU → Dropout(0.5)
FC(512 → 10)
    │
    ▼
Output (10 classes)
```

---

## ⚙️ Configuration d'entraînement

| Hyperparamètre | Valeur |
|----------------|--------|
| Optimiseur | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| Batch size | 128 |
| Epochs | 10 |
| Scheduler | CosineAnnealingLR (T_max=10) |

---

## 🚀 Installation & Utilisation

### Prérequis

```bash
pip install torch torchvision matplotlib numpy
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

### Structure du notebook

```
📓 cifar10_cnn.ipynb
├── 🔧 0. Configuration GPU & Dépendances
├── 📦 1. Chargement et préparation des données CIFAR-10
├── 🖼️  2. Visualisation des données d'entraînement
├── 🧠 3. Architecture du réseau de neurones convolutif
├── ⚙️  4. Fonctions de perte et optimiseur
├── 🚀 5. Boucle d'entraînement
└── 🔍 6. Inférence et visualisation des prédictions
```

---

## 📈 Résultats attendus

| Modèle | Précision |
|--------|-----------|
| CNN basique (original) | ~62% |
| CNN amélioré (BatchNorm + Dropout) | ~85% |
| ResNet-18 (transfer learning) | ~94% |

---

## 🔧 Améliorations implémentées

- ✅ **Data Augmentation** : RandomCrop, HorizontalFlip, ColorJitter
- ✅ **Batch Normalization** : après chaque couche Conv2d
- ✅ **Dropout** : p=0.5 avant les couches fully-connected
- ✅ **Architecture plus profonde** : 64→128 filtres
- ✅ **Learning Rate Scheduler** : CosineAnnealingLR
- ✅ **Optimisation GPU** : `pin_memory=True`, `.to(device)` systématique
- ✅ **Mode évaluation** : `net.eval()` + `torch.no_grad()` à l'inférence

---

## 📁 Structure du projet

```
cifar10-cnn/
├── cifar10_cnn.ipynb       # Notebook principal
├── cifar_net.pth           # Poids du modèle sauvegardés
├── data/                   # Dataset CIFAR-10 (téléchargé automatiquement)
└── README.md
```

---

## 🧪 Inférence

```python
net.eval()
with torch.no_grad():
    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
```

---

## 📄 Licence

![License](https://img.shields.io/badge/Licence-MIT-green?style=flat-square)

Ce projet est sous licence MIT.
