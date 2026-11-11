FROM python:3.10-slim

WORKDIR /app

# Dépendances (roues PyTorch CPU via l'index défini dans requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Télécharge les poids de MobileNetV2 (filtre hors domaine) au build plutôt qu'au premier lancement
RUN python -c "from torchvision.models import mobilenet_v2, MobileNet_V2_Weights; mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)"

COPY app.py model.pth ./

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
