from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# =========================
# LOAD MODEL
# =========================
dbscan = joblib.load('models/fruit_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Reconstruire nn_predictor et train_labels 
_df = pd.read_csv('data/fruits.csv', header=None, names=['Feature1', 'Feature2'])
_X_scaled = scaler.transform(_df.values)
_all_labels = dbscan.labels_
 
# Exclure les points bruit (label == -1) s'il y en a
_mask = _all_labels != -1
train_labels = _all_labels[_mask]
_X_train = _X_scaled[_mask]
 
nn_predictor = NearestNeighbors(n_neighbors=1)
nn_predictor.fit(_X_train)
 
#Load Model
def predict_cluster(X_scaled: np.ndarray) -> np.ndarray:
    """
    DBSCAN ne supporte pas .predict() nativement sur de nouvelles données.
    On utilise NearestNeighbors : chaque nouveau point hérite du cluster
    de son voisin le plus proche parmi les points d'entraînement.
    """
    indices = nn_predictor.kneighbors(X_scaled, return_distance=False)
    return train_labels[indices.flatten()]

CLUSTER_NAMES = {
    0: "fruit intermediaire",
    1: "petits fruits",
    2: "gros fruits"
}
def get_fruit_name(cluster_id: int) -> str:
    return CLUSTER_NAMES.get(cluster_id, f"Groupe inconnu ({cluster_id})")

# =========================
# APP INIT + TITLE SWAGGER
# =========================

app = FastAPI(
    title="Fruit Clustering API",
    description="""
    API de Machine Learning pour regrouper des fruits en clusters.
 
    Modèle : DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    Entrées : 2 features numériques
    Sorties : cluster ID
 
    DBSCAN détecte automatiquement le nombre de clusters selon la densité des données,
    sans avoir à le spécifier à l'avance.
 
    Cette API fait partie d'un pipeline MLOps avec FastAPI + MLflow + Docker.
    """,
    version="2.0.0"
)

# =========================
# INPUT MODEL
# =========================
class FruitData(BaseModel):
    Feature1: float
    Feature2: float

    class Config:
        json_schema_extra = {
            "example": {
                "Feature1": 43.7,
                "Feature2": 38.57
            }
        }

# =========================
# HOME
# =========================
@app.get(
    "/",
    tags=["Health Check"],
    summary="API status",
    description="Vérifie que l'API fonctionne correctement."
)
def home():
    return {"message": "Fruit Clustering API is running",
            "algorithm": "DBSCAN",
            "n_clusters": int(len(set(train_labels)))}

# =========================
# SINGLE PREDICTION
# =========================
@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict single fruit cluster",
    description="Prédit le cluster d’un seul point de données (Feature1, Feature2)."
    "Utilise DBSCAN + NearestNeighbors pour assigner les nouveaux points."
)
def predict(data: FruitData):

    features = np.array([[data.Feature1, data.Feature2]])
    scaled = scaler.transform(features)
    cluster = predict_cluster(scaled)
    cluster_id = int(cluster[0])
    fruit_name = get_fruit_name(cluster_id)
    if cluster_id == 0:
        message = f"C'est un {fruit_name} par exemple : kiwi,petite banane,pêche,etc..)"
    elif cluster_id == 1:
        message = f"C'est un {fruit_name} par exemple : cerise,raisin,framboise,etc..)"
    elif cluster_id == 2:
        message = f"C'est un {fruit_name} par exemple : melon,poire,grosse pomme,etc..)"
    else:
        message = f"{fruit_name}. Aucune description spécifique disponible pour ce cluster."

    return {
        "cluster": cluster_id,
        "fruit_name": fruit_name,
        "message": message,
    }

# =========================
# BATCH PREDICTION
# =========================
@app.post(
    "/predict_batch",
    tags=["Prediction"],
    summary="Predict multiple fruits",
    description="Prend une liste de fruits et retourne leurs clusters via DBSCAN."
)
def predict_batch(data: List[FruitData]):

    features = np.array([[item.Feature1, item.Feature2] for item in data])
    scaled = scaler.transform(features)
    clusters = predict_cluster(scaled)

    return {
        "clusters": [int(c) for c in clusters],
        "fruits": [get_fruit_name(int(c)) for c in clusters]
    }

# =========================
# CSV PREDICTION
# =========================
@app.post(
    "/predict_csv",
    tags=["Prediction"],
    summary="Predict from CSV file",
    description="Upload un fichier CSV contenant Feature1 et Feature2."
)
def predict_csv(file: UploadFile = File(...)):

    df = pd.read_csv(file.file, header=None, names=["Feature1", "Feature2"])

    features = df.values
    scaled = scaler.transform(features)
    clusters = predict_cluster(scaled)

    return {
        "clusters": [int(c) for c in clusters],
        "fruits": [get_fruit_name(int(c)) for c in clusters]
    }

# =========================
# MODEL INFO
# =========================
@app.get(
    "/model_info",
    tags=["Model"],
    summary="DBSCAN model parameters",
    description="Retourne les paramètres du modèle DBSCAN utilisé."
)
def model_info():
    return {
        "algorithm": "DBSCAN",
        "eps": dbscan.eps,
        "min_samples": dbscan.min_samples,
        "n_clusters": int(len(set(train_labels))),
        "prediction_strategy": "NearestNeighbors (1-NN sur les points d'entraînement)"
    }
 