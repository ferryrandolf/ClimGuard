"""
ClimGuard - Backend FastAPI
Auteur : Ferry Ngnepi
Description : API REST pour le système de prévention
              des risques climatiques urbains à Douala
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# INITIALISATION DE L'APPLICATION
# ============================================================

app = FastAPI(
    title="ClimGuard API",
    description="API de prévention des risques climatiques urbains à Douala, Cameroun",
    version="1.0.0"
)

# Configuration CORS
# Permet au frontend HTML de communiquer avec le backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les fichiers statiques du frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ============================================================
# CHARGEMENT DU MODÈLE IA ET DES DONNÉES
# ============================================================

# Chargement du modèle entraîné
MODEL_DIR = "backend/models"
DATA_DIR = "data/raw"

def load_model():
    """Charge le modèle IA et ses composants"""
    with open(f"{MODEL_DIR}/climguard_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/label_encoder.pkl", 'rb') as f:
        le = pickle.load(f)
    with open(f"{MODEL_DIR}/features.pkl", 'rb') as f:
        features = pickle.load(f)
    return model, scaler, le, features

def load_data():
    """Charge les données climatiques et zones à risque"""
    df_weather = pd.read_csv(f"{DATA_DIR}/douala_weather_2010_2024.csv")
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    df_zones = pd.read_csv(f"{DATA_DIR}/douala_zones_risque.csv")
    return df_weather, df_zones

# Chargement au démarrage
model, scaler, le, features = load_model()
df_weather, df_zones = load_data()

print("✅ Modèle IA chargé avec succès")
print("✅ Données climatiques chargées avec succès")

# ============================================================
# ENDPOINT 1 — PAGE D'ACCUEIL
# ============================================================

@app.get("/")
def home():
    """Sert la page principale du dashboard"""
    return FileResponse("frontend/index.html")

# ============================================================
# ENDPOINT 2 — STATUT DE L'API
# ============================================================

@app.get("/api/status")
def get_status():
    """Vérifie que l'API fonctionne correctement"""
    return {
        "status": "online",
        "projet": "ClimGuard",
        "version": "1.0.0",
        "ville": "Douala, Cameroun",
        "modele": "Random Forest - 89% précision",
        "donnees": f"{len(df_weather)} jours de données météo"
    }

# ============================================================
# ENDPOINT 3 — ZONES À RISQUE
# ============================================================

@app.get("/api/zones")
def get_zones():
    """
    Retourne la liste de tous les quartiers de Douala
    avec leur niveau de risque calculé par le modèle IA
    """
    zones = df_zones.to_dict(orient='records')

    # Ajout de la couleur pour la carte
    couleur_map = {
        'Élevé': '#e74c3c',   # Rouge
        'Moyen': '#f39c12',   # Orange
        'Faible': '#2ecc71'   # Vert
    }

    for zone in zones:
        zone['couleur'] = couleur_map.get(zone['niveau_risque'], '#95a5a6')

    return {
        "status": "success",
        "total_quartiers": len(zones),
        "ville": "Douala, Cameroun",
        "zones": zones
    }

# ============================================================
# ENDPOINT 4 — STATISTIQUES CLIMATIQUES
# ============================================================

@app.get("/api/stats")
def get_stats():
    """
    Retourne les statistiques climatiques clés de Douala
    pour affichage dans le dashboard
    """
    # Calculs sur les données réelles
    pluie_moyenne = round(float(df_weather['precipitation_sum'].mean()), 2)
    pluie_max = round(float(df_weather['precipitation_sum'].max()), 2)
    temp_moyenne = round(float(df_weather['temperature_2m_max'].mean()), 2)
    jours_critiques = int(len(df_weather[
        df_weather['precipitation_sum'] > 100
    ]))
    jours_alerte = int(len(df_weather[
        df_weather['precipitation_sum'] > 50
    ]))

    # Pluviométrie par mois
    df_weather['mois'] = df_weather['date'].dt.month
    pluie_mensuelle = df_weather.groupby('mois')[
        'precipitation_sum'
    ].mean().round(2)

    mois_labels = ['Jan','Fév','Mar','Avr','Mai','Jun',
                   'Jul','Aoû','Sep','Oct','Nov','Déc']

    return {
        "status": "success",
        "indicateurs": {
            "pluie_moyenne_journaliere": pluie_moyenne,
            "pluie_maximale_enregistree": pluie_max,
            "temperature_moyenne_max": temp_moyenne,
            "jours_critiques_par_an": round(jours_critiques / 14),
            "jours_alerte_par_an": round(jours_alerte / 14),
            "annees_donnees": 14
        },
        "pluviometrie_mensuelle": {
            "labels": mois_labels,
            "valeurs": pluie_mensuelle.tolist()
        },
        "zones_risque": {
            "eleve": int(len(df_zones[df_zones['niveau_risque'] == 'Élevé'])),
            "moyen": int(len(df_zones[df_zones['niveau_risque'] == 'Moyen'])),
            "faible": int(len(df_zones[df_zones['niveau_risque'] == 'Faible']))
        }
    }

# ============================================================
# ENDPOINT 5 — PRÉDICTION IA
# ============================================================

@app.post("/api/predict")
def predict_risk(data: dict):
    """
    Prédit le niveau de risque d'un quartier
    en utilisant le modèle Random Forest entraîné
    """
    try:
        # Préparation des données
        input_data = pd.DataFrame([data])[features]
        input_scaled = scaler.transform(input_data)

        # Prédiction
        prediction = model.predict(input_scaled)
        probabilites = model.predict_proba(input_scaled)[0]
        niveau = le.inverse_transform(prediction)[0]

        # Formatage des probabilités
        probs = {
            classe: round(float(prob) * 100, 1)
            for classe, prob in zip(le.classes_, probabilites)
        }

        return {
            "status": "success",
            "niveau_risque": niveau,
            "probabilites": probs,
            "recommandation": get_recommandation(niveau)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_recommandation(niveau: str) -> str:
    """Retourne une recommandation selon le niveau de risque"""
    recommandations = {
        "Élevé": "⚠️ Risque élevé : Activation du plan d'urgence recommandée. Évacuation préventive des zones basses.",
        "Moyen": "⚡ Risque modéré : Surveillance renforcée recommandée. Préparer les équipes d'intervention.",
        "Faible": "✅ Risque faible : Surveillance normale. Maintenir les canaux de drainage."
    }
    return recommandations.get(niveau, "Information non disponible")

# ============================================================
# LANCEMENT DU SERVEUR
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )