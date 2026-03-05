"""
ClimGuard - Modèle IA de classification des risques climatiques
Auteur : Ferry Ngnepi
Description : Modèle de classification Random Forest pour prédire
              le niveau de risque d'inondation par quartier à Douala
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "backend/models"
DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# ÉTAPE 1 — PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT
# ============================================================

def prepare_training_data():
    """
    Prépare les données d'entraînement pour le modèle.
    On combine les données météo historiques avec les
    caractéristiques des quartiers de Douala.
    """
    print("📊 Préparation des données d'entraînement...")

    # Chargement des données météo
    df_weather = pd.read_csv(f"{DATA_DIR}/douala_weather_2010_2024.csv")
    df_weather['date'] = pd.to_datetime(df_weather['date'])

    # Calcul des statistiques météo par saison
    df_weather['mois'] = df_weather['date'].dt.month
    df_weather['saison_pluie'] = df_weather['mois'].apply(
        lambda m: 1 if m in [4, 5, 6, 7, 8, 9, 10] else 0
    )

    # Statistiques globales pour référence
    stats_meteo = {
        'pluie_moyenne_journaliere': df_weather['precipitation_sum'].mean(),
        'pluie_max_journaliere': df_weather['precipitation_sum'].max(),
        'pluie_saison_pluie': df_weather[
            df_weather['saison_pluie'] == 1
        ]['precipitation_sum'].mean(),
        'temp_max_moyenne': df_weather['temperature_2m_max'].mean(),
        'jours_pluie_intense': len(
            df_weather[df_weather['precipitation_sum'] > 50]
        )
    }

    print(f"   → Pluie moyenne journalière : "
          f"{stats_meteo['pluie_moyenne_journaliere']:.2f} mm")
    print(f"   → Pluie maximale enregistrée : "
          f"{stats_meteo['pluie_max_journaliere']:.2f} mm")
    print(f"   → Jours de pluie intense (>50mm) : "
          f"{stats_meteo['jours_pluie_intense']} jours")

    # Chargement des zones à risque
    df_zones = pd.read_csv(f"{DATA_DIR}/douala_zones_risque.csv")

    # Enrichissement avec des caractéristiques supplémentaires
    # basées sur la connaissance terrain de Douala
    df_zones['pluie_reference'] = stats_meteo['pluie_moyenne_journaliere']
    df_zones['pluie_saison'] = stats_meteo['pluie_saison_pluie']
    df_zones['jours_risque_annuel'] = stats_meteo['jours_pluie_intense']

    # Ajout d'une caractéristique d'altitude approximative
    # (basée sur la géographie connue de Douala)
    altitude_map = {
        'Bonabéri': 5,
        'Ndokotti': 8,
        'New Bell': 10,
        'Bépanda': 15,
        'Akwa': 12,
        'Bonamoussadi': 35,
        'Makepe': 40,
        'Kotto': 25,
        'Logpom': 20,
        'Deido': 7
    }
    df_zones['altitude_approx'] = df_zones['quartier'].map(altitude_map)

    # Ajout densité de population approximative (habitants/km²)
    densite_map = {
        'Bonabéri': 8500,
        'Ndokotti': 25000,
        'New Bell': 42000,
        'Bépanda': 18000,
        'Akwa': 12000,
        'Bonamoussadi': 9000,
        'Makepe': 7500,
        'Kotto': 6000,
        'Logpom': 11000,
        'Deido': 22000
    }
    df_zones['densite_population'] = df_zones['quartier'].map(densite_map)

    print(f"   → {len(df_zones)} quartiers enrichis avec "
          f"{len(df_zones.columns)} caractéristiques")
    return df_zones, stats_meteo


# ============================================================
# ÉTAPE 2 — ENTRAÎNEMENT DU MODÈLE
# ============================================================

def train_model(df_zones):
    """
    Entraîne un modèle Random Forest pour classifier
    le niveau de risque de chaque quartier.

    Pourquoi Random Forest ?
    - Robuste avec peu de données
    - Interprétable (on peut voir quelles variables comptent)
    - Pas besoin de beaucoup de paramétrage
    - Excellent pour la classification
    """
    print("\n🤖 Entraînement du modèle IA...")

    # Sélection des caractéristiques (features)
    features = [
        'score_risque',           # Score historique de risque
        'population_exposee',     # Population exposée
        'altitude_approx',        # Altitude approximative
        'densite_population',     # Densité de population
        'pluie_reference',        # Pluie de référence
        'pluie_saison',           # Pluie en saison des pluies
        'jours_risque_annuel',    # Jours de pluie intense/an
        'latitude',               # Position géographique
        'longitude'               # Position géographique
    ]

    # Variable cible (ce qu'on veut prédire)
    target = 'niveau_risque'

    X = df_zones[features]
    y = df_zones[target]

    # Encodage de la variable cible
    # (transformer Élevé/Moyen/Faible en nombres 0/1/2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"   → Classes : {list(le.classes_)}")
    distribution = dict(zip(le.classes_, np.bincount(y_encoded)))
    print(f"   → Distribution : {distribution}")

    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Division train/test (80% entraînement, 20% test)
    # Note : avec seulement 10 quartiers, on fait une
    # validation croisée plutôt qu'un vrai split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42
    )

    # Création et entraînement du modèle Random Forest
    model = RandomForestClassifier(
        n_estimators=100,    # 100 arbres de décision
        max_depth=5,         # Profondeur max de chaque arbre
        random_state=42,     # Pour reproductibilité
        class_weight='balanced'  # Équilibre les classes
    )

    model.fit(X_train, y_train)

    # Validation croisée (plus fiable avec peu de données)
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=3)
    print(f"   → Score validation croisée : "
          f"{cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

    # Évaluation sur les données de test
    y_pred = model.predict(X_test)

    # Importance des variables
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n📊 Importance des variables :")
    for _, row in feature_importance.iterrows():
        bar = "█" * int(row['importance'] * 30)
        print(f"   {row['feature']:<30} {bar} "
              f"{row['importance']:.3f}")

    return model, scaler, le, features, feature_importance


# ============================================================
# ÉTAPE 3 — SAUVEGARDE DU MODÈLE
# ============================================================

def save_model(model, scaler, le, features):
    """
    Sauvegarde le modèle entraîné pour utilisation
    par le backend FastAPI.
    Le format pickle permet de sauvegarder l'objet
    Python complet et de le recharger plus tard.
    """
    print("\n💾 Sauvegarde du modèle...")

    # Sauvegarde du modèle
    model_path = f"{MODEL_DIR}/climguard_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Sauvegarde du scaler
    scaler_path = f"{MODEL_DIR}/scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Sauvegarde de l'encodeur
    encoder_path = f"{MODEL_DIR}/label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)

    # Sauvegarde de la liste des features
    features_path = f"{MODEL_DIR}/features.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)

    print(f"   ✅ Modèle sauvegardé : {model_path}")
    print(f"   ✅ Scaler sauvegardé : {scaler_path}")
    print(f"   ✅ Encodeur sauvegardé : {encoder_path}")


# ============================================================
# ÉTAPE 4 — TEST DE PRÉDICTION
# ============================================================

def test_prediction(model, scaler, le, features):
    """
    Teste le modèle avec un exemple concret.
    Simule une prédiction pour un nouveau quartier.
    """
    print("\n🧪 Test de prédiction...")

    # Exemple : nouveau quartier avec caractéristiques connues
    nouveau_quartier = {
        'score_risque': 0.75,
        'population_exposee': 95000,
        'altitude_approx': 8,
        'densite_population': 20000,
        'pluie_reference': 8.26,
        'pluie_saison': 12.5,
        'jours_risque_annuel': 89,
        'latitude': 4.055,
        'longitude': 9.720
    }

    # Préparer les données
    X_new = pd.DataFrame([nouveau_quartier])[features]
    X_new_scaled = scaler.transform(X_new)

    # Prédiction
    prediction = model.predict(X_new_scaled)
    probabilites = model.predict_proba(X_new_scaled)[0]
    niveau_predit = le.inverse_transform(prediction)[0]

    print(f"   → Niveau de risque prédit : {niveau_predit}")
    print(f"   → Probabilités par classe :")
    for classe, prob in zip(le.classes_, probabilites):
        barre = "█" * int(prob * 20)
        print(f"      {classe:<10} {barre} {prob:.1%}")

    return niveau_predit


# ============================================================
# EXÉCUTION PRINCIPALE
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("   ClimGuard — Entraînement du modèle IA")
    print("=" * 55)

    # 1. Préparation des données
    df_zones, stats_meteo = prepare_training_data()

    # 2. Entraînement
    model, scaler, le, features, importance = train_model(df_zones)

    # 3. Sauvegarde
    save_model(model, scaler, le, features)

    # 4. Test
    test_prediction(model, scaler, le, features)

    print("\n" + "=" * 55)
    print("   ✅ Modèle IA ClimGuard prêt !")
    print("   📁 Fichiers dans : backend/models/")
    print("=" * 55)