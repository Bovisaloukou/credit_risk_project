from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import sys

# Ajouter le dossier racine du projet au PATH pour pouvoir importer les modules de src
# Ceci est utile en développement et essentiel pour Docker/Helm si le chemin n'est pas géré autrement
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

# Importer le service de prédiction depuis src
from src.prediction_service import CreditRiskPredictionService

# Initialiser l'application Flask
app = Flask(__name__)

# Initialiser le service de prédiction globalement
# Il chargera le pipeline au démarrage de l'API
prediction_service = CreditRiskPredictionService()

@app.route('/')
def home():
    """
    Endpoint de base pour vérifier si l'API est active et si le modèle est chargé.
    """
    status = "actif" if prediction_service.pipeline is not None else "inactif (modèle non chargé)"
    return f"API de prédiction du risque de crédit ({status}). Utilisez l'endpoint /predict."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour faire une prédiction de risque de crédit.
    Attend un JSON contenant les caractéristiques d'un ou plusieurs clients.
    """
    # Vérifier si le pipeline est chargé
    if prediction_service.pipeline is None:
        return jsonify({"error": "Modèle non chargé. Impossible de faire des prédictions. Vérifiez les logs de l'API pour l'erreur de chargement."}), 500

    # Récupérer les données JSON de la requête
    try:
        json_data = request.get_json()
        if json_data is None:
            return jsonify({"error": "Aucune données JSON reçues."}), 400

    except Exception as e:
        print(f"Erreur lors du parsing JSON: {e}")
        return jsonify({"error": "Format JSON invalide."}), 400

    # Appeler le service de prédiction
    probabilities = prediction_service.predict_proba(json_data)

    # Gérer les erreurs de prédiction
    if probabilities is None:
        return jsonify({"error": "Échec de la prédiction. Vérifiez les données d'entrée et les logs de l'API."}), 400

    # Convertir les probabilités en format JSON
    predictions_list = probabilities.tolist()
    
    # Si l'entrée était un seul dictionnaire, retourner un seul résultat
    if not isinstance(json_data, list):
        return jsonify({"probability_of_default": predictions_list[0]}), 200
    
    # Sinon, retourner une liste de résultats
    response = [{"probability_of_default": prob} for prob in predictions_list]
    return jsonify(response), 200

if __name__ == '__main__':
    print("Lancement de l'API Flask en mode développement...")
    app.run(debug=True, host='0.0.0.0', port=5000)