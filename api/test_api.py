import requests
import json
import os
import numpy as np

# URL de l'API (ajuster si l'API ne tourne pas sur localhost:5000)
API_URL = 'http://127.0.0.1:5000'

# Données d'exemple pour la prédiction
data_single = {
    "income": 75000.0,
    "loan_amount": 30000.0,
    "payment_delays": 1,
    "employment_length": 10.0,
    "age": 45
}

data_multiple = [
    {
        "income": 75000.0,
        "loan_amount": 30000.0,
        "payment_delays": 1,
        "employment_length": 10.0,
        "age": 45
    },
    {
        "income": 35000.0,
        "loan_amount": 22000.0,
        "payment_delays": 4,
        "employment_length": 2.5,
        "age": 28
    },
    {
        "income": 90000.0,
        "loan_amount": 15000.0,
        "payment_delays": 0,
        "employment_length": 18.0,
        "age": 55
    },
    # Exemple avec valeur manquante (l'imputer devrait la gérer)
    {
        "income": 50000.0,
        "loan_amount": 20000.0,
        "payment_delays": 2,
        "employment_length": None,  # Utiliser None au lieu de np.nan pour la compatibilité JSON
        "age": 40
    }
]

def test_home_endpoint():
    """Teste l'endpoint racine."""
    print(f"Testing GET {API_URL}/")
    try:
        response = requests.get(API_URL)
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")
        assert response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du test GET: {e}")
        print(f"Assurez-vous que l'API tourne à {API_URL}")

def test_predict_endpoint(test_data):
    """Teste l'endpoint /predict."""
    print(f"\nTesting POST {API_URL}/predict with data:")
    try:
        response = requests.post(f"{API_URL}/predict", json=test_data)
        print(f"Status Code: {response.status_code}")

        try:
            response_json = response.json()
            print(f"Response Body: {json.dumps(response_json, indent=2)}")
            if response.status_code == 200:
                assert isinstance(response_json, list) if isinstance(test_data, list) else isinstance(response_json, dict)
                if isinstance(response_json, list):
                    assert all("probability_of_default" in item for item in response_json)
                else:
                    assert "probability_of_default" in response_json
                print("Test /predict réussi.")
            else:
                print(f"Test /predict échoué. Code statut : {response.status_code}, Erreur : {response_json.get('error', 'N/A')}")

        except json.JSONDecodeError:
            print(f"Erreur: La réponse n'est pas un JSON valide. Corps de la réponse: {response.text}")
            print("Test /predict échoué (réponse non JSON).")

    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du test POST: {e}")
        print(f"Assurez-vous que l'API tourne à {API_URL}")
        print("Test /predict échoué (erreur réseau).")

if __name__ == '__main__':
    print("--- Début des tests de l'API ---")
    test_home_endpoint()
    print("\n" + "="*30)
    test_predict_endpoint(data_single)
    print("\n" + "="*30)
    test_predict_endpoint(data_multiple)
    print("--- Fin des tests de l'API ---")