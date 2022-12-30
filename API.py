#https://p7opencgk.herokuapp.com/
import pandas as pd
from flask import Flask, jsonify
import joblib

app = Flask(__name__)

PATH = ''

df = pd.read_csv(PATH+'data_test.csv')
print('df shape = ', df.shape)

#Chargement du modèle
load_clf = joblib.load(PATH+r"LGBMClassifier.joblib")

#Premiers pas sur l'API
@app.route('/')
def index():
    return 'Bonjouur, Bienvenue à Mon Api!!'

#C
@app.route('/credit/<id_client>', methods=["GET"])
def credit(id_client):
    print('id client = ', id_client)

# Récupération des données du client en question
    ID = int(id_client)
    X = df[df['SK_ID_CURR'] == ID]

#Isolement des features non utilisées
    ignore_features = ['Unnamed: 0', 'SK_ID_CURR', 'INDEX', 'TARGET']
    relevant_features = [col for col in df.columns if col not in ignore_features]
    X = X[relevant_features]
    print('X shape = ', X.shape)

#Prédiction
    proba = load_clf.predict_proba(X)
    prediction = load_clf.predict(X)
    pred_proba = {
        'prediction': int(prediction),
        'proba': float(proba[0][0].round(3))
    }

    print('Nouvelle Prédiction : \n', pred_proba)

    return jsonify(pred_proba)


# Lancement de l'application
if __name__ == '__main__':
    app.run(debug=True)