import pandas as pd
import numpy as np
import common_modules
import pdpipe as pdp
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from joblib import dump, load
from pathlib import Path


def train_model(X_train, y_train, X_teste, y_teste):
    p = Path('models')
    if not p.exists():
        p.mkdir()
    pipe = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
                     ('logreg', LogisticRegression(C=1000, solver='newton-cg',
                                                   random_state=42, n_jobs=-1))
                     ])

    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_teste = pipe.predict(X_teste)
    print(f"Confusion Matrix para treino regressão logistica {confusion_matrix(y_train, y_pred_train)}")
    print(f"F1 score para treino regressão logistica {f1_score(y_train, y_pred_train)}")
    print(f"Acuracy para treino regressão logistica {accuracy_score(y_train, y_pred_train)}")

    print(f"Confusion Matrix para treino regressão logistica {confusion_matrix(y_teste, y_pred_teste)}")
    print(f"Confusion Matrix para teste regressão logistica {f1_score(y_teste, y_pred_teste)}")
    print(f"Confusion Matrix para teste regressão logistica {accuracy_score(y_teste, y_pred_teste)}")
    dump(pipe, 'models/log_reg.joblib')


def predict_model(texto):
    p = Path('models')
    if not p.exists():
        p.mkdir()

    try:
        model = load('models/log_reg.joblib')
    except:
        raise TypeError("Modelo de regressão logistica não foi treinado ainda")

    texto = common_modules.tag_remove(texto)
    texto = common_modules.trat_texto(texto)
    prediction = model.predict([" ".join(texto)])
    print("\n")
    print(f"Polaridade Predita: {prediction[0]}")
