import pandas as pd
import numpy as np
import common_modules
import pdpipe as pdp
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from joblib import dump, load
from pathlib import Path


def train_model(X_train, y_train, X_teste, y_teste):
    p = Path('models')
    if not p.exists():
        p.mkdir()
    pipe = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), norm='l1', use_idf=True)),
                     ('nb', MultinomialNB(alpha=0.01))
                     ])

    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_teste = pipe.predict(X_teste)
    print(f"Confusion Matrix para treino Naive Bayes {confusion_matrix(y_train, y_pred_train)}")
    print(f"F1 score para treino Naive Bayes {f1_score(y_train, y_pred_train)}")
    print(f"Acuracy para treino Naive Bayes {accuracy_score(y_train, y_pred_train)}")

    print(f"Confusion Matrix para treino Naive Bayes {confusion_matrix(y_teste, y_pred_teste)}")
    print(f"Confusion Matrix para teste Naive Bayes {f1_score(y_teste, y_pred_teste)}")
    print(f"Confusion Matrix para teste Naive Bayes {accuracy_score(y_teste, y_pred_teste)}")
    dump(pipe, 'models/naive_bayes.joblib')


def predict_model(texto):
    p = Path('models')
    if not p.exists():
        p.mkdir()
    try:
        model = load('models/naive_bayes.joblib')
    except:
        raise TypeError("Modelo Naive Bayes não foi treinado ainda.")

    texto = common_modules.tag_remove(texto)
    texto = common_modules.trat_texto(texto)
    prediction = model.predict([" ".join(texto)])
    print("\n")
    print(f"Polaridade Predita: {prediction[0]}")
