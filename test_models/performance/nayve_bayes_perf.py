import pandas as pd
import numpy as np
import common_modules
import pdpipe as pdp
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def naive_bayes(X_train, y_train, X_teste, y_teste):
    pipe = Pipeline([('tfidf', TfidfVectorizer(ngram_range = (1,3), norm = 'l2', use_idf = True)),
                     ('nb', MultinomialNB( alpha = 0.1))
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
