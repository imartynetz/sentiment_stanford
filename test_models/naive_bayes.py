import pandas as pd
import numpy as np
import common_modules
import pdpipe as pdp
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def naive_bayes(X_train, y_train):
    pipe = Pipeline([('tfidf', TfidfVectorizer()),
                     ('nb', GaussianNB())
                     ])

    parameter = {"tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)]}

    nb = GridSearchCV(pipe, param_grid=parameter, cv=5, verbose=1)
    nb.fit(X_train, y_train)
    print(f"Best o score : {nb.best_score} with {nb.best_params}")
    return nb.best_score, nb.best_params
