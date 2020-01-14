import pandas as pd
import numpy as np
import common_modules
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def sup_vec(X_train, y_train):
    pipe = Pipeline([('tfidf', TfidfVectorizer()),
                     ('svm', svm(random_state=42))
                     ])

    parameter = {"tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                 "svm__kernel": ["linear", "poly", "rbf", "sigmoid"],
                 "svm__C": np.logspace(-3, 3, 7)
                 }
    model = GridSearchCV(pipe, param_grid=parameter, cv=5, verbose=1)
    model.fit(X_train, y_train)
    print(f"Best o score : {model.best_score_} with {model.best_params_}")
    return model.best_score_, model.best_params_
