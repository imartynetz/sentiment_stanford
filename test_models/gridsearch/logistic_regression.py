import pandas as pd
import numpy as np
import common_modules
import pdpipe as pdp
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def reg_log(X_train, y_train):
    pipe = Pipeline([('tfidf', TfidfVectorizer()),
                     ('logreg', LogisticRegression(random_state=42, n_jobs = -1))
                     ])

    parameter = {"tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                 "logreg__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                 "logreg__C": np.logspace(-3, 3, 7)
                 }
    model = GridSearchCV(pipe, param_grid=parameter, cv=5, verbose=1)
    model.fit(X_train, y_train)
    print(f"Best o score : {model.best_score_} with {model.best_params_}")
    return model.best_score_, model.best_params_
