import pandas as pd
import numpy as np
import common_modules
import pdpipe as pdp
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def naive_bayes(X_train, y_train):
    pipe = Pipeline([('tfidf', TfidfVectorizer()),
                     ('nb', MultinomialNB())
                     ])

    parameter = {'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2)],
                 'tfidf__use_idf': (True, False),
                 'tfidf__norm': ('l1', 'l2'),
                 'nb__alpha': [1, 1e-1, 1e-2]}

    nb = GridSearchCV(pipe, param_grid=parameter, cv=5, verbose=1)
    nb.fit(X_train, y_train)
    print(f"Best o score : {nb.best_score_} with {nb.best_params_}")
    return nb.best_score_, nb.best_params_
