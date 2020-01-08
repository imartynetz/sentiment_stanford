import pandas as pd
import numpy as np
import common_modules
import pdpipe as pdp
from logistic_regression import reg_log
from svm import svm
from naive_bayes import naive_bayes


def main():
    train, test = common_modules.merge_files()

    pipeline = pdp.ApplyByCols("texto", common_modules.tag_remove, "clean_texto", drop=False)
    pipeline += pdp.ApplyByCols("clean_texto", common_modules.trat_texto)
    train = pipeline(train)
    #test = pipeline(test)

    scores = pd.DataFrame()
    print("Fazendo gridSearch da Regressão logistica")
    score_log_reg, param_log_reg = reg_log(train.texto, train.label)
    scores.loc[0, 'model'] = "Regressão Logistica"
    scores.loc[0, 'score'] = score_log_reg
    scores.loc[0, 'parâmetro'] = param_log_reg

    print("Fazendo gridSearch da SVM")
    score_svm, param_svm = svm(train.texto, train.label)
    scores.loc[0, 'model'] = "SVM"
    scores.loc[0, 'score'] = score_svm
    scores.loc[0, 'parâmetro'] = param_svm

    print("Fazendo gridSearch da Naive bayes")
    score_nb, param_nb = naive_bayes(train.texto, train.label)
    scores.loc[0, 'model'] = "Naive Bayes"
    scores.loc[0, 'score'] = score_nb
    scores.loc[0, 'parâmetro'] = param_nb
    scores.to_csv('ml_parameter.csv')


if __name__ == "__main__":
    main()
