import pandas as pd
import numpy as np
import common_modules
import pdpipe as pdp
from logistic_regression import reg_log
from svm import sup_vec
from naive_bayes import naive_bayes


def main():
    train, test = common_modules.merge_files()

    pipeline = pdp.ApplyByCols("texto", common_modules.tag_remove, "clean_texto", drop=False)
    pipeline += pdp.ApplyByCols("clean_texto", common_modules.trat_texto)
    train = pipeline(train)
    #test = pipeline(test)


    print("Fazendo gridSearch da Regressão logistica")
    score_log_reg, param_log_reg = reg_log(train.texto, train.label)
    print(f"Regressão linear best score {score_log_reg}, com parâmetros {param_log_reg} ")
    print("Fazendo gridSearch da SVM")
    score_svm, param_svm = sup_vec(train.texto, train.label)
    print(f"suport vector best score {score_svm}, com parâmetros {param_svm} ")
    print("Fazendo gridSearch da Naive bayes")
    score_nb, param_nb = naive_bayes(train.texto, train.label)
    print(f"Multinomial Naive bayes best score {score_nb}, com parâmetros {param_nb} ")


if __name__ == "__main__":
    main()
