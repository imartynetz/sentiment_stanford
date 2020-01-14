import pandas as pd
import numpy as np
import common_modules
import pdpipe as pdp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
import regressao_logistica
import naive_bayes



def main():
    print("Qual modelo quer treinar?")
    value = input(
        "1: Regressão Logistica, 2: Multinomial Naive Bayes, 3: CNN, digite o numero correspondente ao modelo:")

    # import dataset
    train, test = common_modules.merge_files()

    if value =='1':
        pipeline = pdp.ApplyByCols("texto", common_modules.tag_remove, "clean_texto", drop=False)
        pipeline += pdp.ApplyByCols("clean_texto", common_modules.trat_texto)
        train = pipeline(train)
        test = pipeline(test)
        print("Treinando modelo de regressão logistica")
        regressao_logistica.train_model(train.texto, train.label, test.texto, test.label)
    if value =='2':
        pipeline = pdp.ApplyByCols("texto", common_modules.tag_remove, "clean_texto", drop=False)
        pipeline += pdp.ApplyByCols("clean_texto", common_modules.trat_texto)
        train = pipeline(train)
        test = pipeline(test)
        print("treinando modelo de Multinomial Naive Bayes")
        naive_bayes.train_model(train.texto, train.label, test.texto, test.label)
    if value =='3':
        import CNN
        print("treinando modelo de CNN")
        CNN.train_model(train, test)


if __name__ == "__main__":
    main()
