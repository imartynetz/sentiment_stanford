import pandas as pd
import numpy as np
import common_modules
import pdpipe as pdp
import regressao_logistica
import naive_bayes
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from pathlib import Path


def main():
    print("Selecione o texto que quer predizer.")
    root = Tk()
    root.withdraw()
    root.update()
    filename = askopenfilename()
    print(filename)
    path = Path(filename)
    content = path.read_text()

    print("Com qual modelo quer predizer?")
    value = input(
        "1: Regressão Logistica, 2: Multinomial Naive Bayes, 3: CNN, digite o numero correspondente ao modelo:")
    print("\n")
    print("="*60)
    print(f"Texto : {content}")
    print("=" *60)
    if value == "1":
        regressao_logistica.predict_model(content)
    if value == "2":
        naive_bayes.predict_model(content)
    if value == "3":
        import CNN
        CNN.predict_model(content)


if __name__ == "__main__":
    main()
