"""
script contendo somente as funções em comum para os modelos.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])


def merge_files():
    """
    Função para importar somente os comentários positivos
    :return: lista contendo todos textos positivos
    """
    path = Path("../files")
    text_train_pos = []
    text_train_neg = []
    text_test_pos = []
    text_test_neg = []
    for e in path.rglob('*/*/*.txt'):
        content = e.read_text()
        if e.parts[2] == 'train':
            if e.parts[3] == 'pos':
                text_train_pos.append(content)
            elif e.parts[3] == 'neg':
                text_train_neg.append(content)
        elif e.parts[2] == 'test':
            if e.parts[3] == 'pos':
                text_test_pos.append(content)
            elif e.parts[3] == 'neg':
                text_test_neg.append(content)
    #print(text_train_pos)
    train_pos = pd.DataFrame({"texto": text_train_pos, "label": 1})
    train_neg = pd.DataFrame({"texto": text_train_neg, "label": -1})
    test_pos = pd.DataFrame({"texto": text_test_pos, "label": 1})
    test_neg = pd.DataFrame({"texto": text_test_neg, "label": -1})
    data_train = [train_pos, train_neg]
    data_test = [test_pos, test_neg]
    train = pd.concat(data_train, ignore_index=True)
    test = pd.concat(data_test, ignore_index=True)
    return train, test


def tag_remove(texto):
    """
    função para remover as tags html do texto
    :param texto: texto que se quer tratar
    :return: texto com todos os tag removido
    """
    # remove qualquer coisa (incluindo) entre < >
    texto = re.sub(r'<.*?>', '', texto)
    return texto


def trat_texto(texto):
    """
    Tratamento do texto como remoção de pontuação, números, espaço extra e stopwords
    :param texto: texto a ser tratado
    :return: texto tokenizado e tratado
    """
    texto = texto.lower() # <- texto em lower case
    # Lemma e remoção de stopwords, punct e espaços em brancos.
    texto = [token.lemma_ for token in nlp(texto) if not (token.is_stop | token.is_punct | token.is_space | token.like_num)]
    return texto
