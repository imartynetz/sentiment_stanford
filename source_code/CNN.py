import pandas as pd
import numpy as np
import common_modules
import pdpipe as pdp
from dcnn_arquitecture import DCNN
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from pathlib import Path

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train_model(train, test):
    p = Path('models')
    if not p.exists():
        p.mkdir()
    # clean text
    data_clean = [common_modules.clean_text(text) for text in train.texto]

    # change label -1 por 0, porque modelo obteve uma melhor performance dessa maneira
    data_labels = train.label.values
    data_labels[data_labels == -1] = 0

    # transforma texto em token utilizar tokenizer do tensorflow nesse caso é melhor
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        data_clean, target_vocab_size=2 ** 16
    )
    # salvando vocabulário
    tokenizer.save_to_file("models/vocab_cnn")

    data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]

    MAX_LEN = max([len(sentence) for sentence in data_inputs])
    # pad vetores, para que todos tenham uma mesma dimensão
    data_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_inputs,
                                                                value=0,
                                                                padding="post",
                                                                maxlen=MAX_LEN)

    # Separar conjunto de dados em 80% treino e 20% validação.
    valid = np.random.randint(0, 12500, 2500)
    valid = np.concatenate((valid, valid + 12500))
    valid_inputs = data_inputs[valid]
    valid_labels = data_labels[valid]
    train_inputs = np.delete(data_inputs, valid, axis=0)
    train_labels = np.delete(data_labels, valid)

    # setar parâmetros da rede
    VOCAB_SIZE = tokenizer.vocab_size
    EMB_DIM = 200
    NB_FILTERS = 100
    FFN_UNITS = 256
    NB_CLASSES = len(set(train.label))
    DROPOUT_RATE = 0.4
    BATCH_SIZE = 10
    NB_EPOCHS = 5

    # instanciar objeto da arquitetura da rede
    Dcnn = get_model(VOCAB_SIZE, EMB_DIM, NB_FILTERS, FFN_UNITS, NB_CLASSES, DROPOUT_RATE)

    Dcnn.compile(loss="binary_crossentropy",
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                 metrics=["accuracy"])
    # fitar modelo e salvar historico.
    Dcnn_history = Dcnn.fit(train_inputs,
                            train_labels,
                            batch_size=BATCH_SIZE,
                            epochs=NB_EPOCHS,
                            validation_data=(valid_inputs, valid_labels))

    train['predict'] = train['texto'].apply(
        lambda x: round(Dcnn(np.array([tokenizer.encode(x)]), training=False).numpy()[0][0]))
    train['precision'] = train['label'] - train['predict']
    acc_train = len(train[train.precision == 0]) / len(train)
    print(f"Accuracy for all train data: {acc_train}")
    Dcnn.save_weights('models/dcnn_weights', save_format='tf')

    test['predict'] = test['texto'].apply(
        lambda x: round(Dcnn(np.array([tokenizer.encode(x)]), training=False).numpy()[0][0]))
    test[test.label == -1] = 0
    test['precision'] = test['label'] - test['predict']
    acc_teste = len(test[test.precision == 0]) / len(test)
    print(f"Accuracy for all test data: {acc_teste}")


def predict_model(texto):
    p = Path('models')
    if not p.exists():
        p.mkdir()
    try:
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file("models/vocab_cnn")
        VOCAB_SIZE = tokenizer.vocab_size
        EMB_DIM = 200
        NB_FILTERS = 100
        FFN_UNITS = 256
        NB_CLASSES = 2
        DROPOUT_RATE = 0.4
        BATCH_SIZE = 10
        NB_EPOCHS = 5
        model = get_model(VOCAB_SIZE, EMB_DIM, NB_FILTERS, FFN_UNITS, NB_CLASSES, DROPOUT_RATE)
        model.compile(loss="binary_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                     metrics=["accuracy"])
        model.load_weights('models/dcnn_weights')

    except:
        raise TypeError("Modelo CNN não foi treinado ainda.")

    predict = round(model(np.array([tokenizer.encode(texto)]), training=False).numpy()[0][0])
    if predict == 0:
        predict = -1

    print("\n")
    print("=" * 60)
    print(f"Polaridade Predita: {predict}")
    print("=" * 60)
    print("\n")


def get_model(VOCAB_SIZE, EMB_DIM, NB_FILTERS, FFN_UNITS, NB_CLASSES, DROPOUT_RATE):
    return DCNN(vocab_size=VOCAB_SIZE,
                emb_dim=EMB_DIM,
                nb_filters=NB_FILTERS,
                FFN_units=FFN_UNITS,
                nb_classes=NB_CLASSES,
                dropout_rate=DROPOUT_RATE)
