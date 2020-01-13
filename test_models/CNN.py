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


def main():
    # import dataset
    train, test = common_modules.merge_files()
    # clean text
    data_clean = [common_modules.clean_text(text) for text in train.texto]

    #change label -1 por 0, porque modelo obteve uma melhor performance dessa maneira
    data_labels = train.label.values
    data_labels[data_labels == -1] = 0

    # transforma texto em token utilizar tokenizer do tensorflow nesse caso é melhor
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        data_clean, target_vocab_size=2 ** 16
    )

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
    NB_EPOCHS = 10
    # instanciar objeto da arquitetura da rede
    Dcnn = DCNN(vocab_size=VOCAB_SIZE,
                emb_dim=EMB_DIM,
                nb_filters=NB_FILTERS,
                FFN_units=FFN_UNITS,
                nb_classes=NB_CLASSES,
                dropout_rate=DROPOUT_RATE)

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

    test['predict'] = test['texto'].apply(
        lambda x: round(Dcnn(np.array([tokenizer.encode(x)]), training=False).numpy()[0][0]))
    test[test.label == -1] = 0
    test['precision'] = teste['label'] - test['predict']
    acc_teste = len(test[test.precision == 0]) / len(test)
    print(f"Accuracy for all test data: {acc_teste}")


    """
    Mostra o histórico de treino e validação do modelo. Se a val_acc subir no final 
    das epochs é sinal que modelo está overfitando, ideal é que ambos val_acc quanto
    acc fiquem proximas e valores baixos.
    """

    style.use("seaborn-ticks")
    Dcnn.evaluate(train_inputs,
                  train_labels)
    plt.figure(figsize=(12, 8))
    x_axis = np.linspace(1, 10, 10)
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, Dcnn_history.history['accuracy'], Dcnn_history.history['val_accuracy'])
    plt.title('Training metrics')
    plt.ylabel('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(x_axis, Dcnn_history.history['loss'], Dcnn_history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("DCNN_history.png")

if __name__ == "__main__":
    main()
