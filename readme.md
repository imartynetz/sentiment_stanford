# Sentimental Análise usando Stanford Dataset

Repositório com scrips para análise de sentimento usando dataset do Stanford. Um relatório em pdf está anexado no diretório 
para melhor informação de passos que foram feitos, e escolhas

## Getting Started

### Rodar Local
Projeto foi todo feito em python

Clone o repositório git e instale os requerimento

```
git clone https://github.com/imartynetz/sentiment_stanford
cd sentiment_stanford
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Estrutura dos diretórios
Os diretórios foram organizados da seguinte forma.

* Diretório test_model/gridsearch está todos scripts usado para fazer gridsearch dos modelos.
* Diretório test_model/performance está todos scripts usado para determinar acc dos modelos utilizando os hiperparâmetro 
    determinado pelo gridsearch.
* Diretório files contém todos arquivos de texto para treinar, dentro desse diretório tem os diretórios train e teste
    cada um contendo os diretórios pos e neg, que é aonde está localizado todos arquivos de treino e teste com polaridade
    positiva e negativa.
* Diretório source_code estão os arquivos que serão utilizados para treinar modelos e predizer modelo

## Como utilizar essa biblioteca
Para treinar primeiramente é necessário colocar na pasta files os arquivos para treino (repositório já tem todos arquivos
para treino e teste, mas caso queira mudar arquivos basta substituir), após isso basta fazer
```
cd source_code/
python train_model.py
```
Será pedido para escolher qual modelo quer treinar. Digitando 1 ele treinara um modelo de regressão logistica, digitando 2
ele treinará o modelo de Naive Bayes, e digitando 3 treinará um modelo de CNN, em todos casos após o treinamento será 
informado qual acurácia de treino e teste, bem como matriz de confusão e F1 score o modelo obteve (esse ultimo somente 
para regressão logistica e Naive Bayes).

Uma vez treinado o modelo, basta fazer
```
python predict_text.py
```
Uma janela aparecerá para selecionar o arquivo de texto que queira saber a polaridade. O arquivo de texto deve estar em 
formato .txt e conter somente um texto.

Depois de selecionado o texto será perguntado qual modelo quer utilizar para treinar o modelo, mesma coisa do treino, 
digitando 1 ele usará modelo treinado de regressão logistica, 2 usará Naive Bayes e 3 usará CNN.

Caso não queira rodar os treinos, disponibilizo os modelos que treinei em

https://drive.google.com/open?id=10DKR71tdEYLk9ZJW1BRa_d6IZrKMrNlV 

Eles devem ser colocados dentro da pasta source_code

# Troubleshooting
O treinamento do modelo de CNN poderá ocasionar problemas dependendo da sua máquina, uma vez que ele foi feito usando
tensorflow 2.x, ele tende a utilizar a GPU, porém só funciona para placas de vídeo da NVIDIA, e caso tenha problemas para 
fazer ele funcionar usando a GPU o link a seguir explica como instalar.

https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-ubuntu-version-4260a52dd7b0

Requer  Cuda 10.0 e CuDnn 7.

O arquivo relatório.pdf é um breve relatório de qual procedimento utilizado, e porque foi utilizado.