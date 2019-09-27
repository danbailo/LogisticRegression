# Trabalho 2 - Inteligência Artificial

## Descrição
Para esta tarefa utilizaremos o dataset Wine Quality Data Set (red), disponível em: http://archive.ics.uci.edu/ml/datasets/Wine+Quality, relacionado ao artigo: [P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties.](https://linkinghub.elsevier.com/retrieve/pii/S0167923609001377)

A tarefa consiste em implementar uma regressão linear para predizer a qualidade de um determinado vinho tinto com base em suas características

As características disponíveis são: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.

Inicialize os valores de θ com 0 e teste valores variados para α e quantidade de iterações (atualizações dos valores de θ) e plote os valores de J.

*O código fonte do programa está disponibilizado num arquivo Python.*

---
## Requisitos

* `python3`
* `pip3`
  
---
## Dependências

Para instalar as dependências, execute os comandos abaixo num terminal/prompt de comando:

* Linux
  * `python3 -m pip install -r requirements.txt --user`

* Windows
  * `python -m pip install -r requirements.txt`

---
## Como usar

Para executar o programa, ainda com o terminal/prompt de comando aberto, siga os passos abaixo:
* `cd src/`
  * Linux
  * `python3 script.py`


  * Windows
  * `python script.py`

---
## Output

A saída do script serão os gráficos de J, as imagens serão salvas no diretório `imgs/`.
