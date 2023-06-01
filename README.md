# ML_Boosting_Bagging

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/PowerBI-red.svg)](https://powerbi.microsoft.com/pt-br/)

![Logo](https://img.freepik.com/vetores-gratis/conceito-de-transformacao-digital-de-vetor-de-fundo-de-cerebro-de-tecnologia-de-ia_53876-117812.jpg?w=1380&t=st=1685638742~exp=1685639342~hmac=15ce4e608d7e591e0564fbef6de250ffed0278b7f824c006f0afc18ff89f39d2)

Neste repositório, você encontrará um estudo detalhado sobre os algoritmos de machine learning de classificação, especificamente sobre boosting e bagging.

O estudo aborda não apenas a teoria e os princípios dessas técnicas, mas também inclui uma metodologia detalhada, etapas de pré-processamento e limpeza de dados, além de uma conclusão abrangente.

# Metodologia
A metodologia adotada no estudo consistiu em coletar conjuntos de dados relevantes para problemas de classificação e realizar uma análise comparativa dos algoritmos de boosting e bagging. Os algoritmos foram implementados utilizando uma popular biblioteca de machine learning e foram ajustados com base em parâmetros específicos para cada algoritmo.

# Pré-processamento
Antes da aplicação dos algoritmos, as etapas de pré-processamento e limpeza de dados foram realizadas para garantir a qualidade e consistência dos conjuntos de dados. Essas etapas incluíram tratamento de valores ausentes, remoção de outliers, normalização ou padronização dos dados, e codificação de variáveis categóricas, quando necessário. O pré-processamento foi realizado de acordo com as melhores práticas do campo. Após o pré-processamento, os algoritmos de boosting e bagging foram aplicados aos conjuntos de dados. Métricas de avaliação padrão foram utilizadas para comparar o desempenho dos algoritmos, incluindo acurácia, precisão, recall e F1-score. Os resultados foram registrados e analisados em termos de desempenho e robustez dos modelos gerados.

# Conclusão
Por fim, uma conclusão foi tirada com base nos resultados obtidos. Os pontos fortes e limitações de cada algoritmo foram discutidos, assim como as situações em que um algoritmo se mostrou mais adequado do que o outro. Recomendações e direções para pesquisas futuras foram oferecidas com base nas descobertas e nas experiências adquiridas durante o estudo. Espera-se que este repositório seja uma fonte valiosa de informações para aqueles que desejam compreender e aplicar os algoritmos de boosting e bagging em problemas de classificação. A metodologia detalhada, o pré-processamento de dados e a limpeza, bem como a conclusão abrangente, fornecem uma visão completa e prática sobre o assunto. O repositório inclui o código-fonte, documentação e recursos adicionais necessários para reproduzir e expandir o estudo.

# Algoritmos machine learning Bagging Boosting

## Bagging (Bootstrap Aggregating)
Bagging (Bootstrap Aggregating) é uma técnica de ensemble learning em machine learning que combina múltiplos modelos de aprendizado para melhorar a precisão e o desempenho preditivo. O conceito básico do bagging é treinar várias instâncias do mesmo algoritmo de aprendizado em diferentes conjuntos de dados amostrados com reposição dos dados de treinamento originais. Existem vários algoritmos de machine learning que podem ser usados com a técnica de bagging. Aqui estão alguns dos mais comuns:

1) Bagged Decision Trees (Bagging de Árvores de Decisão): Neste algoritmo, várias árvores de decisão são treinadas em diferentes conjuntos de dados de treinamento criados por amostragem com reposição dos dados originais. Cada árvore produz uma previsão e, em seguida, essas previsões são agregadas para tomar uma decisão final.

2) Random Forest (Floresta Aleatória): É uma extensão do bagging de árvores de decisão. Nesse algoritmo, além do bagging, cada árvore de decisão é treinada em um subconjunto aleatório de recursos (variáveis) do conjunto de dados original. Essa aleatoriedade adicional ajuda a reduzir a correlação entre as árvores e a aumentar ainda mais a diversidade do modelo.

3) AdaBoost (Adaptive Boosting): Embora não seja estritamente um algoritmo de bagging, é uma técnica de ensemble learning semelhante que combina várias versões do mesmo algoritmo de aprendizado. No AdaBoost, cada modelo subsequente é treinado em uma versão ponderada do conjunto de dados, onde os pesos são ajustados com base no desempenho dos modelos anteriores. O objetivo é dar mais importância às amostras que foram classificadas incorretamente pelos modelos anteriores.

Esses são alguns dos algoritmos de bagging mais populares em machine learning. Cada um deles tem suas próprias características e propriedades, mas todos eles seguem o conceito básico de combinar múltiplos modelos para melhorar a precisão preditiva.

# Boosting
Boosting é outra técnica de ensemble learning em machine learning que combina vários modelos de aprendizado fracos para criar um modelo forte. Diferente do bagging, onde os modelos são treinados independentemente, no boosting, os modelos são treinados de forma sequencial, onde cada modelo subsequente é ajustado para corrigir os erros cometidos pelos modelos anteriores.

Existem vários algoritmos de boosting, mas aqui estão alguns dos mais comuns:

1) AdaBoost (Adaptive Boosting): É um dos algoritmos de boosting mais conhecidos. No AdaBoost, cada modelo subsequente é treinado em uma versão ponderada do conjunto de dados, onde os pesos são ajustados com base no desempenho dos modelos anteriores. Os modelos subsequentes concentram-se nas amostras que foram classificadas incorretamente pelos modelos anteriores, para tentar corrigir esses erros.

2) Gradient Boosting: É uma família de algoritmos de boosting que inclui algoritmos como Gradient Boosting Machine (GBM), XGBoost e LightGBM. Nesses algoritmos, cada modelo subsequente é treinado para prever os resíduos do modelo anterior. Os resíduos são as diferenças entre as previsões do modelo atual e os valores reais do conjunto de treinamento. O objetivo é minimizar esses resíduos em cada iteração para melhorar o desempenho geral do modelo.

3) Stochastic Gradient Boosting: É uma variante do gradient boosting em que os modelos são treinados em subconjuntos aleatórios dos dados de treinamento. Essa aleatoriedade adiciona mais diversidade ao ensemble e ajuda a evitar overfitting.

Assim como o bagging, o boosting tem como objetivo combinar modelos para obter uma maior precisão e desempenho preditivo. No entanto, a abordagem de boosting difere em como os modelos são treinados e ajustados sequencialmente para corrigir os erros cometidos pelos modelos anteriores.

## Autores

- [@RafaelGallo](https://github.com/RafaelGallo)


## Licença

[MIT](https://choosealicense.com/licenses/mit/)


## Resultados - Dos modelos machine learning 

- Melhorar o suporte de navegadores

- Adicionar mais integrações


## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

`API_KEY`

`ANOTHER_API_KEY`
## Instalação 

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib
  conda install keras
  conda install tensorflow-gpu==2.5.0

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```
## Stack utilizada

**Machine learning:** Python, R

**Framework:** Scikit-learn

**Análise de dados:** Python, R

## Base de dados - Modelos de machine learning

| Dataset               | Link                                                 |
| ----------------- | ---------------------------------------------------------------- |
| | |
| | |

## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

Instalando a virtualenv

`pip install virtualenv`

Nova virtualenv

`virtualenv nome_virtualenv`

Ativando a virtualenv

`source nome_virtualenv/bin/activate` (Linux ou macOS)

`nome_virtualenv/Scripts/Activate` (Windows)

Retorno da env

`projeto_py source venv/bin/activate` 

Desativando a virtualenv

`(venv) deactivate` 

Instalando pacotes

`(venv) projeto_py pip install flask`

Instalando as bibliotecas

`pip freeze`

## Uso/Exemplos - Modelo machine learning

# Instalação

Instalação das bibliotecas para esse projeto no python.

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib
  conda install nltk

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
  nltk==3.6.7
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```
    
## Exemplo Modelo exemplo modelo bagging

```
# Importação das bibliotecas
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregando o conjunto de dados Iris
data = load_iris()
X = data.data
y = data.target

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criação do classificador base (árvore de decisão)
base_classifier = DecisionTreeClassifier()

# Criação do classificador bagging com 10 estimadores (10 árvores de decisão)
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10)

# Treinamento do modelo bagging
bagging_classifier.fit(X_train, y_train)

# Previsões do modelo bagging para os dados de teste
y_pred = bagging_classifier.predict(X_test)

# Avaliação da precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo de Bagging: {:.2f}".format(accuracy))

Exemplo Modelo exemplo modelo Boosting

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregando o conjunto de dados Iris
data = load_iris()
X = data.data
y = data.target

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criação do classificador base (árvore de decisão)
base_classifier = DecisionTreeClassifier(max_depth=1)

# Criação do classificador AdaBoost com 50 estimadores
adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=50)

# Treinamento do modelo AdaBoost
adaboost_classifier.fit(X_train, y_train)

# Previsões do modelo AdaBoost para os dados de teste
y_pred = adaboost_classifier.predict(X_test)

# Avaliação da precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo AdaBoost: {:.2f}".format(accuracy))


```

## Feedback

Se você tiver algum feedback, por favor nos deixe saber por meio de rafaelhenriquegallo@gmail.com.br


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Melhorias

Que melhorias você fez no seu código? Ex: refatorações, melhorias de performance, acessibilidade, etc


## Referência

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Licença

[MIT](https://choosealicense.com/licenses/mit/)


## Suporte

Para suporte, mande um email para rafaelhenriquegallo@gmail.com
