import pandas as pd #biblioteca para criação e manipulaçao de tabelas (DB's)
import numpy as np #biblioteca para operações matemáticas e arranajos multidimensionais
import openpyxl


tabela = pd.read_csv('barcos_ref.csv')


import seaborn as sns #bilioteca de gráficos
import matplotlib.pyplot as plt #bilioteca de gráficos
#sns.heatmap(tabela.corr()[['Preco']], cmap='Blues', annot=True)
#plt.show() DUAS linhas de código para teste do progresso

from sklearn.model_selection import train_test_split  #biblioteca para machine learning
y = tabela['Preco'] # início das atribuíções dos parâmetros
x = tabela.drop('Preco', axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LinearRegression #importei dois modelos para testar qual se saíria melhor
from sklearn.ensemble import RandomForestRegressor

modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

from sklearn import metrics #importa a função de métricas para checar a assertividade ds modelos

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste) #realiza os testes nos dois modelos
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

print(f'PRL com assertividade de {metrics.r2_score(y_teste, previsao_regressaolinear)}%')
print(f'ARD com assrtividade de {metrics.r2_score(y_teste, previsao_arvoredecisao)}%')


tabela_auxiliar = pd.DataFrame()  #criado uma tabela auxiliar para guardar os resultados
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['Previsoes Arvore Decisao'] = previsao_arvoredecisao
tabela_auxiliar['Previsoes Regressao Linear'] = previsao_regressaolinear

#sns.lineplot(data=tabela_auxiliar)
#plt.show()  DUAS linhas de código para teste do progresso


tabela_nova = pd.read_csv('novos_barcos.csv') #cria uma nova tabela com os resultados definitivos
previsao = modelo_arvoredecisao.predict(tabela_nova) #o modelo de arvore de decisao se mostrou mais assertivo com 85%. (Os dados são de uma empresa de venda de barcos, está ótimo
#entenda seu problema de negócio, se for necessário peça mais dados, mais tempo para coletar e testar. Ex: 85% não está bom para eficácia de um remédio.
print(tabela_nova)
print(previsao)


