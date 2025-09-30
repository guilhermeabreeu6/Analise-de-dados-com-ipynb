# Projeto Python IA: Inteligência Artificial e Previsões

### Case: Score de Crédito dos Clientes

Você foi contratado por um banco para conseguir definir o score de crédito dos clientes. Você precisa analisar todos os clientes do banco e, com base nessa análise, criar um modelo que consiga ler as informações do cliente e dizer automaticamente o score de crédito dele: Ruim, Ok, Bom

#passo a passo
# Passo a passo
# Passo 0 - Entender a empresa e o desafio da empresa
# Passo 1 - Importar a base de dados
import pandas as pd
tabela = pd.read_csv("clientes.csv")

display(tabela)
# Score de crédito = Nota de crédito
# Good = Boa
# Standard = OK
# Poor = Ruim

id_cliente	mes	idade	profissao	salario_anual	num_contas	num_cartoes	juros_emprestimo	num_emprestimos	dias_atraso	...	idade_historico_credito	investimento_mensal	comportamento_pagamento	saldo_final_mes	score_credito	emprestimo_carro	emprestimo_casa	emprestimo_pessoal	emprestimo_credito	emprestimo_estudantil
0	3392	1	23.0	cientista	19114.12	3.0	4.0	3.0	4.0	3.0	...	265.0	21.465380	alto_gasto_pagamento_baixos	312.494089	Good	1	1	1	1	0
1	3392	2	23.0	cientista	19114.12	3.0	4.0	3.0	4.0	3.0	...	266.0	21.465380	baixo_gasto_pagamento_alto	284.629162	Good	1	1	1	1	0
2	3392	3	23.0	cientista	19114.12	3.0	4.0	3.0	4.0	3.0	...	267.0	21.465380	baixo_gasto_pagamento_medio	331.209863	Good	1	1	1	1	0
3	3392	4	23.0	cientista	19114.12	3.0	4.0	3.0	4.0	5.0	...	268.0	21.465380	baixo_gasto_pagamento_baixo	223.451310	Good	1	1	1	1	0
4	3392	5	23.0	cientista	19114.12	3.0	4.0	3.0	4.0	6.0	...	269.0	21.465380	alto_gasto_pagamento_medio	341.489231	Good	1	1	1	1	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
99995	37932	4	25.0	mecanico	39628.99	4.0	6.0	7.0	2.0	23.0	...	378.0	24.028477	alto_gasto_pagamento_alto	479.866228	Poor	1	0	0	0	1
99996	37932	5	25.0	mecanico	39628.99	4.0	6.0	7.0	2.0	18.0	...	379.0	24.028477	alto_gasto_pagamento_medio	496.651610	Poor	1	0	0	0	1
99997	37932	6	25.0	mecanico	39628.99	4.0	6.0	7.0	2.0	27.0	...	380.0	24.028477	alto_gasto_pagamento_alto	516.809083	Poor	1	0	0	0	1
99998	37932	7	25.0	mecanico	39628.99	4.0	6.0	7.0	2.0	20.0	...	381.0	24.028477	baixo_gasto_pagamento_alto	319.164979	Standard	1	0	0	0	1
99999	37932	8	25.0	mecanico	39628.99	4.0	6.0	7.0	2.0	18.0	...	382.0	24.028477	alto_gasto_pagamento_medio	393.673696	Poor	1	0	0	0	1
100000 rows × 25 columns
#passo 2 - prepar a base de dados

#int - numérico inteiro
#float - numérico decimal
#object - texto (string)

#labelEncoder - transformar texto em números
from sklearn.preprocessing import LabelEncoder

codificador1 = LabelEncoder()

#profissões
# cientista - 1
# bombeiro - 2
# engenheiro - 3
# dentista - 4
# artista - 5
tabela["profissao"] = codificador1.fit_transform(tabela["profissao"])


codificador2 = LabelEncoder()
tabela["mix_credito"] = codificador2.fit_transform(tabela["mix_credito"])

codificador3 = LabelEncoder()
tabela["comportamento_pagamento"] = codificador3.fit_transform(tabela["comportamento_pagamento"])

display(tabela.info())

#   Column                    Non-Null Count   Dtype  
---  ------                    --------------   -----  
 0   id_cliente                100000 non-null  int64  
 1   mes                       100000 non-null  int64  
 2   idade                     100000 non-null  float64
 3   profissao                 100000 non-null  int64  
 4   salario_anual             100000 non-null  float64
 5   num_contas                100000 non-null  float64
 6   num_cartoes               100000 non-null  float64
 7   juros_emprestimo          100000 non-null  float64
 8   num_emprestimos           100000 non-null  float64
 9   dias_atraso               100000 non-null  float64
 10  num_pagamentos_atrasados  100000 non-null  float64
 11  num_verificacoes_credito  100000 non-null  float64
 12  mix_credito               100000 non-null  int64  
 13  divida_total              100000 non-null  float64
 14  taxa_uso_credito          100000 non-null  float64
 15  idade_historico_credito   100000 non-null  float64
 16  investimento_mensal       100000 non-null  float64
 17  comportamento_pagamento   100000 non-null  int64  
 18  saldo_final_mes           100000 non-null  float64
 19  score_credito             100000 non-null  object 
...
 23  emprestimo_credito        100000 non-null  int64  
 24  emprestimo_estudantil     100000 non-null  int64  
dtypes: float64(14), int64(10), object(1)

# y -> é a coluna da base de dados que eu quero prever
y = tabela["score_credito"]

# x -> as colunas da base de dados que eu vou usar pra fazer a previsão
x = tabela.drop(columns=["score_credito", "id_cliente"])

# separar em dados de treino e dados de teste
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Passo 3 - Treinar a Inteligência Artificial -> 
# Criar o modelo: Nota de crédito: Boa, Ok, Ruim

# Arvore de Decisão -> RandomForest
# Nearest Neighbors -> KNN -> Vizinhos Próximos

# importar a IA (Inteligencia Artificial)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# criar a IA
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

# treinar a IA
modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)
revisao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

#acurracy
from sklearn.metrics import accuracy_score

display(accuracy_score(y_teste, previsao_arvoredecisao))
display(accuracy_score(y_teste, previsao_knn))
0.8272666666666667
0.7385333333333334

#fazer novas previsões
#modelo_arvoredecisao
tabela_nova = pd.read_csv("novos_clientes.csv")
display(tabela_nova)

tabela_nova["profissao"] = codificador1.fit_transform(tabela_nova["profissao"])


codificador2 = LabelEncoder()
tabela_nova["mix_credito"] = codificador2.fit_transform(tabela_nova["mix_credito"])

codificador3 = LabelEncoder()
tabela_nova["comportamento_pagamento"] = codificador3.fit_transform(tabela_nova["comportamento_pagamento"])

previsao = modelo_arvoredecisao.predict(tabela_nova)
display(previsao)
mes	idade	profissao	salario_anual	num_contas	num_cartoes	juros_emprestimo	num_emprestimos	dias_atraso	num_pagamentos_atrasados	...	taxa_uso_credito	idade_historico_credito	investimento_mensal	comportamento_pagamento	saldo_final_mes	emprestimo_carro	emprestimo_casa	emprestimo_pessoal	emprestimo_credito	emprestimo_estudantil
0	1	31.0	empresario	19300.340	6.0	7.0	17.0	5.0	52.0	19.0	...	29.934186	218.0	44.50951	baixo_gasto_pagamento_baixo	312.487689	1	1	0	0	0
1	4	32.0	advogado	12600.445	5.0	5.0	10.0	3.0	25.0	18.0	...	28.819407	12.0	0.00000	baixo_gasto_pagamento_medio	300.994163	0	0	0	0	1
2	2	48.0	empresario	20787.690	8.0	6.0	14.0	7.0	24.0	14.0	...	34.235853	215.0	0.00000	baixo_gasto_pagamento_alto	345.081577	0	1	0	1	0
3 rows × 23 columns

