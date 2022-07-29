import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer 
from decisionTree import decisionTreeTest
from randomForest import randomForestTest

urlTest = "https://raw.githubusercontent.com/zahangirbd/medical_data_for_classification/master/data/Hepatitis/hepatitis.data.test.csv"

urlTrain = "https://raw.githubusercontent.com/zahangirbd/medical_data_for_classification/master/data/Hepatitis/hepatitis.data.train.csv"

def main(model):
  #loading csv
  trainDf = pd.read_csv(urlTrain)
  testDf = pd.read_csv(urlTest)
  trainDf = pd.concat([trainDf,testDf], ignore_index = True)
  #replacing invalid values with NaN
  trainDf = trainDf.replace("?", np.nan)
  #saving column names
  columns = trainDf.columns.tolist()
  #loading imputer to fill NaN values with actual data
  imputer = KNNImputer(n_neighbors=5)
  trainDf = imputer.fit_transform(trainDf)
  trainDf = pd.DataFrame(trainDf, columns = columns)
  
  #plotting mean value and standard deviation for each column
  print("Media e Desvio padrão dos dados")
  metaDf = pd.concat([trainDf.mean(),trainDf.std()], axis=1,join="inner")
  metaDf.columns = ["mean","deviation"]
  print(metaDf.to_string())
  print()

  

  
  if(model =="tree"):
    decisionTreeTest(trainDf)
  elif(model == "rf"):
    randomForestTest(trainDf)
  elif(model == "rf_sqrt"):
    randomForestTest(trainDf,"sqrt")
    
  plt.show()

if __name__ == "__main__":
  print("Bem-Vindo ao meu Projeto 1 de FSI")
  print()
  while True:
    print("Escolha uma opcao para definir qual voce gostaria de analisar:")
    print("1 - Arvore de Decisao")
    print("2 - Random forest com os parametros padrao")
    print("3 - Random forest com as raizes dos parametros")
    print("EXIT - Encerrar a execucao do programa")
    print()
    input = input(">")
    
    if (input.lower() == "exit",):
      break
    elif(input == "1"):
      main("tree")
    elif(input == "2"):
      main("rf")
    elif(input == "3"):
      main("rf_sqrt")
    else:
      os.system('cls' if os.name == 'nt' else 'clear')
      print("Erro: Entrada invalida, selecione uma opcao valida")
