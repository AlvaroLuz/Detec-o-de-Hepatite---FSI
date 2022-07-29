from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from rocCurves import rocCurves
import pandas as pd
# Run classifier with cross-validation and plot ROC curves
def decisionTreeTest(trainDf):
  #defining classifier and cross-validation models
  classifier = DecisionTreeClassifier(random_state=0)
  cv = StratifiedKFold(n_splits=10)
  #splitting between classification data and the labels
  y = trainDf['CLASS']
  X = trainDf.drop('CLASS', axis=1)

  #training, testing and plotting curves and confusion matrix
  bestclf = rocCurves(classifier, X, y, cv)

  #showing most relevant parameters
  print("Parametros mais relevantes para o DecisionTree e importancia media correspondente")
  importances = bestclf.feature_importances_
  forest_importances = pd.Series(importances, index=X.columns)
  forest_importances = forest_importances.sort_values(ascending= False)
  print(forest_importances[0:2].to_string())