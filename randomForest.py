from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from rocCurves import rocCurves
import pandas as pd
# Run classifier with cross-validation and plot ROC curves
def randomForestTest(trainDf, features = None):
  #defining classifier and cross validation models
  classifier = RandomForestClassifier(random_state=0, max_features=features)
  cv = StratifiedKFold(n_splits=10)
  #splitting the data and the labels
  y = trainDf['CLASS']
  X = trainDf.drop('CLASS', axis=1)

  #training, testing and plotting curve and confusion matrix
  bestclf = rocCurves(classifier, X, y, cv)

  #showing most relevant parameters
  print("Parametros mais relevantes para o Random Forest e importancia media correspondente")
  importances = bestclf.feature_importances_
  forest_importances = pd.Series(importances, index=X.columns)
  forest_importances = forest_importances.sort_values(ascending= False)
  print(forest_importances[0:2].to_string())