import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score

def rocCurves(classifier, X, y, cv):
  bestf1 = 0
  bestclf = None
  
  tprs = []
  aucs = []
  actClas = np.array([],dtype = int)
  predClas = np.array([],dtype = int)
  mean_fpr = np.linspace(0, 1, 100)
  
  fig, ax = plt.subplots()
  #cross validation and graph information
  for i, (train, test) in enumerate(cv.split(X, y)):
    #training the classifier
    classifier.fit(X.iloc[train], y.iloc[train])
    #roc auc stuff
    viz = RocCurveDisplay.from_estimator(
      classifier,
      X.iloc[test],
      y.iloc[test],
      name="ROC fold {}".format(i),
      alpha=0.3,
      lw=1,
      ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    #confusion matrix stuff
    aux = f1_score( y.iloc[test],classifier.predict(X.iloc[test]))
    if ( aux > bestf1):
      bestclf = classifier
      bestf1 = aux
    
    actClas = np.append(actClas, y.iloc[test])
    predClas = np.append( predClas,classifier.predict(X.iloc[test]))
  
  #plotting stuff
  ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  ax.plot(
      mean_fpr,
      mean_tpr,
      color="b",
      label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
      lw=2,
      alpha=0.8,
  )
  
  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  ax.fill_between(
      mean_fpr,
      tprs_lower,
      tprs_upper,
      color="grey",
      alpha=0.2,
      label=r"$\pm$ 1 std. dev.",
  )
  
  ax.set(
      xlim=[-0.05, 1.05],
      ylim=[-0.05, 1.05],
      title="Receiver operating characteristic example",
  )
  ax.legend(loc="lower right")
  
  ConfusionMatrixDisplay.from_predictions(actClas, predClas)
  
  return bestclf
  
