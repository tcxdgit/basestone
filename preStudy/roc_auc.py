'''
Compute Area Under the Curve (AUC) from prediction scores
'''
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)
roc_auc_score(y, scores)

