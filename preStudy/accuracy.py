import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
# 0.5
accuracy_score(y_true, y_pred, normalize=False)
# 2