from sklearn.metrics import recall_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
recall_score(y_true, y_pred, average='macro')
# 0.33
recall_score(y_true, y_pred, average='micro')
# 0.33
recall_score(y_true, y_pred, average='weighted')
# 0.33
recall_score(y_true, y_pred, average=None)
# array([ 1.,  0.,  0.])
