from sklearn.metrics import precision_score
y_true = [1, 2, 0, 1, 2, 0]
y_pred = [2, 1, 0, 0, 1, 0]
precision_score(y_true, y_pred, average='macro')
# 0.22
precision_score(y_true, y_pred, average='micro')
# 0.33
precision_score(y_true, y_pred, average='weighted')
# 0.22
precision = precision_score(y_true, y_pred, average=None)
# array([ 0.66...,  0.        ,  0.        ])

print precision
print 'success'
