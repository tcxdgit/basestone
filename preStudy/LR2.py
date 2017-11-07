import pickle
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

n_samples = len(X_digits)

X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]
X_test = X_digits[int(.9 * n_samples):]
y_test = y_digits[int(.9 * n_samples):]

logistic = linear_model.LogisticRegression()
logreg = logistic.fit(X_train, y_train)

# save the model to disk
filename = 'LR_model.pkl'
pickle.dump(logreg, open(filename, 'wb'))

# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))

y_pred = logreg.predict(X_test)
probas = logreg.predict_proba(X_test)
probas = probas[:,1]

# model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
fpr, tpr, thresholds = roc_curve(y_test, probas, pos_label=0)
roc_auc_score(y_test, probas)

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print('LogisticRegression score: %f'% logreg.score(X_test, y_test))
coef = logreg.coef_
proba = logreg.predict_log_proba(X_test)
coef2 = logreg.densify()
params = logreg.get_params()

print 'finished'