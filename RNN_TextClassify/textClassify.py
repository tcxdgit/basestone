# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout,Activation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import csv, datetime

np.random.seed(1337)  # for reproducibility
anum=13

def load_data(original_data = None, train_size = 0.7, m = None):

    with open(original_data, 'r') as g:
        read = csv.reader(g)
        for item in read:
            a = item
        dataset = np.array(a, dtype='float32')
        g.close()

    # dataset = np.transpose(dataset)
    num_data = dataset.shape[0]
    num_train = round(0.7*num_data)
    num_train = int(num_train)
    num_test = round(0.3*num_data)
    num_test = int(num_test)

    # Data scaling
    min_max_scaler = MinMaxScaler()
    data_n = min_max_scaler.fit_transform(dataset)

    # Construct input-output
    Xn_train = np.zeros((num_train - m, m))
    for i in range(num_train - m):
        a = data_n[i:m + i].copy()
        Xn_train[i] = np.transpose(a)

    Yn_train = np.zeros((num_train - m))
    for i in range(num_train - m):
        Yn_train[i] = data_n[m + i].copy()

    miss_num = (10 - num_test % 10)
    if (miss_num == 10):
        Xn_test = np.zeros((num_test, m))
        for i in range(num_test):
            a = data_n[num_train - m + i:num_train + i].copy()
            Xn_test[i] = np.transpose(a)
    else:
        miss_point = np.zeros(miss_num)
        data_n = np.append(data_n, miss_point)
        Xn_test = np.zeros((num_test+miss_num, m))
        for i in range(num_test+miss_num):
            a = data_n[num_train - m + i:num_train + i].copy()
            Xn_test[i] = np.transpose(a)

    Y_test = dataset[num_train:]

    #return Xn_train, Yn_train, Xn_test, Y_test, min_max_scaler
    return Xn_train[:1490], Yn_train[:1490], Xn_test, Y_test, min_max_scaler, miss_num
    # return Xn_train[:1490], Yn_train[:1490], Xn_test[:640], Y_test[:640], min_max_scaler
    #return Xn_train[:1510], Yn_train[:1510], Xn_test[:650], Y_test[:650], min_max_scaler
    #return Xn_train[:1530], Yn_train[:1530], Xn_test[:660], Y_test[:660], min_max_scaler
    #return Xn_train[:1530], Yn_train[:1530], Xn_test[:660], Y_test[:660], min_max_scaler

Xn_train, Yn_train, Xn_test, Y_test, min_max_scaler, miss_num = load_data(original_data='data1.csv', train_size=0.7, m=anum)

Xn_train = Xn_train.reshape(len(Xn_train), anum, 1)
Yn_train = Yn_train.reshape(len(Yn_train), 1)
Xn_test = Xn_test.reshape(len(Xn_test), anum, 1)

d1 = datetime.datetime.now()
model = Sequential()

model.add(LSTM(300, return_sequences=False,batch_input_shape=(10,anum,1),stateful=True))
#model.add(LSTM(300, return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(300, return_sequences=False))
#model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(Xn_train, Yn_train, nb_epoch=75, batch_size=10, verbose=2)

Yn_pred = model.predict(Xn_test,batch_size=10)
# Yn_pred = model.predict(Xn_test)
Yn_pred = Yn_pred.ravel()
Y_pred = min_max_scaler.inverse_transform(Yn_pred)
Y_pred = Y_pred[:(-miss_num)]

R = np.corrcoef(Y_test, Y_pred).min()
RMSE = np.sqrt(np.mean((Y_test - Y_pred)**2))
d2 = datetime.datetime.now()
s = (d2 - d1).seconds
print('R: %f\nRMSE: %f\nspend: %f' % (R, RMSE,s))

pd.DataFrame({'pred': Y_pred,'true': Y_test},columns=['pred', 'test']).plot()
# plt.savefig('comparision.png')
plt.show()

shaped_y = np.reshape(Y_pred,(np.product(Y_pred.shape),1))
#R: 0.975967 RMSE: 66.827560
#72.3 110 76.8 43.2
np.savetxt("Y_pred1.csv", shaped_y, delimiter=",")