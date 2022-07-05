from model.AdaBoost import AdaBoost
import pandas as pd
import numpy as np

X = pd.read_csv('./data/data.csv', header = None).to_numpy()
Y = pd.read_csv('./data/targets.csv', header = None).to_numpy().flatten()

X = (X - np.mean(X, axis = 0)) / np.var(X, axis = 0)



for i in range(10):
    sub_X1 = X[:368 * i]
    sub_Y1 = Y[:368 * i]
    sub_X2 = X[386 * (1 + i):]
    sub_Y2 = Y[386 * (1 + i):]
    train_X = np.r_[sub_X1, sub_X2]
    train_Y = np.r_[sub_Y1, sub_Y2]
    test_X = X[368 * i:368 * (i + 1)]
    model = AdaBoost(1)
    model.train(train_X, train_Y)
    pred = model.predict(test_X)
    pred = pd.DataFrame(pred, columns = {'test'}, index = np.arange(368 * i, 368 * (i + 1)))
    pred.to_csv('./experiments/base1_fold%d.csv' % (i + 1), header = None)
    model = AdaBoost(5)
    model.train(train_X, train_Y)
    pred = model.predict(test_X)
    pred = pd.DataFrame(pred, columns = {'test'}, index = np.arange(368 * i, 368 * (i + 1)))
    pred.to_csv('./experiments/base5_fold%d.csv' % (i + 1), header = None)
    model = AdaBoost(10)
    model.train(train_X, train_Y)
    pred = model.predict(test_X)
    pred = pd.DataFrame(pred, columns = {'test'}, index = np.arange(368 * i, 368 * (i + 1)))
    pred.to_csv('./experiments/base10_fold%d.csv' % (i + 1), header = None)
    model = AdaBoost(100)
    model.train(train_X, train_Y)
    pred = model.predict(test_X)
    pred = pd.DataFrame(pred, columns = {'test'}, index = np.arange(368 * i, 368 * (i + 1)))
    pred.to_csv('./experiments/base100_fold%d.csv' % (i + 1), header = None)
