# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing,linear_model
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"C:\Users\itaku\Desktop\SampleData\MA-3-1.csv",header=None)
df.columns=[u'広さ',u'築年数',u'価格']
x=df[['広さ']].values
y=df[['築年数']].values
z=df['価格'].values
plt.scatter(x,y)
plt.title('X and Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
plt.scatter(x,z)
plt.title('X and Z')
plt.xlabel('X')
plt.ylabel('Z')
plt.show()
plt.scatter(y,z)
plt.title('Y and Z')
plt.xlabel('Y')
plt.ylabel('Z')
plt.show()
print(x)
print(y)
print(z)

sc=preprocessing.StandardScaler()
sc.fit(x)
x=sc.transform(x)

x_train,x_test,z_train,z_test=train_test_split(x,z,test_size=0.2,random_state=0)
print(x_train)
print(z_train)
print(x_test)
print(z_test)

clf=linear_model.SGDRegressor(max_iter=1000)
clf.fit(x_train,z_train)
print('切片',clf.intercept_)
print('係数',clf.coef_)

line_X = np.arange(-2, 2, 0.1)  # 3から10まで1刻み
line_Y = clf.predict(line_X[:, np.newaxis])
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.scatter(x_train, z_train, c='b', marker='s')
plt.plot(line_X, line_Y, c='r')
plt.show()

z_pred = clf.predict(x_test)
plt.subplot(2, 1, 2)
plt.scatter(z_test, z_pred - z_test, c='r', marker='s')
plt.hlines(y=0, xmin=0, xmax=50, colors='black')
plt.show()

print("平均二乗誤差",np.mean((z_pred-z_test)**2))