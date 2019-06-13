# データ加工・処理・分析モジュール
import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

# 学習用データとテストデータに分けるためのモジュール（正解率を出すため）
from sklearn.model_selection import train_test_split
# アヤメの花(学習するデータ)
from sklearn.datasets import load_iris

# アヤメの花データ(150個)
iris = load_iris()
# irisをDataFrameで扱う。
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# アヤメの種別(ラベル)を追加
df["target"] = iris.target_names[iris.target]

# 引数で表示数を変更できます。defaultは5
df.head()

# 説明変数X(特徴量4つ×150)と目的変数Y(アヤメの種類×150)に分ける
X = df.drop('target', axis=1)
Y = df['target']

#ここから学習用データとテスト用データに分ける。random_stateは乱数を固定
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

# k-近傍法（k-NN）
from sklearn.neighbors import KNeighborsClassifier

#k-NNインスタンス。今回は3個で多数決。3の値を変更して色々試すと〇
model = KNeighborsClassifier(n_neighbors=3)
#学習モデル構築。引数に訓練データの特徴量と、それに対応したラベル
model.fit(X_train, y_train)

# .scoreで正解率を算出。
print("train score:",model.score(X_train,y_train))
print("test score:",model.score(X_test,y_test))

# 上記データ
data = [[5.2, 3.0, 1.5, 0.6]]

# 構築したモデルからアヤメの種類を求める
print(model.predict(data))

