import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = os.listdir('C:/Users/itaku/Desktop/SampleData/Apple_Grape/Apple')
Y = os.listdir('C:/Users/itaku/Desktop/SampleData/Apple_Grape/Grape')
X_std = []

for i in range(len(X)):
    img = cv2.imread(os.path.join('C:/Users/itaku/Desktop/SampleData/Apple_Grape/Apple/', X[i]))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = rgb/255
    rgb = cv2.resize(rgb, dsize=(100, 100))
    rgb = rgb.reshape(-1)
    X_std.append(rgb)

for i in range(len(Y)):
    img = cv2.imread(os.path.join('C:/Users/itaku/Desktop/SampleData/Apple_Grape/Grape/', Y[i]))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = rgb / 255
    rgb = cv2.resize(rgb, dsize=(100, 100))
    rgb = rgb.reshape(-1)
    X_std.append(rgb)

K = KMeans(n_clusters=2)
K.fit(X_std)
pre = K.predict(X_std)

fig, ax = plt.subplots(5, 4, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(len(pre)):
    img = X_std[i].reshape(100, 100, 3)
    ax[i].imshow(img)
    ax[i].set_title('Cluster:%d' % pre[i])

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()