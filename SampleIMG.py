import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

(train_data,train_label),(test_data,test_label)=tf.keras.datasets.cifar10.load_data()
train_data=train_data/255
test_data=test_data/255
train_label=train_label.ravel()
test_label=test_label.ravel()
train_data.shape
label_names=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

fig,axes=plt.subplots(10,figsize=(8,15))
for i in range(10):
    imgs=train_data[train_label==i][:10]
    axes[i].imshow(imgs.transpose(1,0,2,3).reshape(32,10*32,3))
    axes[i].axis("off")
    axes[i].set_title(label_names[i])

model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),
                           input_shape=(32,32,3),activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dense(10,activation="softmax")
])
print(model.summary())
model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=["accuracy"])

np.random.seed(1)
tf.set_random_seed(2)
model.fit(train_data,train_label,epochs=20)

pred=model.predict_classes(test_data)
print("正答率",(pred==test_label).sum()/len(test_label))

