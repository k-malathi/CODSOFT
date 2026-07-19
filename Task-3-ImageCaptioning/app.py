!pip install tensorflow opencv-python matplotlib
!pip list
import tensorflow as tf
import os
import cv2
data_dir= 'data'
img_exts= ['jpeg','jpg','bmp','png']
for img_class in os.listdir(data_dir):
    if not os.path.isdir(os.path.join(data_dir, img_class)):
        continue
    for img in os.listdir(os.path.join(data_dir, img_class)):
        img_dir= os.path.join(data_dir, img_class, img)
        try:
            img= cv2.imread(img_dir)
            ext= img_dir.split('.')[-1].lower()
            if ext not in img_exts:
                print('Image not in ext list{}'.format(img_dir))
                os.remove(img_dir)
        except Exception as e:
            print('Issue with image{}'.format(img_dir))
tf.data.Dataset.list_files
import numpy as np
from matplotlib import pyplot as plt
data = tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator= data.as_numpy_iterator()
batch= data_iterator.next()
fig, ax = plt.subplots(ncols= 4, figsize= (20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
data= data.map(lambda x, y: (x/255, y))
scaled_iterator= data.as_numpy_iterator()
batch= scaled_iterator.next()
fig, ax = plt.subplots(ncols= 4, figsize= (20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
len(data)
train_size= int(len(data)*.7)
val_size= int(len(data)*.2)
test_size= int(len(data)*.1)
train= data.take(train_size)
val= data.skip(train_size).take(val_size)
test= data.skip(train_size + val_size).take(test_size)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
model= Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()
pip install tensorboard
logdir= 'logs'
tensorboard_callback= tf.keras.callbacks.TensorBoard(log_dir= logdir)
hist= model.fit(train, epochs=20, validation_data= val, callbacks= [tensorboard_callback])
fig= plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper right")
plt.show()
fig= plt.figure()
plt.plot(hist.history['accuracy'], color='green', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='blue', label='val_acc')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='lower left')
plt.show()
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre= Precision()
re= Recall()
acc= BinaryAccuracy()
for batch in test.as_numpy_iterator():
    x, y= batch
    yhat= model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(f'Presicion:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
import cv2
img= cv2.imread('sadtest.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show
resize= tf.image.resize(img,(256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()
resize.shape
np.expand_dims(resize, 0).shape
yhat= model.predict(np.expand_dims(resize/255, 0))
yhat
if yhat > 0.5:
    print('image is sad')
else:
    print('image is happy')
import os
from PIL import Image

data_dir = "data"   # change if your dataset folder has a different name

for root, dirs, files in os.walk(data_dir):
    for file in files:
        path = os.path.join(root, file)
        try:
            if os.path.getsize(path) == 0:
                print("EMPTY:", path)
                continue

            with Image.open(path) as img:
                img.verify()

        except Exception as e:
            print("CORRUPTED:", path, e)
