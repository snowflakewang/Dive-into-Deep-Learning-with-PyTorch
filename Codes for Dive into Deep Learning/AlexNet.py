import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

inputs = tf.keras.Input(shape=(224,224,3))
#卷积核96个，尺寸（11，11），步长为4，输入维度为（224，224，3），输出维度：（224-11）/4+1=54，（54，54，96）
x = Conv2D(filters=96, kernel_size=(11,11),strides=4)(inputs)
#批标准化
x = BatchNormalization()(x)
x = Activation('relu')(x)
#最大池化，输出维度为（27，27，96）
x = MaxPool2D(pool_size = (3,3),strides=2)(x)
#padding=same，输出维度为（27，27，256）
x = Conv2D(filters=256, kernel_size=(5,5),padding='same',strides=1)(x)
#最大池化，输出维度为（13，13，256）
x = MaxPool2D(pool_size = (3,3),strides=2)(x)
#padding=same，输出维度为（13，13，384）
x = Conv2D(filters=384, kernel_size=(3,3),padding='same',strides=1)(x)
#padding=same，输出维度为（13，13，384）
x = Conv2D(filters=384, kernel_size=(3,3),padding='same',strides=1)(x)
#padding=same，输出维度为（13，13，256）
x = Conv2D(filters=256, kernel_size=(3,3),padding='same',strides=1)(x)
#最大池化，输出维度为（6，6，256）
x = MaxPool2D(pool_size = (3,3),strides=2)(x)
#连接全连接层前拉直
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1000, activation='softmax')(x)

model = tf.keras.Model(inputs,predictions)
model.summary()


