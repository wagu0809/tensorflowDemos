from PIL import Image
import numpy as np
import tensorflow as tf
import os


# def abs(x):
#     if x >= 0:
#         return x
#     else:
#         return -x


files = os.listdir("img/")
images = list(filter(lambda img: img.endswith('.png'), files))
input_x = np.zeros((len(images), 784))
for i in range(len(images)):
    im = Image.open("img/" + images[i])
    im = im.convert('L')  # 将图片转换成黑白模式，为灰色图像
    a = np.array(im, np.float32).reshape(784)
    for j in range(784):
        a[j] = abs(a[j] - 255)/255
    input_x[i] = a
xin = tf. placeholder(tf.float32, shape=(None, 784))

convw1 = np.load('w/conv_w1.npy')
convw2 = np.load('w/conv_w2.npy')
convb1 = np.load('w/b_w1.npy')
convb2 = np.load('w/b_w2.npy')
fw1 = np.load('w/w1.npy')
fw2 = np.load('w/w2.npy')
fb1 = np.load('w/b1.npy')
fb2 = np.load('w/b2.npy')

conv_w1 = tf.Variable(convw1)
b_w1 = tf.Variable(convb1)
conv_w2 = tf.Variable(convw2)
b_w2 = tf.Variable(convb2)

w1 = tf.Variable(fw1)
b1 = tf.Variable(fb1)
w2 = tf.Variable(fw2)
b2 = tf.Variable(fb2)
x = tf.reshape(xin, [-1, 28, 28, 1])

conv1 = tf.nn.conv2d(x, conv_w1, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)
conv1 = tf.nn.relu(conv1 + b_w1)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv2 = tf.nn.conv2d(pool1, conv_w2, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)
conv2 = tf.nn.relu(conv2 + b_w2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

flat = tf.reshape(pool2, [-1, 7*7*64])
# print((tf.matmul(flat, w1)+b1))
l1 = tf.nn.relu(tf.matmul(flat, w1)+b1)
# print((tf.matmul(l1, w2)+b2))
y = tf.argmax(tf.matmul(l1, w2)+b2, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

out = np.array(sess.run(y, {xin: input_x}))
a = np.zeros((len(images), 2))
summ = 0
for i in range(len(images)):
    print(images)
    a[i][0] = int(images[i][0])
    a[i][1] = out[i]
    if a[i][0] == a[i][1]:
        summ += 1

accuracy = summ / len(images)
print(a.shape)
print(a)
print('accuracy: ' + str(accuracy))
