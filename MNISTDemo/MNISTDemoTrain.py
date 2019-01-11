import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def main():
    xin = tf.placeholder(tf.float32, shape=(None, 784))  # 图片的像素点数量
    y_ = tf.placeholder(tf.float32, shape=(None, 10))  # 0 - 9 label的个数

    # 设置权重和偏移量
    if input('New or ... ') == 'new':
    # if True:
        # 卷积层的权重和偏移量===========================
        # [3, 3, 1, 32] 前两个3的意思是在图像上进行窗口扫描，这个窗口的大小为3*3，‘1’为通道数量
        # ‘32’为想要得到的特征图的数量
        conv_w1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
        b_w1 = tf.Variable(tf.constant(0.1, shape=[32]))
        # 这里的32位第一层卷积输出特征图的数量决定的，64位第二次卷积输出的特征图数量
        conv_w2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
        b_w2 = tf.Variable(tf.constant(0.1, shape=[64]))

        # 连接层的权重和偏移量============================
        # 1024为目标向量长度 7*7*64：7*7为两次池化后特征图的大小，64位特征图的数量 具体见池化操作注释
        w1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        # 第二次连接操作 将1024长度的向量转化为10，即label的个数
        w2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[10]))
    else:
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

    # 卷积层操作================================
    # shape `[batch, in_height, in_width, in_channels]`
    x = tf.reshape(xin, [-1, 28, 28, 1])
    conv1 = tf.nn.conv2d(x, conv_w1, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)
    conv1 = tf.nn.relu(conv1 + b_w1)
    # 第一次池化使用2*2大小的窗口，结果特征图从原图28*28变成了14*14大小
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Pooling

    conv2 = tf.nn.conv2d(pool1, conv_w2, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)
    conv2 = tf.nn.relu(conv2 + b_w2)
    # 第二次池化同样使用2*2大小的窗口，结果特征图从第一次池化的特征图14*14变成了7*7大小
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(pool2)

    # 连接层操作=================================
    # 将第二次池化的输出结果reshape
    flat = tf.reshape(pool2, [-1, 7*7*64])
    print(flat)
    l1 = tf.nn.relu(tf.matmul(flat, w1)+b1)
    # dropout的操作
    keep_prob = tf.placeholder(tf.float32)
    drop1 = tf.nn.dropout(l1, keep_prob)
    y = tf.matmul(drop1, w2) + b2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    summ = 0
    # t1 = time . clock()

    for i in range(300):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xin: batch_xs,
                                        y_: batch_ys,
                                        keep_prob: 0.7})
        if i % 100 == 0:
            print('step: ' + str(i))
            rate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1),
                                                   tf.argmax(y_, 1)),
                                          tf.float32))
            print(sess.run(rate, feed_dict={xin: batch_xs,
                                            y_: batch_ys,
                                            keep_prob: 1}))
    # print(' time : '+ str(time . clock() -t1))

    # time_all = time . clock() - t1
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    np.save('w/conv_w1', sess.run(conv_w1))
    np.save('w/conv_w2', sess.run(conv_w2))
    np.save('w/b_w1', sess.run(b_w1))
    np.save('w/b_w2', sess.run(b_w2))
    np.save('w/w1', sess.run(w1))
    np.save('w/w2', sess.run(w2))
    np.save('w/b1', sess.run(b1))
    np.save('w/b2', sess.run(b2))
    # print(sess.run(accuracy, feed_dict={xin: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))
    # print(' time : '+ str(time_all))


main()

