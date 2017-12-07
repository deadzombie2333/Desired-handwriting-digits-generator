from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib.layers import dropout
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = "/temp/run-{}".format(now)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

x_input = tf.placeholder(tf.float32,[None,28,28,1])
is_training = tf.placeholder(tf.bool, shape=())
keep_prob = 0.5
x_n_hidden1 = 32
x_n_hidden2 = 64
x_conv_shape = 7*7*64
x_n_hidden3 = 1024
x_n_hidden4 = 1

x_w1 = tf.Variable(xavier_init([5,5,1,x_n_hidden1]))
x_b1 = tf.Variable(tf.zeros(shape = [x_n_hidden1]))

x_w2 = tf.Variable(xavier_init([5,5,x_n_hidden1,x_n_hidden2]))
x_b2 = tf.Variable(tf.zeros(shape = [x_n_hidden2]))

x_w3 = tf.Variable(xavier_init([x_conv_shape,x_n_hidden3]))
x_b3 = tf.Variable(tf.zeros(shape = [x_n_hidden3]))

x_w4 = tf.Variable(xavier_init([x_n_hidden3,x_n_hidden4]))
x_b4 = tf.Variable(tf.zeros(shape = [x_n_hidden4]))

theta_DIS = [x_w1,x_w2,x_w3,x_w4,x_b1,x_b2,x_b3,x_b4]

def discriminator(X):
  
  x_hidden1 = tf.nn.elu(tf.nn.conv2d(X,x_w1,strides = [1,1,1,1], padding = "SAME") + x_b1)
  x_hidden1_pool = tf.nn.max_pool(x_hidden1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "VALID")
  
  x_hidden2 = tf.nn.elu(tf.nn.conv2d(x_hidden1_pool,x_w2,strides = [1,1,1,1], padding = "SAME") + x_b2)
  x_hidden2_pool = tf.nn.max_pool(x_hidden2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "VALID")
  
  x_hidden3 = tf.nn.elu(tf.matmul(tf.reshape(x_hidden2_pool, shape = [-1,x_conv_shape]), x_w3) + x_b3)
  x_hidden3_drop = dropout(x_hidden3, keep_prob = keep_prob, is_training = is_training)
  
  x_output_logits = tf.matmul(x_hidden3_drop, x_w4) + x_b4
  
  return x_output_logits

y_input = tf.placeholder(tf.float32,[None,10])
z_n_hidden1 = 100
z_n_hidden2 = 500
z_n_hidden3 = 784

z_w1 = tf.Variable(xavier_init([100,z_n_hidden1]))
z_b1 = tf.Variable(tf.zeros(shape = [z_n_hidden1]))

z_w2 = tf.Variable(xavier_init([z_n_hidden1,z_n_hidden2]))
z_b2 = tf.Variable(tf.zeros(shape = [z_n_hidden2]))

z_w3 = tf.Variable(xavier_init([z_n_hidden2,z_n_hidden3]))
z_b3 = tf.Variable(tf.zeros(shape = [z_n_hidden3]))

z_w4 = tf.Variable(xavier_init([z_n_hidden3,784]))
z_b4 = tf.Variable(tf.zeros(shape = [784]))

theta_G = [z_w1,z_w2,z_w3,z_w4,z_b1,z_b2,z_b3,z_b4]

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

def generator(Y):
  z_init = np.zeros((10,100),dtype = np.float32)
  for i in range(10):
    z_init[i][i*10:(i+1)*10]=np.random.uniform(-1,1,[10])
  Z = tf.matmul(Y,z_init)
  
  z_hidden1 = tf.nn.elu(tf.matmul(Z,z_w1) + z_b1)
  z_hidden1_drop = dropout(z_hidden1,keep_prob = keep_prob, is_training = is_training)
  
  z_hidden2 = tf.nn.elu(tf.matmul(z_hidden1_drop,z_w2) + z_b2)
  z_hidden2_drop = dropout(z_hidden2,keep_prob = keep_prob, is_training = is_training)
  
  z_hidden3 = tf.nn.elu(tf.matmul(z_hidden2_drop,z_w3) + z_b3)
  z_hidden3_drop = dropout(z_hidden3,keep_prob = keep_prob, is_training = is_training)
  
  z_output_logits = tf.matmul(z_hidden3_drop,z_w4) + z_b4
  z_output = tf.nn.sigmoid(z_output_logits)
  z_output = tf.reshape(z_output,shape = [-1,28,28,1])
  
  return z_output

y_n_hidden1 = 32
y_n_hidden2 = 64
y_conv_shape = 7*7*64
y_n_hidden3 = 1024

y_w1 = tf.Variable(xavier_init([5,5,1,y_n_hidden1]))
y_b1 = tf.Variable(xavier_init([y_n_hidden1]))

y_w2 = tf.Variable(xavier_init([5,5,y_n_hidden1,y_n_hidden2]))
y_b2 = tf.Variable(xavier_init([y_n_hidden2]))

y_w3 = tf.Variable(xavier_init([y_conv_shape,y_n_hidden3]))
y_b3 = tf.Variable(xavier_init([y_n_hidden3]))

y_w4 = tf.Variable(xavier_init([y_n_hidden3,10]))
y_b4 = tf.Variable(xavier_init([10]))

def determinator(Y):
  y_hidden1 = tf.nn.elu(tf.nn.conv2d(Y, y_w1, strides = [1,1,1,1], padding = "SAME") + y_b1)
  y_hidden1_pool = tf.nn.max_pool(y_hidden1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "VALID")
  
  y_hidden2 = tf.nn.elu(tf.nn.conv2d(y_hidden1_pool, y_w2, strides = [1,1,1,1], padding = "SAME") + y_b2)
  y_hidden2_pool = tf.nn.max_pool(y_hidden2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "VALID")
  
  y_hidden3 = tf.nn.elu(tf.matmul(tf.reshape(y_hidden2_pool,shape = [-1,y_conv_shape]), y_w3) + y_b3)
  y_hidden3_drop = dropout(y_hidden3, keep_prob = keep_prob, is_training = is_training)
  
  y_hidden4_logits = tf.matmul(y_hidden3_drop, y_w4) + y_b4
  
  return y_hidden4_logits
  
theta_DER = [y_w1,y_w2,y_w3,y_w4,y_b1,y_b2,y_b3,y_b4]
  
G_sample = generator(y_input)
DIS_logits_real = discriminator(x_input)
DIS_logits_fake = discriminator(G_sample)

DIS_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DIS_logits_real, labels=tf.ones_like(DIS_logits_real)))
DIS_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DIS_logits_fake, labels=tf.zeros_like(DIS_logits_fake)))
DIS_reg_loss = tf.reduce_sum(tf.abs(x_w3)) + tf.reduce_sum(tf.abs(x_w4))
DIS_loss_0 = DIS_loss_real + DIS_loss_fake
DIS_loss = DIS_loss_0 + 0.001 * DIS_reg_loss

train_DIS = tf.train.AdamOptimizer().minimize(DIS_loss, var_list = theta_DIS)

DER_real = determinator(x_input)
DER_fake = determinator(G_sample)

DER_loss_0 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_input, logits = DER_real))
DER_reg_loss = tf.reduce_sum(tf.abs(y_w3)) + tf.reduce_sum(tf.abs(y_w4))
DER_loss = DER_loss_0 + 0.001 * DER_reg_loss

train_DER = tf.train.AdamOptimizer().minimize(DER_loss, var_list = theta_DER)

G_loss_DER = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_input, logits = DER_fake))
G_loss_DIS = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DIS_logits_fake, labels=tf.ones_like(DIS_logits_fake)))
G_reg_loss = tf.reduce_sum(tf.abs(z_w1)) + tf.reduce_sum(tf.abs(z_w2)) + tf.reduce_sum(tf.abs(z_w3)) + tf.reduce_sum(tf.abs(z_w4))

G_loss_0 = G_loss_DIS + G_loss_DER
G_loss = G_loss_0 + 0.001 * G_reg_loss

train_G = tf.train.AdamOptimizer().minimize(G_loss, var_list = theta_G)

saver = tf.train.Saver() 
n_episode = 10

dis_summary = tf.summary.scalar("discriminator_loss",DIS_loss_0)
der_summary = tf.summary.scalar("determinator_loss",DER_loss_0)
gen_dis_summary = tf.summary.scalar("generator_loss_dis",G_loss_DIS)
gen_der_summary = tf.summary.scalar("generator_loss_der",G_loss_DER)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for episode in range(n_episode):
    for _ in range(2000):
      batch_x, batch_y = mnist.train.next_batch(100)
      batch_x = batch_x.reshape([-1,28,28,1])
      
      if batch_index % 10 == 0:
        dis_summ = sess.run(dis_summary, feed_dict = {x_input:batch_x, y_input:batch_y, is_training: False})
        der_summ = sess.run(der_summary, feed_dict = {x_input:batch_x, y_input:batch_y, is_training: False})
        gen_dis_summ = sess.run(gen_dis_summary, feed_dict = {y_input: batch_y, is_training: False})
        gen_der_summ = sess.run(gen_der_summary, feed_dict = {y_input: batch_y, is_training: False})
        step = episode * 20 + batch_index
        file_writer.add_summary(dis_summ, step)
        file_writer.add_summary(der_summ, step)
        file_writer.add_summary(gen_dis_summ, step)
        file_writer.add_summary(gen_der_summ, step)
        
      sess.run(train_DIS, feed_dict = {x_input: batch_x, y_input: batch_y, is_training: True})
      sess.run(train_G, feed_dict = {y_input: batch_y, is_training: True})
      sess.run(train_DER, feed_dict = {x_input: batch_x, y_input: batch_y, is_training: True})
      
    num = np.random.choice(10,10,replace = False).reshape(10,1)
    encoder = OneHotEncoder()
    num_encoded = encoder.fit_transform(num).toarray()
    samples = sess.run(G_sample, feed_dict={y_input:num_encoded, is_training: False})
    
    batch_x, batch_y = mnist.test.next_batch(50)
    batch_x = batch_x.reshape([-1,28,28,1])
    DIS_error = sess.run(DIS_loss_0, feed_dict = {x_input: batch_x, y_input: batch_y, is_training: False})
    DER_error = sess.run(DER_loss_0, feed_dict = {x_input: batch_x, y_input: batch_y, is_training: False})
    G_error_DIS = sess.run(G_loss_DIS, feed_dict = {y_input:batch_y, is_training: False})
    G_error_DER = sess.run(G_loss_DER, feed_dict = {x_input: batch_x, y_input:batch_y, is_training: False})
    
    print("Discriminator error {}".format(DIS_error))
    print("Determinator error {}".format(DER_error))
    print("Generator error, DIS:{}, DER:{}".format(G_error_DIS,G_error_DER))
    
    for i in range(10):
      ax = plt.subplot(2,5,i+1)
      plt.axis('off')
      ax.set_title(num[i])
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.set_aspect('equal')
      plt.imshow(samples[i].reshape(28, 28), cmap='Greys_r', interpolation = "nearest")
    plt.show()
  save_path = saver.save(sess,"/tmp/my_gan.ckpt")

file_writer.close()

print("Run the command line:\n" \
    "--> tensorboard --logdir /temp " \
    "\nThen open http://0.0.0.0:6006/ into your web browser")
