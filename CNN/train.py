#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pandas as pd
import time
import argparse
import sys
import matplotlib.pyplot as plt
tf.set_random_seed(1234)

#parameters
# args={}
# args['lr'] = 0.001
# args['batch_size'] = 300
# args['init'] =  1
# args['save_dir'] = 'output'
# args['epochs'] = 25
# args['dataAugment'] =  1
# args['train'] = 'train.csv'
# args['val'] = 'valid.csv'
# args['test'] = 'test.csv'
early_stopping=0
early_stopping_epochs=5

def argument_parser(): 
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--lr', default=0.001 ,type = float)
    parser.add_argument('--batch_size', default=250,type = int)
    parser.add_argument('--init',  default = 1,type = int)
    parser.add_argument('--save_dir',default='output')
    parser.add_argument('--epochs',  type = int,default=30)
    parser.add_argument('--dataAugment', default = 0,type = int,)
    parser.add_argument('--train',help='Training data File', default='train.csv')
    parser.add_argument('--val',help='Validation Data File', default='valid.csv')
    parser.add_argument('--test',help='testing data File', default='test.csv')

    args = vars(parser.parse_args())
    return args


def load_data(train_file):
    data=pd.read_csv(train_file)
    X=data.loc[:,data.columns !='label']
    X=X.loc[:,X.columns !='id']
    rows=X.shape[0]
    cols=X.shape[1]
    X=X.values.reshape(rows,cols)
    #X=np.multiply(X, 1.0/255.0)
    X=np.reshape(X,[-1,64,64,3])
    Y=data.loc[:,'label']
    Y_one_hot = np.zeros((rows, 20))
    for i in range(rows):
        Y_one_hot[i][Y[i]]=1
    return X,Y_one_hot

def load_test(test_file):
    data=pd.read_csv(test_file)
    X=data.loc[:,data.columns !='id']
    rows=X.shape[0]
    cols=X.shape[1]
    X=X.values.reshape(rows,cols)
    X=np.reshape(X,[-1,64,64,3])
    #X=np.multiply(X, 1.0/255.0)
    return X,rows


# In[2]:


def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.5
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    
    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy

def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (64,64, 3))
    tf_img=[]
    tf_img.append(tf.image.flip_left_right(X))
    tf_img.append(tf.image.flip_up_down(X))
    tf_img.append(tf.image.transpose_image(X))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3):#change to 3
            for img in X_imgs:
                flipped_imgs = sess.run([tf_img[i]], feed_dict = {X: img})
                X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip

def rotate_images(X_imgs):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (64, 64, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3):#to 3  # Rotation at 90, 180 and 270 degrees
            for img in X_imgs:
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)
        
    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate


# In[3]:


def data_augment(data,y):
    salt = add_salt_pepper_noise(data)
    rotated = rotate_images(data)
    flipped = flip_images(data)
    X=np.concatenate((data,salt,rotated,flipped),axis=0)
    Y=np.concatenate((y,y,y,y,y,y,y,y),axis=0)
    #print(X.shape)
    #print(Y.shape)
    return X,Y


# In[4]:


args = argument_parser()
lr=args['lr']
epochs=args['epochs']
batch_size=args['batch_size']

train_X , train_Y = load_data(args['train'])
val_X , val_Y = load_data(args['val'])
test_X , test_Y = load_test(args['test'])

#print('Before augmentation:',train_X.shape)

if args['dataAugment']==1:
    train_X,train_Y = data_augment(train_X,train_Y)
    
#print('After augmentation:',train_X.shape)



def plots(train,val,name):
    
    x_axis = np.linspace(0, epochs, num=epochs)
    
    y_axis=train
    plt.plot(x_axis, y_axis , label='Training')
    #plt.scatter(x_axis, y_axis, label=None)
    y_axis=val
    plt.plot(x_axis, y_axis , label='Validation')
    #plt.scatter(x_axis, y_axis, label=None)

    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.legend(prop={'size' : 10})
    plt.title('Training Vs Validation')
    plt.show()      


# In[6]:


def printfilters(F1):
    grid = np.random.rand(4, 8)
    fig, axs = plt.subplots(nrows=4, ncols=8, figsize=(9.3, 6),subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0.03, right=0.7, hspace=0.3, wspace=0.05)
    for ax,i in zip(axs.flat,range(0,32)):
        a=F1[:,:,:,i]*255
        ax.imshow(a.astype('uint8'))
        ax.set_title(str(i+1))
    plt.tight_layout()
    plt.show()


# In[9]:


tf.reset_default_graph() 

if args['init'] == 1:
    #Filters
    filters={
    'F1' : tf.get_variable("W1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer()),
    'F2' : tf.get_variable("W2", shape=[5, 5, 32, 32], initializer=tf.contrib.layers.xavier_initializer()),
    'F3' : tf.get_variable("W3", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer()),
    'F4' : tf.get_variable("W4", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer()),
    'F5' : tf.get_variable("W5", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer()),
    'F6' : tf.get_variable("W6", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer()),
    'WFC1' : tf.get_variable("W7", shape=[128*7*7,256], initializer=tf.contrib.layers.xavier_initializer()),
    'WFC2' : tf.get_variable("W8", shape=[256,20], initializer=tf.contrib.layers.xavier_initializer())
    }
    #Biases
    bias={
    'B1' : tf.get_variable("bc1", shape=[32], initializer=tf.contrib.layers.xavier_initializer()),
    'B2' : tf.get_variable("bc2", shape=[32], initializer=tf.contrib.layers.xavier_initializer()),
    'B3' : tf.get_variable("bc3", shape=[64], initializer=tf.contrib.layers.xavier_initializer()),
    'B4' : tf.get_variable("bc4", shape=[64], initializer=tf.contrib.layers.xavier_initializer()),
    'B5' : tf.get_variable("bc5", shape=[64], initializer=tf.contrib.layers.xavier_initializer()),
    'B6' : tf.get_variable("bc6", shape=[128], initializer=tf.contrib.layers.xavier_initializer()),
    'BFC1' : tf.get_variable("bcFC1", shape=[256], initializer=tf.contrib.layers.xavier_initializer()),
    'BFC2' : tf.get_variable("bcFC2", shape=[20], initializer=tf.contrib.layers.xavier_initializer())
    }
if args['init'] == 2:
    #Filters
    filters={
    'F1' : tf.get_variable("W1", shape=[5, 5, 3, 32], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'F2' : tf.get_variable("W2", shape=[5, 5, 32, 32], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'F3' : tf.get_variable("W3", shape=[3, 3, 32, 64], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'F4' : tf.get_variable("W4", shape=[3, 3, 64, 64], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'F5' : tf.get_variable("W5", shape=[3, 3, 64, 64], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'F6' : tf.get_variable("W6", shape=[3, 3, 64, 128], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'WFC1' : tf.get_variable("W7", shape=[128*7*7,256], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'WFC2' : tf.get_variable("W8", shape=[256,20], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True))
    }
    #Biases
    bias={
    'B1' : tf.get_variable("bc1", shape=[32], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'B2' : tf.get_variable("bc2", shape=[32], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'B3' : tf.get_variable("bc3", shape=[64], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'B4' : tf.get_variable("bc4", shape=[64], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'B5' : tf.get_variable("bc5", shape=[64], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'B6' : tf.get_variable("bc6", shape=[128],initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'BFC1' : tf.get_variable("bcFC1", shape=[256], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True)),
    'BFC2' : tf.get_variable("bcFC2", shape=[20], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True))
    }

def batchnorm(x):
    batch_mean, batch_var = tf.nn.moments(x,[0])
    BN = tf.nn.batch_normalization(x,mean=batch_mean,variance=batch_var,offset=True,scale=True,variance_epsilon=1e-4)
    return BN

def CNN(X1,filters,bias):
     #Convolution
    #layer1
    conv1=tf.nn.conv2d(X1, filters['F1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1=tf.nn.bias_add(conv1,bias['B1'])
    conv1=batchnorm(conv1)
    conv1=tf.nn.relu(conv1) 
    
    #conv1=tf.layers.batch_normalization(conv1, training=is_train)
    #conv1=tf.nn.dropout(conv1,drop)
    #layer2
    conv2=tf.nn.conv2d(conv1, filters['F2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2=tf.nn.bias_add(conv2,bias['B2'])
    conv2=batchnorm(conv2)
    conv2=tf.nn.relu(conv2)
    #conv2=tf.layers.batch_normalization(conv2, training=is_train)
    #conv2=tf.nn.dropout(conv2,drop)
    #layer3
    pool1=tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #layer4
    conv3=tf.nn.conv2d(pool1, filters['F3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3=tf.nn.bias_add(conv3,bias['B3'])
    conv3=batchnorm(conv3)
    conv3=tf.nn.relu(conv3)
    #conv3=tf.layers.batch_normalization(conv3, training=is_train)
    #conv3=tf.nn.dropout(conv3,drop)
    #layer5
    conv4=tf.nn.conv2d(conv3, filters['F4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4=tf.nn.bias_add(conv4,bias['B4'])
    conv4=batchnorm(conv4)
    conv4=tf.nn.relu(conv4)
    #layer6
    pool2=tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #layer7
    conv5=tf.nn.conv2d(pool2, filters['F5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5=tf.nn.bias_add(conv5,bias['B5'])
    conv5=batchnorm(conv5)
    conv5=tf.nn.relu(conv5)
    #layer8
    conv6=tf.nn.conv2d(conv5, filters['F6'], strides=[1, 1, 1, 1], padding='VALID')
    conv6=tf.nn.bias_add(conv6,bias['B6'])
    conv6=batchnorm(conv6)
    conv6=tf.nn.relu(conv6)
    conv6=tf.nn.dropout(conv6,drop)
    #layer9
    pool3=tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #layer10
    #FC
    flat=tf.reshape(pool3, [-1, filters['WFC1'].get_shape().as_list()[0]])
    FC1=tf.nn.bias_add(tf.matmul(flat, filters['WFC1']), bias['BFC1'])
    FC1=batchnorm(FC1)
    FC1=tf.nn.relu(FC1)
    
    FC1=tf.nn.dropout(FC1,drop)
    #layer11
    FC2=tf.nn.bias_add(tf.matmul(FC1, filters['WFC2']), bias['BFC2'])
    #FC2=tf.nn.softmax(FC2)
    FC2=batchnorm(FC2)
    return FC2


#Initializations
x = tf.placeholder("float", [None, 64,64,3])
y = tf.placeholder("float", [None, 20])
drop=tf.placeholder(tf.float32)

#negative log likelihood
pred = CNN(x, filters, bias)
pred=tf.nn.softmax(pred)
cost = -tf.reduce_sum(y*tf.log(pred + 1e-7))
regularizer = (tf.nn.l2_loss(filters['WFC2']) + tf.nn.l2_loss(bias['BFC2']))
cost += 5e-4 * regularizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Cross entropy
# pred = CNN(x, filters, bias)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
# correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) 
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    summary_writer = tf.summary.FileWriter("./"+args['save_dir'], sess.graph)
    for i in range(epochs):
        for batch in range(train_X.shape[0]//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,train_X.shape[0])]
            batch_y = train_Y[batch*batch_size:min((batch+1)*batch_size,train_Y.shape[0])]    
            opt = sess.run(optimizer, feed_dict={x: batch_x,y: batch_y,drop:0.5})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y,drop:0.5})
                        
        print("Epoch " + str(i) )
        print("\tLoss= " + str(loss))
        print("\tTraining Accuracy= " + str(acc))

        # Calculate accuracy for test data
        val_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: val_X,y : val_Y,drop:1})
        train_loss.append(loss)
        val_loss.append(valid_loss)
        train_accuracy.append(acc)
        val_accuracy.append(val_acc)
        print("\tValidation Accuracy:"+str(val_acc))
        if early_stopping==1 and i-early_stopping_epochs+1>=0:
            j=i-early_stopping_epochs+1
            flag=0
            while j<i:
                if val_accuracy[j]<val_accuracy[j+1]:
                    flag=1
                j=j+1
            if flag==0:
                break
            
    y_pred_cls = tf.argmax(tf.nn.softmax(pred), dimension=1)
    final_ans=sess.run(y_pred_cls,feed_dict={x:test_X,drop:1})
    f= open("test_submission.csv","w+")
    f.write("id,label\n")
    for i in range(test_Y):
        f.write(str(i)+","+str(final_ans[i])+"\n")
    f.close()
    plots(train_accuracy,val_accuracy,'Accuracy')
    plots(train_loss,val_loss,'Loss')
    printfilters(sess.run(filters['F1']))
    summary_writer.close()
    saver.save(sess,"./"+args['save_dir']+"/model.ckpt")
    