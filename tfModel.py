import numpy as np
import tensorflow as tf
import pandas as pd

class tfModel(object):
    
    def __init__(self):
        self.data = []
        self.learning_rate = 0.01
        self.batch_size = 32
        self.display_step = 1
        self.train_epochs = 30
        self.n_hidden = 128
        self.n_hidden2 = 64
        self.n_classes = 2
        self.weights = {
            'hidden1': tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden,self.n_classes]))
        }
        self.biases = {
            'hidden1': tf.Variable(tf.random_normal([self.n_hidden2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

    def dynamicRNN(self, x, seqlen, weights, biases, keep_prob=1):
        x = tf.unstack(x, self.max_seqlen, 1)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length = seqlen)

        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1,0,2])

        batch_size = tf.shape(outputs)[0]
        index = tf.range(0,batch_size)*self.max_seqlen + (seqlen-1)
        
        outputs = tf.gather(tf.reshape(outputs,[-1,self.n_hidden]),index)
        #layer1 = tf.matmul(outputs,weights['hidden1'])+biases['hidden1']
        layer1 = tf.matmul(outputs,weights['out'])+biases['out']
        pred = tf.nn.softmax(layer1)
        return pred, layer1
        
    def get_nxt_batch(self, train_data, train_label, seqlen1, index):
        start = self.batch_size*index;
        end = start+self.batch_size
        return train_data[start:end], train_label[start:end],seqlen1[start:end]

    def build(self, max_seqlen, train_seqlen, xt, yt, test_seqlen, test_xt, test_yt):
        self.max_seqlen = max_seqlen
        x = tf.placeholder("float", [None, max_seqlen, 300])
        y = tf.placeholder(tf.int32, [None,])
        seqlen = tf.placeholder(tf.int32,[None,])
        keep_prob = tf.placeholder(tf.float32)
        lr = tf.placeholder(tf.float32)

        y_pred, logit = self.dynamicRNN(x, seqlen, self.weights, self.biases, keep_prob)

        l2_beta = 0.0005
        l2_loss = l2_beta*(tf.nn.l2_loss(self.weights['out'])+tf.nn.l2_loss(self.biases['out']))
        #l2_loss = 0
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=y))+l2_loss
        #cost = tf.reduce_mean(tf.pow(y_pred-y,2))
        global_step = tf.Variable(0, trainable=False)
        
        learning_rate = tf.train.exponential_decay(self.learning_rate,global_step, 600, 0.85, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
        #y_pred = np.array(y_pred)
        #a = np.array(y_pred)
        #a[y_pred<0.5]=0
        #a[y_pred>=0.5]=1
        #correct_pred = tf.equal(tf.cast(a,tf.int32),y)
        #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            rows = xt.shape[0]
            
            for i in range(self.train_epochs):
                step = 0
                while step*self.batch_size<rows:
                    batch_x, batch_y, batch_seqlen = self.get_nxt_batch(xt, yt, train_seqlen, step)
                    sess.run(optimizer, feed_dict={x:batch_x,y:batch_y,seqlen:batch_seqlen, keep_prob:0.5, lr:self.learning_rate})
                    step+=1
                    #sess.run(optimizer, feed_dict={x:batch_x,y:batch_y,seqlen:batch_seqlen, keep_prob:0.5, lr:0.005})
         
                if i % self.display_step == 0:
                
                    predictions, loss = sess.run([y_pred,cost], feed_dict={x:test_xt, y:test_yt, seqlen:test_seqlen, keep_prob:1})
                    acc = 0
                    for j in range(0,len(test_yt)):
                        if predictions[j][1] > 0.5 and test_yt[j]>0.9:
                            acc+=1
                        elif predictions[j][1] <= 0.5 and test_yt[j]<0.1:
                            acc+=1
                    #print("acc: "+str(acc))
                    acc = acc/len(predictions)
                    
                    print("Iter: "+str(i)+",Loss= "+ "{:.6f}".format(loss)+", Training Accuracy= "+ "{:.5f}".format(acc))

            print("Optimization Finished")
            saver = tf.train.Saver()
            saver.save(sess, "tf_model.ckpt")
            print("Saved Model")

    def predict(self, max_seqlen, test_seqlen, test_xt):
        self.max_seqlen = max_seqlen
        x = tf.placeholder("float", [None, max_seqlen, 300])
        #y = tf.placeholder(tf.int32, [None,])
        seqlen = tf.placeholder(tf.int32,[None,])
        keep_prob = tf.placeholder(tf.float32)

        y_pred, logit = self.dynamicRNN(x, seqlen, self.weights, self.biases, keep_prob)

        saver = tf.train.Saver()
        #predictions=[]
        with tf.Session() as sess:
            rows = test_xt.shape[0]
            saver.restore(sess, "tf_model.ckpt")
            #acc = 0
            feed_dict = {x: test_xt, seqlen:test_seqlen, keep_prob:1}
            predictions = sess.run(y_pred, feed_dict=feed_dict)
            print(predictions)
            '''
            for j in range(0, len(predictions)):
                if predictions[j][1] > 0.5 and test_yt[j] > 0.9:
                    acc += 1
                elif predictions[j][1] <= 0.5 and test_yt[j] < 0.1:
                    acc += 1
                
            print("acc: "+str(acc))
            acc = acc/len(test_yt)
            print ("Testing Accuracy: " + "{:.5f}".format(acc))
            '''
        return predictions





