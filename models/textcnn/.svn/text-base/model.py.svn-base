import tensorflow as tf
import numpy as np

class TextCNN(object):
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,num_filters,filter_sizes):
       self.input_x=tf.placeholder(tf.int32,shape=[None,sequence_length],name='input_x')
       self.target=tf.placeholder(tf.int32,shape=[None,num_classes],name='target')
       self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')


       with tf.name_scope('embedding'):
           self.W=tf.Variable(tf.random_uniform([vocab_size+1,embedding_size],-1,1),name='embedding_W')
           self.embed=tf.nn.embedding_lookup(self.W,self.input_x)
       self.embed_expand=tf.expand_dims(self.embed,-1)
       pooled_outputs=[]
       for index,filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpooling{}'.format(filter_size)):
                filter_shape=[filter_size,embedding_size,1,num_filters]
                filter_W=tf.truncated_normal(filter_shape,stddev=0.1,name='filter_W')
                filter_b=tf.truncated_normal(shape=[num_filters],name='filter_b')
                conv=tf.nn.conv2d(self.embed_expand,filter_W,[1,1,1,1],'VALID',name='conv')
                h=tf.nn.relu(tf.nn.bias_add(conv,filter_b))
                pooled=tf.nn.max_pool(value=h,ksize=[1,sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID')
                pooled_outputs.append(pooled) #shape=[batch_size,1,1,num_filters]
       #combine all the pooled features
       num_filters_total=len(filter_sizes)*num_filters
       self.h_pool=tf.concat(pooled_outputs,3)
       self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])
       #dropout
       with tf.name_scope('dropout'):
           self.h_dropout=tf.nn.dropout(self.h_pool_flat,self.keep_prob,name='dropout')

       #output
       with tf.name_scope('output'):
           W=tf.Variable(tf.random_uniform([num_filters_total,num_classes],-1,1),name='w')
           b=tf.Variable(tf.random_uniform([num_classes],-1,1))
           self.logits=tf.nn.xw_plus_b(self.h_dropout,W,b,name='logits')
           self.predictions=tf.argmax(self.logits,1,name='predictions')

       with tf.name_scope('loss'):
           self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target,logits=self.logits),name='loss')

       with tf.name_scope('accuracy'):
           correct_predictions=tf.equal(self.predictions,tf.argmax(self.target,1))
           self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')




