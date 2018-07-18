__author__ = 'jellyzhang'
import datetime
import time
from corpus_helper import export_corpus
from train_textrnn import train
from freeze import freeze_graph
'''
1、从mysql获取corpus，并生成porn.txt和unporn.txt文件
2、使用rnn network 生成最终模型
3、转成固化模型
'''






def train_model():
    #timeprefix='20180511174340'
    timeprefix=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    export_corpus()
    train(timeprefix)
    freeze_graph('rnnmodel/{}/'.format(timeprefix), 'freeze_model', 'textrnn',timeprefix,
                 'Inputs/batch_ph,Inputs/target_ph,Inputs/seq_len_ph,Inputs/keep_prob_ph,Fully_connected_layer/y_hat,Metrics/predictions')

if __name__=='__main__':
     while True:
        train_model()
        time.sleep(24*3600) #隔一天训练一次