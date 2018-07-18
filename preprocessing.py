__author__ = 'jellyzhang'
import numpy as np
import jieba
import re
from keras.preprocessing import sequence
from keras.models import Sequential
import itertools
from collections import Counter
#加载停用词
def get_stopwords(path):
    return [line.strip() for line in open(path,'r',encoding='utf-8').readlines()]
#句子去停用词
def removestopwords(sentence):
        stopwords_list=get_stopwords('stopwords.txt')
        outstr=[]
        for word in sentence:
            if not word in stopwords_list:
                if word!='\n' and word!='\t':
                     outstr.append(word)
        return outstr

#分词 并去掉停用词
def cut(sentence):
    r= u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
    sentence=re.sub(r,'',sentence)
    return removestopwords(jieba.cut(sentence))

#获取训练word
def get_vocab(sentences):
    counts = Counter(list(itertools.chain.from_iterable(sentences)))
    # 选择超过10次的value
    vocab_list = []
    for word in counts:
        if counts[word] >= 10 and counts[word]<=len(sentences)*0.95:
            vocab_list.append(word)
    vocab = sorted(vocab_list)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return vocab, vocab_to_int

#word被替换成索引后的句子representation
def get_sentence2int(sentences,vocab_to_int,maxlength):
    reviews_ints = []
    for each in sentences:
        int_eachsententce=[]
        for word in each:
            if word in vocab_to_int:
                int_eachsententce.append(vocab_to_int[word])
        reviews_ints.append(int_eachsententce)
    reviews_ints=sequence.pad_sequences(reviews_ints,padding='post', maxlen=maxlength)
    return reviews_ints


#加载训练数据
def load_data(pornfile,unpornfile,maxlength):
    data=[]
    porndata=[]
    unporndata=[]
    words=[]
    with open(pornfile,'r',encoding='utf-8',errors='ignore') as fread1:
        for line in fread1:
            porndata.append(line.rstrip())
    with open(unpornfile, 'r', encoding='utf-8', errors='ignore') as fread2:
        for line in fread2:
            unporndata.append(line.rstrip())
    data=porndata+unporndata
    X=[cut(x) for x in data]
    vocab, vocab_to_int=get_vocab(X)
    texts_ints=get_sentence2int(X,vocab_to_int,maxlength)
    # print(X[:10])
    # print('************')
    # print(X[-10:])
    porn_labels=[[0,1] for _ in range(len(porndata))]
    unporn_labels=[[1,0] for _ in range(len(unporndata))]
    labels=np.concatenate((np.array(porn_labels),np.array(unporn_labels)),axis=0)
    # print(labels[:10])
    # print('************')
    # print(labels[-10:])

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(labels)))
    x_shuffled = texts_ints[shuffle_indices]
    y_shuffled = labels[shuffle_indices]

    return X,vocab,vocab_to_int,x_shuffled,y_shuffled
#print(load_data('data/porn.txt','data/unporn.txt'))

#生成可以填充的批量训练数据
def get_batch_iter(data,epochs,batch_size,shuffle=True):
    data=np.array(data)
    batches=len(data)//batch_size
    for epoch in range(epochs):
        if shuffle:
            indices = np.random.permutation(len(data))
            data_shuffle = data[indices]
        else:
            data_shuffle = data
        for batch in range(batches):
            start=batch*batch_size
            end=(batch+1)*batch_size

            yield data[start:end]

