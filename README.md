该项目主要train一个模型（判断短文本是否为色情文本），主要有三个model:textcnn/textrnn+attention/textrcnn

最开始打算使用textcnn和textrnn进行融合，但是textcnn模型效果较差，故只使用了textrnn+attention模型

流程如下：

1、data目录下的import_porn.txt色情短文本；import_unporn.txt正常短文本。通过corpus_helper.py中的insert_data方法可以将语料导入到mysql中

2、执行python  train.py可以将mysql中语料获取、训练textrnn+attention模型；最后保存模型并固化
