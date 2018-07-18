__author__ = 'jellyzhang'
import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
"""
freeze.py:
        主要用来固化图和模型参数，生成缩减版的pb文件
"""

def  freeze_graph(in_model_path,out_model_dir,out_model_name,timeprefix,output_node_names):
    if not os.path.exists(out_model_dir):
        os.mkdir(out_model_dir)
    if os.path.exists(in_model_path):
        ckpt = tf.train.get_checkpoint_state(in_model_path)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        # for nodedef in input_graph_def.node._values:
        #     print(nodedef.name)
        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(",")  # We split on comma for convenience
            )
            output_timeprefix=out_model_dir+'/'+timeprefix+'/'
            if not os.path.exists(output_timeprefix):
                os.mkdir(output_timeprefix)
            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_timeprefix+out_model_name+'.pb', "wb") as f:
                f.write(output_graph_def.SerializeToString())
                print('成功转换成固化模型')

    else:
            print('请检查模型文件是否存在')




#textcnn/textrnn/textrcnn 分别固化
if __name__=='__main__':
    try:
        #freeze_graph('checkpoints/','freeze_model','textcnn.pb','input_x,keep_prob,output/logits,output/predictions')
        freeze_graph('rnnmodel/20180601134046/','freeze_model','textrnn','20180601134046','Inputs/batch_ph,Inputs/target_ph,Inputs/seq_len_ph,Inputs/keep_prob_ph,Fully_connected_layer/y_hat,Metrics/predictions')
        # freeze_graph('','')
    except Exception as ex:
        print(ex)