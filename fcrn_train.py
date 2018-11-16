import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorflow.python.ops import variables
from tensorflow.python import pywrap_tensorflow
import re
import numpy as np
import os
need_pretrain=True

trainsize=17582
testsize=1000
#huber_c=.2*max(|y˜i − yi|)

input_h=318
input_w=424

height = 228
width = 304
depth_height=128
depth_width=160
channels = 3
checkpoint_path = "./checkpoints/NYU_FCRN.ckpt"

restore_from_fcrn=False
restore_from_ckpt=True
batch_size=4
TRAIN_TFRECORD='./train.tfrecords'
TEST_TFRECORD='./test.tfrecords'
BATCH_CAPACITY=512  #
MIN_AFTER_DEQU=256  #
MAX_Cycle=100000
TRAIN_CYCLE=int(trainsize/batch_size)
TEST_CYCLE=int(testsize/batch_size)
learning_rt = 0.001
savepath='./fcrnckpt/'
logpath='./fcrnlogs/'
ckpt_path = savepath+'124.ckpt'

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img': tf.FixedLenFeature([], tf.string),
                                           'label' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [input_h,input_w, 3])

    img = tf.image.random_brightness(img, max_delta=0.3)
    img = tf.image.random_contrast(img, lower=0.1, upper=0.6)
    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.random_saturation(img, lower=0.5, upper=2.5)
    img = tf.cast(img, tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)
    label = tf.reshape(label, [input_h,input_w, 1])
    imgt=tf.concat([img,label],axis=-1)

    random_rota=tf.random_uniform([],-0.15,0.15,tf.float32)
    imgt=tf.contrib.image.rotate(imgt,random_rota,'BILINEAR')
    imgt=tf.random_crop(imgt,[height,width,4])
    img=imgt[:,:,0:3]
    img = tf.cast(img, tf.float32)
    # img = tf.image.per_image_standardization(img)
    label=tf.image.resize_images(imgt[:,:,3:],[depth_height,depth_width])
    label= tf.cast(label,tf.float32)#/65535.
    return img,label

def read_and_decode_test(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img': tf.FixedLenFeature([], tf.string),
                                           'label' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来

    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [input_h,input_w, 3])
    img=tf.image.resize_images(img,[height,width])
    img = tf.cast(img, tf.float32)
    # img = tf.image.per_image_standardization(img)
    label = tf.decode_raw(features['label'], tf.float32)
    label = tf.reshape(label, [input_h,input_w , 1])
    label=tf.image.resize_images(label,[depth_height,depth_width])
    label= tf.cast(label,tf.float32)#/65535.
    return img,label

def get_incoming_shape(incoming):
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")

def interleave(tensors, axis):
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    return tf.reshape(tf.stack(tensors, axis + 1), new_shape)

def inference(inputs,istrain,reuse):
    with tf.variable_scope('model',reuse=reuse):
        net=InputLayer(inputs,name='inputs')
        net=Conv2d(net,64,(7,7),(2,2),name='conv1')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn_conv1')
        pool1=MaxPool2d(net,(3,3),(2,2),name='pool1')
        net=Conv2d(pool1,256,(1,1),b_init=None,name='res2a_branch1')
        bn2a=BatchNormLayer(net,0.999,0.00001,None,istrain,name='bn2a_branch1')

        net=Conv2d(pool1,64,(1,1),(1,1),b_init=None,name='res2a_branch2a')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn2a_branch2a')
        net=Conv2d(net,64,(3,3),(1,1),b_init=None,name='res2a_branch2b')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn2a_branch2b')
        net=Conv2d(net,256,(1,1),(1,1),b_init=None,name='res2a_branch2c')
        net=BatchNormLayer(net,0.999,0.00001,None,istrain,name='bn2a_branch2c')

        res2a=ElementwiseLayer([bn2a,net],combine_fn=tf.add, name='res2a')
        relu2a=LambdaLayer(res2a,tf.nn.relu,name='relu2a')
        net=Conv2d(relu2a,64,(1,1),(1,1),b_init=None,name='res2b_branch2a')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn2b_branch2a')
        net=Conv2d(net,64,(3,3),(1,1),b_init=None,name='res2b_branch2b')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn2b_branch2b')
        net=Conv2d(net,256,(1,1),(1,1),b_init=None,name='res2b_branch2c')
        net=BatchNormLayer(net,0.999,0.00001,None,istrain,name='bn2b_branch2c')

        res2b=ElementwiseLayer([res2a,net],combine_fn=tf.add, name='res2b')
        relu2b=LambdaLayer(res2b,tf.nn.relu,name='relu2b')
        net=Conv2d(relu2b,64,(1,1),(1,1),b_init=None,name='res2c_branch2a')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn2c_branch2a')
        net=Conv2d(net,64,(3,3),(1,1),b_init=None,name='res2c_branch2b')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn2c_branch2b')
        net=Conv2d(net,256,(1,1),(1,1),b_init=None,name='res2c_branch2c')
        net=BatchNormLayer(net,0.999,0.00001,None,istrain,name='bn2c_branch2c')
        res2c=ElementwiseLayer([res2b,net],combine_fn=tf.add, name='res2c')
        relu2c=LambdaLayer(res2c,tf.nn.relu,name='relu2c')
        net=Conv2d(relu2c,512,(1,1),(2,2),b_init=None,name='res3a_branch1')
        bn3a=BatchNormLayer(net,0.999,0.00001,None,istrain,name='bn3a_branch1')

        net=Conv2d(relu2c,128,(1,1),(2,2),b_init=None,name='res3a_branch2a')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn3a_branch2a')
        net=Conv2d(net,128,(3,3),(1,1),b_init=None,name='res3a_branch2b')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn3a_branch2b')
        net=Conv2d(net,512,(1,1),(1,1),b_init=None,name='res3a_branch2c')
        net=BatchNormLayer(net,0.999,0.00001,None,istrain,name='bn3a_branch2c')
        res3a=ElementwiseLayer([bn3a, net],combine_fn=tf.add,  name='res3a')
        relu3a = LambdaLayer(res3a, tf.nn.relu, name='res3a_relu')
        net=Conv2d(relu3a,128,(1,1),(1,1),b_init=None,name='res3b_branch2a')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn3b_branch2a')
        net=Conv2d(net,128,(3,3),(1,1),b_init=None,name='res3b_branch2b')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn3b_branch2b')
        net=Conv2d(net,512,(1,1),(1,1),b_init=None,name='res3b_branch2c')
        net=BatchNormLayer(net,0.999,0.00001,None,istrain,name='bn3b_branch2c')
        res3b = ElementwiseLayer([relu3a, net], combine_fn=tf.add, name='res3b')
        relu3b = LambdaLayer(res3b, tf.nn.relu, name='res3b_relu')
        net = Conv2d(relu3b, 128, (1, 1), (1, 1), b_init=None, name='res3c_branch2a')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn3c_branch2a')
        net = Conv2d(net, 128, (3, 3), (1, 1), b_init=None, name='res3c_branch2b')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn3c_branch2b')
        net = Conv2d(net, 512, (1, 1), (1, 1), b_init=None, name='res3c_branch2c')
        net = BatchNormLayer(net, 0.999, 0.00001, None, istrain, name='bn3c_branch2c')
        res3c = ElementwiseLayer([relu3b, net],combine_fn=tf.add,  name='res3c')
        relu3c = LambdaLayer(res3c, tf.nn.relu, name='res3c_relu')
        net = Conv2d(relu3c, 128, (1, 1), (1, 1), b_init=None, name='res3d_branch2a')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn3d_branch2a')
        net = Conv2d(net, 128, (3, 3), (1, 1), b_init=None, name='res3d_branch2b')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn3d_branch2b')
        net = Conv2d(net, 512, (1, 1), (1, 1), b_init=None, name='res3d_branch2c')
        net = BatchNormLayer(net, 0.999, 0.00001, None, istrain, name='bn3d_branch2c')
        res3d=ElementwiseLayer([relu3c,net],combine_fn=tf.add, name='res3d')
        relu3d=LambdaLayer(res3d,tf.nn.relu,name='relu3d')
        net=Conv2d(relu3d,1024,(1,1),(2,2),b_init=None,name='res4a_branch1')
        bn4a=BatchNormLayer(net,0.999,0.00001,None,istrain,name='bn4a_branch1')

        net=Conv2d(relu3d,256,(1,1),(2,2),b_init=None,name='res4a_branch2a')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn4a_branch2a')
        net=Conv2d(net,256,(3,3),(1,1),b_init=None,name='res4a_branch2b')
        net=BatchNormLayer(net,0.999,0.00001,tf.nn.relu,istrain,name='bn4a_branch2b')
        net=Conv2d(net,1024,(1,1),(1,1),b_init=None,name='res4a_branch2c')
        net=BatchNormLayer(net,0.999,0.00001,None,istrain,name='bn4a_branch2c')
        res4a = ElementwiseLayer([bn4a, net],combine_fn=tf.add,  name='res4a')
        relu4a = LambdaLayer(res4a, tf.nn.relu, name='res4a_relu')
        net = Conv2d(relu4a, 256, (1, 1), (1, 1), b_init=None, name='res4b_branch2a')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn4b_branch2a')
        net = Conv2d(net, 256, (3, 3), (1, 1), b_init=None, name='res4b_branch2b')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn4b_branch2b')
        net = Conv2d(net, 1024, (1, 1), (1, 1), b_init=None, name='res4b_branch2c')
        net = BatchNormLayer(net, 0.999, 0.00001, None, istrain, name='bn4b_branch2c')
        res4b = ElementwiseLayer([relu4a, net],combine_fn=tf.add, name='res4b')
        relu4b = LambdaLayer(res4b, tf.nn.relu, name='res4b_relu')
        net = Conv2d(relu4b, 256, (1, 1), (1, 1), b_init=None, name='res4c_branch2a')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn4c_branch2a')
        net = Conv2d(net, 256, (3, 3), (1, 1), b_init=None, name='res4c_branch2b')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn4c_branch2b')
        net = Conv2d(net, 1024, (1, 1), (1, 1), b_init=None, name='res4c_branch2c')
        net = BatchNormLayer(net, 0.999, 0.00001, None, istrain, name='bn4c_branch2c')
        res4c = ElementwiseLayer([relu4b, net],combine_fn=tf.add, name='res4c')
        relu4c = LambdaLayer(res4c, tf.nn.relu, name='res4c_relu')
        net = Conv2d(relu4c, 256, (1, 1), (1, 1), b_init=None, name='res4d_branch2a')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn4d_branch2a')
        net = Conv2d(net, 256, (3, 3), (1, 1), b_init=None, name='res4d_branch2b')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn4d_branch2b')
        net = Conv2d(net, 1024, (1, 1), (1, 1), b_init = None, name = 'res4d_branch2c')
        net = BatchNormLayer(net, 0.999, 0.00001, None, istrain, name='bn4d_branch2c')
        res4d = ElementwiseLayer([relu4c, net],combine_fn=tf.add, name='res4d')
        relu4d = LambdaLayer(res4d, tf.nn.relu, name='res4d_relu')
        net = Conv2d(relu4d, 256, (1, 1), (1, 1), b_init=None, name='res4e_branch2a')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn4e_branch2a')
        net = Conv2d(net, 256, (3, 3), (1, 1), b_init=None, name='res4e_branch2b')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn4e_branch2b')
        net = Conv2d(net, 1024, (1, 1), (1, 1), b_init=None, name='res4e_branch2c')
        net = BatchNormLayer(net, 0.999, 0.00001, None, istrain, name='bn4e_branch2c')
        res4e = ElementwiseLayer([relu4d, net], combine_fn=tf.add, name='res4e')
        relu4e = LambdaLayer(res4e, tf.nn.relu, name='res4e_relu')
        net = Conv2d(relu4e, 256, (1, 1), (1, 1), b_init=None, name='res4f_branch2a')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn4f_branch2a')
        net = Conv2d(net, 256, (3, 3), (1, 1), b_init=None, name='res4f_branch2b')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn4f_branch2b')
        net = Conv2d(net, 1024, (1, 1), (1, 1), b_init=None, name='res4f_branch2c')
        net = BatchNormLayer(net, 0.999, 0.00001, None, istrain, name='bn4f_branch2c')
        res4f=ElementwiseLayer([relu4e,net],combine_fn=tf.add, name='res4f')
        relu4f=LambdaLayer(res4f,tf.nn.relu,name='relu4f')
        net=Conv2d(relu4f,2048,(1,1),(2,2),b_init=None,name='res5a_branch1')
        bn5a=BatchNormLayer(net,0.999,0.00001,None,istrain,name='bn5a_branch1')
        net = Conv2d(relu4f, 512, (1, 1), (2, 2), b_init=None, name='res5a_branch2a')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn5a_branch2a')
        net = Conv2d(net, 512, (3, 3), (1, 1), b_init=None, name='res5a_branch2b')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn5a_branch2b')
        net = Conv2d(net, 2048, (1, 1), (1, 1), b_init=None, name='res5a_branch2c')
        net = BatchNormLayer(net, 0.999, 0.00001, None, istrain, name='bn5a_branch2c')
        res5a = ElementwiseLayer([bn5a, net], combine_fn=tf.add, name='res5a')
        relu5a = LambdaLayer(res5a, tf.nn.relu, name='res5a_relu')
        net = Conv2d(relu5a, 512, (1, 1), (1, 1), b_init=None, name='res5b_branch2a')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn5b_branch2a')
        net = Conv2d(net, 512, (3, 3), (1, 1), b_init=None, name='res5b_branch2b')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn5b_branch2b')
        net = Conv2d(net, 2048, (1, 1), (1, 1), b_init=None, name='res5b_branch2c')
        net = BatchNormLayer(net, 0.999, 0.00001, None, istrain, name='bn5b_branch2c')
        res5b= ElementwiseLayer([relu5a, net], combine_fn=tf.add, name='res5b')
        relu5b = LambdaLayer(res5b, tf.nn.relu, name='res5b_relu')
        net = Conv2d(relu5b, 512, (1, 1), (1, 1), b_init=None, name='res5c_branch2a')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn5c_branch2a')
        net = Conv2d(net, 512, (3, 3), (1, 1), b_init=None, name='res5c_branch2b')
        net = BatchNormLayer(net, 0.999, 0.00001, tf.nn.relu, istrain, name='bn5c_branch2b')
        net = Conv2d(net, 2048, (1, 1), (1, 1), b_init=None, name='res5c_branch2c')
        net = BatchNormLayer(net, 0.999, 0.00001, None, istrain, name='bn5c_branch2c')
        res5c= ElementwiseLayer([relu5b, net], combine_fn=tf.add, name='res5c')
        relu5c = LambdaLayer(res5c, tf.nn.relu, name='res5c_relu')
        net = Conv2d(relu5c, 1024, (1, 1), (1, 1),act=None ,name='layer1')
        net = BatchNormLayer(net, 0.999, 0.00001, None, istrain, name='layer1_BN')

        layer2x_br1_ConvA=Conv2d(net, 512, (3, 3), (1, 1),act=None ,padding='SAME',name='layer2x_br1_ConvA')
        layer2x_br1_ConvB_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 0], [1, 1], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer2x_br1_ConvB_pad')
        layer2x_br1_ConvB = Conv2d(layer2x_br1_ConvB_input, 512, (2, 3), (1, 1), act=None, padding='VALID',
                                   name='layer2x_br1_ConvB')
        layer2x_br1_ConvC_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer2x_br1_ConvC_pad')
        layer2x_br1_ConvC = Conv2d(layer2x_br1_ConvC_input, 512, (3, 2), (1, 1), act=None, padding='VALID',
                                   name='layer2x_br1_ConvC')
        layer2x_br1_ConvD_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [0, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer2x_br1_ConvD_pad')
        layer2x_br1_ConvD = Conv2d(layer2x_br1_ConvD_input, 512, (2, 2), (1, 1), act=None, padding='VALID',
                                   name='layer2x_br1_ConvD')
        layer2x_br1_Left=LambdaLayer([layer2x_br1_ConvA,layer2x_br1_ConvB], fn=interleave, fn_args={'axis':1},
                                              name='layer2x_br1_Left')
        layer2x_br1_Right=LambdaLayer([layer2x_br1_ConvC,layer2x_br1_ConvD], fn=interleave, fn_args={'axis':1},
                                              name='layer2x_br1_Right')
        layer2x_br1_Out=LambdaLayer([layer2x_br1_Left,layer2x_br1_Right], fn=interleave, fn_args={'axis':2},
                                              name='layer2x_br1_Out')
        layer2x_br1_Bn=BatchNormLayer(layer2x_br1_Out,0.999,0.00001,tf.nn.relu,is_train=istrain,name='layer2x_br1_BN')
        layer2x_Conv=Conv2d(layer2x_br1_Bn,512,(3,3),(1,1),None,'SAME',name='layer2x_Conv')
        layer2x_BN = BatchNormLayer(layer2x_Conv,0.999,0.00001,tf.nn.relu,is_train=istrain ,name='layer2x_BN')
        layer2x_br2_ConvA=Conv2d(net, 512, (3, 3), (1, 1),act=None ,padding='SAME',name='layer2x_br2_ConvA')
        layer2x_br2_ConvB_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 0], [1, 1], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer2x_br2_ConvB_pad')
        layer2x_br2_ConvB = Conv2d(layer2x_br2_ConvB_input, 512, (2, 3), (1, 1), act=None, padding='VALID',
                                   name='layer2x_br2_ConvB')
        layer2x_br2_ConvC_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer2x_br2_ConvC_pad')
        layer2x_br2_ConvC = Conv2d(layer2x_br2_ConvC_input, 512, (3, 2), (1, 1), act=None, padding='VALID',
                                   name='layer2x_br2_ConvC')
        layer2x_br2_ConvD_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [0, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer2x_br2_ConvD_pad')
        layer2x_br2_ConvD = Conv2d(layer2x_br2_ConvD_input, 512, (2, 2), (1, 1), act=None, padding='VALID',
                                   name='layer2x_br2_ConvD')
        layer2x_br2_Left=LambdaLayer([layer2x_br2_ConvA,layer2x_br2_ConvB], fn=interleave, fn_args={'axis':1},
                                              name='layer2x_br2_Left')
        layer2x_br2_Right=LambdaLayer([layer2x_br2_ConvC,layer2x_br2_ConvD], fn=interleave, fn_args={'axis':1},
                                              name='layer2x_br2_Right')
        layer2x_br2_Out=LambdaLayer([layer2x_br2_Left,layer2x_br2_Right], fn=interleave, fn_args={'axis':2},
                                              name='layer2x_br2_Out')
        layer2x_br2_Bn=BatchNormLayer(layer2x_br2_Out,0.999,0.00001,tf.nn.relu,is_train=istrain,name='layer2x_br2_BN')
        layer2x_Out=ElementwiseLayer([layer2x_BN, layer2x_br2_Bn], combine_fn=tf.add, name='layer2x_Sum')
        net=LambdaLayer(layer2x_Out, tf.nn.relu, name='layer2x_Relu')
        layer4x_br1_ConvA=Conv2d(net, 256, (3, 3), (1, 1),act=None ,padding='SAME',name='layer4x_br1_ConvA')

        layer4x_br1_ConvB_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 0], [1, 1], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer4x_br1_ConvB_pad')
        layer4x_br1_ConvB = Conv2d(layer4x_br1_ConvB_input, 256, (2, 3), (1, 1), act=None, padding='VALID',
                                   name='layer4x_br1_ConvB')
        layer4x_br1_ConvC_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer4x_br1_ConvC_pad')
        layer4x_br1_ConvC = Conv2d(layer4x_br1_ConvC_input, 256, (3, 2), (1, 1), act=None, padding='VALID',
                                   name='layer4x_br1_ConvC')
        layer4x_br1_ConvD_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [0, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer4x_br1_ConvD_pad')
        layer4x_br1_ConvD = Conv2d(layer4x_br1_ConvD_input, 256, (2, 2), (1, 1), act=None, padding='VALID',
                                   name='layer4x_br1_ConvD')
        layer4x_br1_Left=LambdaLayer([layer4x_br1_ConvA,layer4x_br1_ConvB], fn=interleave, fn_args={'axis':1},
                                              name='layer4x_br1_Left')
        layer4x_br1_Right=LambdaLayer([layer4x_br1_ConvC,layer4x_br1_ConvD], fn=interleave, fn_args={'axis':1},
                                              name='layer4x_br1_Right')
        layer4x_br1_Out=LambdaLayer([layer4x_br1_Left,layer4x_br1_Right], fn=interleave, fn_args={'axis':2},
                                              name='layer4x_br1_Out')
        layer4x_br1_Bn=BatchNormLayer(layer4x_br1_Out,0.999,0.00001,tf.nn.relu,is_train=istrain,name='layer4x_br1_BN')
        layer4x_Conv=Conv2d(layer4x_br1_Bn,256,(3,3),(1,1),None,'SAME',name='layer4x_Conv')
        layer4x_BN = BatchNormLayer(layer4x_Conv,0.999,0.00001,tf.nn.relu,is_train=istrain ,name='layer4x_BN')
        layer4x_br2_ConvA=Conv2d(net, 256, (3, 3), (1, 1),act=None ,padding='SAME',name='layer4x_br2_ConvA')
        layer4x_br2_ConvB_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 0], [1, 1], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer4x_br2_ConvB_pad')
        layer4x_br2_ConvB = Conv2d(layer4x_br2_ConvB_input, 256, (2, 3), (1, 1), act=None, padding='VALID',
                                   name='layer4x_br2_ConvB')
        layer4x_br2_ConvC_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer4x_br2_ConvC_pad')
        layer4x_br2_ConvC = Conv2d(layer4x_br2_ConvC_input, 256, (3, 2), (1, 1), act=None, padding='VALID',
                                   name='layer4x_br2_ConvC')
        layer4x_br2_ConvD_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [0, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer4x_br2_ConvD_pad')
        layer4x_br2_ConvD = Conv2d(layer4x_br2_ConvD_input, 256, (2, 2), (1, 1), act=None, padding='VALID',
                                   name='layer4x_br2_ConvD')
        layer4x_br2_Left=LambdaLayer([layer4x_br2_ConvA,layer4x_br2_ConvB], fn=interleave, fn_args={'axis':1},
                                              name='layer4x_br2_Left')
        layer4x_br2_Right=LambdaLayer([layer4x_br2_ConvC,layer4x_br2_ConvD], fn=interleave, fn_args={'axis':1},
                                              name='layer4x_br2_Right')
        layer4x_br2_Out=LambdaLayer([layer4x_br2_Left,layer4x_br2_Right], fn=interleave, fn_args={'axis':2},
                                              name='layer4x_br2_Out')
        layer4x_br2_Bn=BatchNormLayer(layer4x_br2_Out,0.999,0.00001,tf.nn.relu,is_train=istrain,name='layer4x_br2_BN')
        layer4x_Out=ElementwiseLayer([layer4x_BN, layer4x_br2_Bn], combine_fn=tf.add, name='layer4x_Sum')
        net=LambdaLayer(layer4x_Out, tf.nn.relu, name='layer4x_Relu')
        layer8x_br1_ConvA=Conv2d(net, 128, (3, 3), (1, 1),act=None ,padding='SAME',name='layer8x_br1_ConvA')

        layer8x_br1_ConvB_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 0], [1, 1], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer8x_br1_ConvB_pad')
        layer8x_br1_ConvB = Conv2d(layer8x_br1_ConvB_input, 128, (2, 3), (1, 1), act=None, padding='VALID',
                                   name='layer8x_br1_ConvB')
        layer8x_br1_ConvC_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer8x_br1_ConvC_pad')
        layer8x_br1_ConvC = Conv2d(layer8x_br1_ConvC_input, 128, (3, 2), (1, 1), act=None, padding='VALID',
                                   name='layer8x_br1_ConvC')
        layer8x_br1_ConvD_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [0, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer8x_br1_ConvD_pad')
        layer8x_br1_ConvD = Conv2d(layer8x_br1_ConvD_input, 128, (2, 2), (1, 1), act=None, padding='VALID',
                                   name='layer8x_br1_ConvD')
        layer8x_br1_Left=LambdaLayer([layer8x_br1_ConvA,layer8x_br1_ConvB], fn=interleave, fn_args={'axis':1},
                                              name='layer8x_br1_Left')
        layer8x_br1_Right=LambdaLayer([layer8x_br1_ConvC,layer8x_br1_ConvD], fn=interleave, fn_args={'axis':1},
                                              name='layer8x_br1_Right')
        layer8x_br1_Out=LambdaLayer([layer8x_br1_Left,layer8x_br1_Right], fn=interleave, fn_args={'axis':2},
                                              name='layer8x_br1_Out')
        layer8x_br1_Bn=BatchNormLayer(layer8x_br1_Out,0.999,0.00001,tf.nn.relu,is_train=istrain,name='layer8x_br1_BN')
        layer8x_Conv=Conv2d(layer8x_br1_Bn,128,(3,3),(1,1),None,'SAME',name='layer8x_Conv')
        layer8x_BN = BatchNormLayer(layer8x_Conv,0.999,0.00001,tf.nn.relu,is_train=istrain ,name='layer8x_BN')
        layer8x_br2_ConvA=Conv2d(net, 128, (3, 3), (1, 1),act=None ,padding='SAME',name='layer8x_br2_ConvA')
        layer8x_br2_ConvB_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 0], [1, 1], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer8x_br2_ConvB_pad')
        layer8x_br2_ConvB = Conv2d(layer8x_br2_ConvB_input, 128, (2, 3), (1, 1), act=None, padding='VALID',
                                   name='layer8x_br2_ConvB')
        layer8x_br2_ConvC_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer8x_br2_ConvC_pad')
        layer8x_br2_ConvC = Conv2d(layer8x_br2_ConvC_input, 128, (3, 2), (1, 1), act=None, padding='VALID',
                                   name='layer8x_br2_ConvC')
        layer8x_br2_ConvD_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [0, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer8x_br2_ConvD_pad')
        layer8x_br2_ConvD = Conv2d(layer8x_br2_ConvD_input, 128, (2, 2), (1, 1), act=None, padding='VALID',
                                   name='layer8x_br2_ConvD')
        layer8x_br2_Left=LambdaLayer([layer8x_br2_ConvA,layer8x_br2_ConvB], fn=interleave, fn_args={'axis':1},
                                              name='layer8x_br2_Left')
        layer8x_br2_Right=LambdaLayer([layer8x_br2_ConvC,layer8x_br2_ConvD], fn=interleave, fn_args={'axis':1},
                                              name='layer8x_br2_Right')
        layer8x_br2_Out=LambdaLayer([layer8x_br2_Left,layer8x_br2_Right], fn=interleave, fn_args={'axis':2},
                                              name='layer8x_br2_Out')
        layer8x_br2_Bn=BatchNormLayer(layer8x_br2_Out,0.999,0.00001,tf.nn.relu,is_train=istrain,name='layer8x_br2_BN')
        layer8x_Out=ElementwiseLayer([layer8x_BN, layer8x_br2_Bn], combine_fn=tf.add, name='layer8x_Sum')
        net=LambdaLayer(layer8x_Out, tf.nn.relu, name='layer8x_Relu')
        layer16x_br1_ConvA=Conv2d(net, 64, (3, 3), (1, 1),act=None ,padding='SAME',name='layer16x_br1_ConvA')
        layer16x_br1_ConvB_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 0], [1, 1], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer16x_br1_ConvB_pad')
        layer16x_br1_ConvB = Conv2d(layer16x_br1_ConvB_input, 64, (2, 3), (1, 1), act=None, padding='VALID',
                                   name='layer16x_br1_ConvB')
        layer16x_br1_ConvC_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer16x_br1_ConvC_pad')
        layer16x_br1_ConvC = Conv2d(layer16x_br1_ConvC_input, 64, (3, 2), (1, 1), act=None, padding='VALID',
                                   name='layer16x_br1_ConvC')
        layer16x_br1_ConvD_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [0, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer16x_br1_ConvD_pad')
        layer16x_br1_ConvD = Conv2d(layer16x_br1_ConvD_input, 64, (2, 2), (1, 1), act=None, padding='VALID',
                                   name='layer16x_br1_ConvD')
        layer16x_br1_Left=LambdaLayer([layer16x_br1_ConvA,layer16x_br1_ConvB], fn=interleave, fn_args={'axis':1},
                                              name='layer16x_br1_Left')
        layer16x_br1_Right=LambdaLayer([layer16x_br1_ConvC,layer16x_br1_ConvD], fn=interleave, fn_args={'axis':1},
                                              name='layer16x_br1_Right')
        layer16x_br1_Out=LambdaLayer([layer16x_br1_Left,layer16x_br1_Right], fn=interleave, fn_args={'axis':2},
                                              name='layer16x_br1_Out')
        layer16x_br1_Bn=BatchNormLayer(layer16x_br1_Out,0.999,0.00001,tf.nn.relu,is_train=istrain,name='layer16x_br1_BN')
        layer16x_Conv=Conv2d(layer16x_br1_Bn,64,(3,3),(1,1),None,'SAME',name='layer16x_Conv')
        layer16x_BN = BatchNormLayer(layer16x_Conv,0.999,0.00001,tf.nn.relu,is_train=istrain ,name='layer16x_BN')
        layer16x_br2_ConvA=Conv2d(net, 64, (3, 3), (1, 1),act=None ,padding='SAME',name='layer16x_br2_ConvA')
        layer16x_br2_ConvB_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 0], [1, 1], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer16x_br2_ConvB_pad')
        layer16x_br2_ConvB = Conv2d(layer16x_br2_ConvB_input, 64, (2, 3), (1, 1), act=None, padding='VALID',
                                   name='layer16x_br2_ConvB')
        layer16x_br2_ConvC_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [1, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer16x_br2_ConvC_pad')
        layer16x_br2_ConvC = Conv2d(layer16x_br2_ConvC_input, 64, (3, 2), (1, 1), act=None, padding='VALID',
                                   name='layer16x_br2_ConvC')
        layer16x_br2_ConvD_input = LambdaLayer(net, fn=tf.pad, fn_args={'paddings': [[0, 0], [0, 1], [1, 0], [0, 0]],
                                                                       'mode': "CONSTANT"},
                                              name='layer16x_br2_ConvD_pad')
        layer16x_br2_ConvD = Conv2d(layer16x_br2_ConvD_input, 64, (2, 2), (1, 1), act=None, padding='VALID',
                                   name='layer16x_br2_ConvD')
        layer16x_br2_Left=LambdaLayer([layer16x_br2_ConvA,layer16x_br2_ConvB], fn=interleave, fn_args={'axis':1},
                                              name='layer16x_br2_Left')
        layer16x_br2_Right=LambdaLayer([layer16x_br2_ConvC,layer16x_br2_ConvD], fn=interleave, fn_args={'axis':1},
                                              name='layer16x_br2_Right')
        layer16x_br2_Out=LambdaLayer([layer16x_br2_Left,layer16x_br2_Right], fn=interleave, fn_args={'axis':2},
                                              name='layer16x_br2_Out')
        layer16x_br2_Bn=BatchNormLayer(layer16x_br2_Out,0.999,0.00001,tf.nn.relu,is_train=istrain,name='layer16x_br2_BN')
        layer16x_Out=ElementwiseLayer([layer16x_BN, layer16x_br2_Bn], combine_fn=tf.add, name='layer16x_Sum')
        net=LambdaLayer(layer16x_Out, tf.nn.relu, name='layer16x_Relu')
        net=DropoutLayer(net,0.5,True,is_train=istrain,name='drop')
        net=Conv2d(net,1,(3,3),(1,1),name='ConvPred')

        return net,net.outputs

def HuberLoss(x,huber_c):
    # huber_c=tf.expand_dims(huber_c,-1)
    # huber_c=tf.expand_dims(huber_c,-1)
    # huber_c=tf.expand_dims(huber_c,-1)
    # huber_c=tf.multiply(tf.ones_like(x),huber_c)*0.2
    huber_c=huber_c*0.2
    return tf.where(tf.less_equal(tf.abs(x),huber_c),tf.abs(x),tf.multiply(tf.pow(x, 2.0)+tf.pow(huber_c, 2.0),tf.div(.5,huber_c+0.000001)))

def cal_loss(logits,labels):
    huber_c=  tf.reduce_max(tf.abs(labels-logits),[1,2,3],True)
    # return tf.clip_by_value(tf.reduce_mean(HuberLoss(labels-logits,huber_c)),0.000001,10000000.)
    return tf.reduce_mean(HuberLoss(labels-logits,huber_c))
    # return tf.reduce_mean(tf.losses.mean_squared_error(labels/tf.reduce_max(labels,[1,2,3],True),logits/tf.reduce_max(logits,[1,2,3],True)))
    # return  tf.losses.huber_loss(labels,logits)
def cal_acc(logits,labels):
    #tf.reduce_max(labels,[1,2,3],True))*.005
    return tf.reduce_mean( tf.cast( tf.less_equal(tf.abs(labels-logits),tf.multiply(tf.ones_like(labels),10.)),tf.float32))
def cal_mean_var(logits,labels):
    #tf.reduce_max(labels,[1,2,3],True))*.005
    vector=tf.reshape(logits-labels,[-1])
    mean,var=tf.nn.moments(vector,0)
    return mean,var
def cal_mean_var_alter(logits,labels):
    #tf.reduce_max(labels,[1,2,3],True))*.005
    temp_logits=logits[:,24:104,32:128,:]
    temp_labels=labels[:,24:104,32:128,:]
    vector=tf.reshape(temp_logits-temp_labels,[-1])
    mean,var=tf.nn.moments(vector,0)
    return mean,var

if __name__ == '__main__':

    img_train,label_train  = read_and_decode(TRAIN_TFRECORD)
    img_test,label_test  = read_and_decode(TEST_TFRECORD)
    img_train_batch,  label_train_batch = tf.train.shuffle_batch(
        [img_train, label_train], batch_size=batch_size, capacity=BATCH_CAPACITY,
        min_after_dequeue=MIN_AFTER_DEQU)
    img_test_batch,  label_test_batch = tf.train.shuffle_batch(
        [img_test,label_test], batch_size=batch_size, capacity=BATCH_CAPACITY,
        min_after_dequeue=MIN_AFTER_DEQU)
    net,logits_train=inference(img_train_batch,True,None)
    _,logits_test=inference(img_test_batch,False,True)
    loss_train=cal_loss(logits_train,label_train_batch)
    loss_test=cal_loss(logits_test,label_test_batch)
    acc_test=cal_acc(logits_test,label_test_batch)
    acc_train=cal_acc(logits_train,label_train_batch)
    train_mean,train_var=cal_mean_var(logits_train,label_train_batch)
    print(train_mean,train_var)
    test_mean,test_var=cal_mean_var_alter(logits_test,label_test_batch)
    all_var = variables._all_saveable_objects().copy()
    lens_all=len(all_var)
    for _ in range(lens_all-2):
        del all_var[0]

    print(all_var)
    pre_train = tf.train.MomentumOptimizer(0.005,momentum=0.9).minimize(loss_train,var_list=all_var)
    global_step=tf.train.create_global_step()
    global_step=tf.train.get_global_step()
    learning_rate=tf.train.exponential_decay(learning_rt,global_step,
                                           10000, 0.9, staircase=True)
    train = tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(loss_train,global_step=global_step)
    # train = tf.train.AdamOptimizer(learning_rt).minimize(loss_train)
    tf.summary.scalar('loss_train', loss_train)
    tf.summary.scalar('acc_train', acc_train)
    merged = tf.summary.merge_all()

    all_var_list = variables._all_saveable_objects()
    with tf.Session() as sess:
        trainwrite = tf.summary.FileWriter(logpath, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())

        maingraph = tf.get_default_graph()
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        if restore_from_fcrn:
            print('\nStart Restore')
            saver_varlist={}
            for key in var_to_shape_map:
                graph_key=re.sub(r'mean', 'moving_mean', key)
                graph_key=re.sub(r'variance', 'moving_variance', graph_key)
                graph_key=re.sub(r'weights', 'kernel', graph_key)
                graph_key=re.sub(r'biases', 'bias', graph_key)
                graph_key=re.sub(r'offset', 'beta', graph_key)
                graph_key=re.sub(r'scale', 'gamma', graph_key)
                graph_key='model/'+graph_key+':0'
                if not (re.search(r'ConvPred',graph_key)):
                    saver_varlist[key]=maingraph.get_tensor_by_name(graph_key)
            print(saver_varlist)
            saver = tf.train.Saver(var_list=saver_varlist)
            saver.restore(sess, checkpoint_path)
            saver = tf.train.Saver()
            print('\nEnd Restore')
        elif restore_from_ckpt:
            print('\nStart Restore')
            vlist=tf.trainable_variables()
            print(vlist)
            saver = tf.train.Saver(var_list=vlist)
            saver.restore(sess, ckpt_path)
            saver = tf.train.Saver()
            print('\nEnd Restore')
        else:
            saver = tf.train.Saver()
        run_cycle=0
        print('\nStart Training')
        try:
            while not coord.should_stop():
                l_tall = 0
                a_tall = 0
                mean_all=0
                var_all=0
                if need_pretrain:
                    for train_c in range(TRAIN_CYCLE):
                        _, l_train, a_train,train_m ,train_v= sess.run([pre_train, loss_train, acc_train,train_mean,train_var])
                        l_tall += l_train
                        a_tall += a_train
                        mean_all += train_m
                        var_all += train_v
                        if (train_c + 1) % 100 == 0:
                            print('train_loss:%f' % (l_tall / 100.))
                            print('train_acc:%f' % (a_tall / 100.))
                            print('train_mean_all:%f' % (mean_all / 100.))
                            print('train_var_all:%f' % (var_all / 100.))
                            l_tall = 0
                            a_tall = 0
                            mean_all = 0
                            var_all = 0
                    print('Finish PreTrain')
                while run_cycle < MAX_Cycle:
                    run_cycle+=1
                    l_tall=0
                    a_tall=0
                    l_teall=0
                    a_teall=0
                    mean_all = 0
                    var_all = 0
                    mean_tall = 0
                    var_tall = 0
                    for train_c in range(TRAIN_CYCLE):
                        _,l_train,a_train,train_m,train_v=sess.run([train,loss_train,acc_train,train_mean,train_var])
                        l_tall+=l_train
                        a_tall+=a_train
                        mean_all += train_m
                        var_all += train_v
                        if (train_c+1)%100==0:
                            print('train_loss:%f'%(l_tall/100.))
                            print('train_acc:%f'%(a_tall/100.))
                            print('train_mean_all:%f' % (mean_all / 100.))
                            print('train_var_all:%f' % (var_all / 100.))
                            l_tall = 0
                            a_tall = 0
                        if (train_c+1)%500==0:
                            result_merged=sess.run(merged)
                            trainwrite.add_summary(result_merged, run_cycle*TRAIN_CYCLE+train_c)
                    for test_c in range(TEST_CYCLE):
                        l_test,a_test,test_m,test_v=sess.run([loss_test,acc_test,test_mean,test_var])
                        l_teall+=l_test
                        a_teall+=a_test
                        mean_tall += test_m
                        var_tall += test_v
                        if (test_c+1)%TEST_CYCLE==0:
                            print('------------------')
                            print('test_loss:%f'%(l_teall/TEST_CYCLE))
                            print('test_acc:%f'%(a_teall/TEST_CYCLE))
                            print('train_mean_all:%f' % (mean_tall / TEST_CYCLE))
                            print('train_var_all:%f' % (var_tall / TEST_CYCLE))
                            print('------------------')
                            l_teall = 0
                            l_teall = 0
                    saver.save(sess, savepath+ str(run_cycle) + '.ckpt')

        except tf.errors.OutOfRangeError:
            print('Done training!!!')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
