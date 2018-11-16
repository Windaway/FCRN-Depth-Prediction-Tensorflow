import tensorflow as tf
import os
from PIL import Image
import numpy as np

height = 318
width = 424
depth_height=318
depth_width=424
trainwriter= tf.python_io.TFRecordWriter("train.tfrecords")
rgbdir = './rgb_train/'
depthdir = './depth_train/'
eptlist=[]  #
for img_name in os.listdir(rgbdir):
    if os.path.splitext(img_name)[1] == '.png':
        if os.path.isfile(os.path.join(depthdir, img_name)):
            eptlist.append(img_name)
lens=len(eptlist)
print(lens)
indexar=np.arange(lens)

randindex=np.random.permutation(indexar)
for indexx in randindex:
    filename = eptlist[indexx]
    imgraw = Image.open(rgbdir+filename).convert('RGB')
    imgraw = imgraw.resize((width, height),Image.BILINEAR)
    imgraw = imgraw.tobytes()
    imglabel = Image.open(depthdir+filename).convert('F')
    imglabel = imglabel.resize((depth_width, depth_height),Image.BILINEAR)
    imglabel = imglabel.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgraw])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imglabel]))
    }))
    trainwriter.write(example.SerializeToString())
trainwriter.close()

testwriter= tf.python_io.TFRecordWriter("test.tfrecords")
rgbdir = './rgb_test/'
depthdir = './depth_test/'
eptlist=[]
for img_name in os.listdir(rgbdir):
    if os.path.splitext(img_name)[1] == '.png':
        if os.path.isfile(depthdir+img_name):
            eptlist.append(img_name)
lens=len(eptlist)
print(lens)
indexar=np.arange(lens)
randindex=np.random.permutation(indexar)
for indexx in randindex:
    filename=eptlist[indexx]
    imgraw=Image.open(rgbdir+filename).convert('RGB')
    imgraw = imgraw.resize((width, height),Image.BILINEAR)
    imgraw = imgraw.tobytes()
    imglabel=Image.open(depthdir+filename).convert('F')
    imglabel = imglabel.resize((depth_width, depth_height),Image.BILINEAR)
    imglabel = imglabel.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgraw])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imglabel]))
    }))
    testwriter.write(example.SerializeToString())
testwriter.close()