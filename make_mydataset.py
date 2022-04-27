'''
Author: your name
Date: 2022-04-20 22:10:24
LastEditTime: 2022-04-20 22:37:54
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /NS/make_mydataset.py
'''
import random
import os
from glob import glob
import numpy as np
from PIL import Image
import tensorflow as tf


def list_dir(path):
    return [os.path.join(path,temp) for temp in os.listdir(path)]


def build_testset(path, name):
    image_list=list_dir(path)
    len2 = len(image_list)
    print("len=", len2)
    writer = tf.python_io.TFRecordWriter(name)
    for i in range(len2):

        image = Image.open(image_list[i])
        image = image.resize((256, 128), Image.BILINEAR)
        image = image.convert('RGB')

        image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_bytes = image.tobytes()

        features = {}

        features['image'] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes]))

        tf_features = tf.train.Features(feature=features)

        tf_example = tf.train.Example(features=tf_features)

        tf_serialized = tf_example.SerializeToString()

        writer.write(tf_serialized)

        # flip image
        image = image_flip

        image_bytes = image.tobytes()

        features = {}

        features['image'] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes]))

        tf_features = tf.train.Features(feature=features)

        tf_example = tf.train.Example(features=tf_features)

        tf_serialized = tf_example.SerializeToString()

        writer.write(tf_serialized)

    writer.close()


if __name__=='__main__':
    mydir='mydataset'
    name='./dataset/mydatatest.tfr'
    build_testset(mydir,name)