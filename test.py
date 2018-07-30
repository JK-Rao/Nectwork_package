# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      test.py
# @Software  PyCharm
# @Author    JK.Rao

import tensorflow as tf
from libs.logger.data_pipeline import TfReader
import cv2


def tet():
    recorder = TfReader(32, 20, 1)
    images=recorder.load_sample('./data/train/train_num0.tfrecords', 128, None)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    img_data = sess.run(images)

    # for i in range(128):
    #     cv2.imshow('test',img_data[i])
    #     if cv2.waitKey()==ord('q'):
    #         break
    coord.request_stop()
    coord.join(threads)
    sess.close()

with tf.Session() as sess:
    tet()
with tf.Session() as sess:
    tet()
