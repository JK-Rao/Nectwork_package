# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      test2.py
# @Software  PyCharm
# @Author    JK.Rao

from .DCGAN_line import DCGANLine


def train_model(name, inster_number, annotion):
    if name == 'DCGAN':
        line = DCGANLine(inster_number, annotion, 128, 64)
        line.structure_train_context()
