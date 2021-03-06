# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      factory.py
# @Software  PyCharm
# @Author    JK.Rao

from .DCGANnet import DCGANnet


def get_network(name):
    if name == 'DCGAN':
        return DCGANnet(['gen', 'dis'], [32, 20, 1])


def get_network_name(name):
    if name == 'DCGAN':
        return ['gen', 'dis']
