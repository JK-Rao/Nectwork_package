# _*_ coding: utf-8 _*_
# @Time      18-7-30 上午10:11
# @File      model_test.py`
# @Software  PyCharm
# @Author    JK.Rao

from libs.crafting_table.factory import test_model

if __name__ == '__main__':
    model_path = './model_DCGAN_num0'
    model_name = 'iter_meta.ckpt.meta'
    para_name = 'iter_40000.ckpt'
    test_model('DCGAN', model_path, model_name, para_name)
