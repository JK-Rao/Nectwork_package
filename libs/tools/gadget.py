import os


def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
