from .data_pipeline import TfReader, TfWriter, DataPipeline
import cv2


class DCGANReader(TfReader):
    def __init__(self):
        TfReader.__init__(self, 32, 20, 1)


class DCGANWriter(TfWriter):
    def __init__(self):
        TfWriter.__init__(self, 32, 20, 1)


def DCGAN_get_pipeline(sess, batch_size, filename):
    stream = DataPipeline('tfrecords', './data')
    return stream.tfrecords2imgs(sess, filename, batch_size, DCGANReader(), cv2.IMREAD_GRAYSCALE)
