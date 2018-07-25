from .DCGAN_data_stream import DCGAN_get_pipeline


def get_sample(model_name, sess, batch_size, filename):
    if model_name == 'DCGAN':
        return DCGAN_get_pipeline(sess, batch_size, filename)
