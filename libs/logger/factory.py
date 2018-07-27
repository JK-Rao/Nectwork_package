from .DCGAN_data_stream import DCGAN_get_pipeline


def get_sample_tensor(model_name, sess, propose, batch_size, filename):
    if model_name == 'DCGAN':
        return DCGAN_get_pipeline(sess, propose, batch_size, filename)
