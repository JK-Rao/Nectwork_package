import DCGAN_data_stream


def get_sample(model_name, sess, batch_size, filename):
    if model_name == 'DCGAN':
        return DCGAN_data_stream.DCGAN_get_pipeline(sess, batch_size, filename)
