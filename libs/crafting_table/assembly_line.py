import tensorflow as tf
import os


class AssemblyLine(object):
    def __init__(self, config, network):
        self.sess = tf.Session(config=config)
        self.network = network
        self.iter_num = 0
        self.summary_writer = None

    def create_summary(self, log_path):
        summ_dict = self.network.get_summary()
        merged = None
        writer = None
        for key in summ_dict:
            tf.summary.scalar(key, summ_dict[key])
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(log_path, self.sess.graph)
            self.summary_writer = writer
        return {'merged': merged,
                'summary_writer': writer}

    def write_summary(self, mg):
        with self.sess:
            self.summary_writer.add_summary(mg, self.iter_num)

    def close_summary_writer(self):
        if self.summary_writer is None:
            print('Error in close writer...')
        else:
            self.summary_writer.close()

    def structure_train_context(self):
        raise NotImplementedError('Must be subclassed.')

    def restore_test_context(self):
        raise NotImplementedError('Must be subclassed.')

    def get_saver(self, vars, max_to_keep=100):
        return tf.train.Saver(vars, max_to_keep=max_to_keep)

    def save_model(self, saver, save_path_name, write_meta_graph=True):
        with self.sess:
            saver.save(self.sess, save_path_name, write_meta_graph=write_meta_graph)


