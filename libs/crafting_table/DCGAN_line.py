from .assembly_line import AssemblyLine
import tensorflow as tf
from ..network.factory import get_network
import numpy as np
from ..logger.factory import get_sample
from ..logger.data_pipeline import DataPipeline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class DCGANLine(AssemblyLine):
    def __init__(self, inster_number, annotion,batch_size):
        AssemblyLine.__init__(self, DCGANLine.get_config(), get_network('DCGAN'))
        self.inster_number = inster_number
        self.annotion = annotion
        self.batch_size=batch_size

        self.Z_dim = 100

    @staticmethod
    def get_config():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return config

    def sample_Z(self,m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def plot(self,samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
            sample = sample + 0.5
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_aspect('equal')
            if self.network.IMG_CHANEL == 1:
                plt.imshow(sample.reshape(32, 20), cmap='Greys_r')
            else:
                plt.imshow(sample.reshape(32, 20, self.network.IMG_CHANEL), cmap='Greys_r')
        return fig
    def structure_train_context(self):
        saver = self.get_saver(self.network.get_trainable_var(self.network.net_name[0]))
        opti_dict=self.network.define_optimizer()
        loss_dict=self.network.structure_loss()
        with self.sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            self.sess.run(tf.global_variables_initializer())
            self.save_model(saver,
                            './model_DCGAN_num%d%s/iter_meta.ckpt' % (self.inster_number, self.annotion))
            summary = self.create_summary('./logs/log_num%d%s/' % (self.inster_number, self.annotion))

            i = 0
            train_D = True
            prepare_save = False
            for iter in range(200000):
                samples = None
                if iter % 1000 == 0:
                    self.iter_num = iter
                    samples = self.sess.run(self.network.get_pred()['gen_im'], feed_dict={
                        self.network.Z: self.sample_Z(16, self.Z_dim), self.network.on_train: False, self.network.batch_size: 16})  # 16*784
                    fig = self.plot(samples)
                    plt.savefig('out_bank_CNN_num%d%s/' % (self.inster_number, self.annotion) + '/{}.png'.format(str(i).zfill(3)),
                                bbox_inches='tight')
                    i += 1
                    plt.close(fig)

                X_mb = get_sample('DCGAN',self.sess,'train',self.batch_size,'train_num')
                # sess.run(D_optimizer, feed_dict={
                #     X: X_mb,
                #     Z: sample_Z(mb_size, Z_dim),
                #     on_train: True,
                #     batch_size: mb_size})
                _, D_loss_curr = self.sess.run([opti_dict['d_opti'], loss_dict['d_loss']], feed_dict={
                    self.network.X: X_mb,
                    self.network.Z: self.sample_Z(self.batch_size, self.Z_dim),
                    self.network.on_train: True,
                    self.network.batch_size: self.batch_size})

                _, G_loss_curr = self.sess.run([opti_dict['g_opti'], loss_dict['g_loss']], feed_dict={
                    self.network.Z: self.sample_Z(self.batch_size, self.Z_dim),
                    self.network.on_train: True,
                    self.network.batch_size: self.batch_size})
                # if iter % 100 == 0:
                #     print('Iter:%d  G_loss:%f,D_loss:%f' % (iter, G_loss_curr, D_loss_curr))
                if iter % 1000 == 0:
                    # overfitting record
                    j = 0
                    print('Iter:%d  G_loss:%f,D_loss:%f' % (iter, G_loss_curr, D_loss_curr))
                    samples = sess.run(gen_im, feed_dict={
                        Z: sample_Z(10000, Z_dim), on_train: False, batch_size: 10000})
                    for line in range(10000):
                        cv2.imwrite('./temp_CNN_num%d/%08d.jpg' % (inster_number, j),
                                    np.round((samples[line, :, :, 0] + 0.5) * 255))
                        j += 1
                    iter_num = 10
                    PATH = './temp_CNN_num%d' % inster_number
                    file_names = os.listdir(PATH)
                    file_names = [os.path.join(PATH, a) for a in file_names]
                    min_dis, ave_dis, _, _ = overfitting_testing.random_sampling(file_names, iter_num)
                    sess.run(tf.assign(Min_distance, min_dis))
                    sess.run(tf.assign(Ave_distance, ave_dis))
                    # loss record
                    X_val = sess.run(X_tensor_val) / 255. - 0.5
                    mg = sess.run(merged, feed_dict={
                        X: X_val,
                        on_train: False,
                        Z: sample_Z(val_size, Z_dim),
                        batch_size: val_size})
                    writer.add_summary(mg, iter)

                gl_step = sess.run(global_step)
                if gl_step % 10000 == 0:
                    saver.save(sess, './model_DCGAN_num%d%s/iter_%d_num%d.ckpt' \
                               % (inster_number, annotion, gl_step, inster_number),
                               write_meta_graph=False)
            writer.close()

            coord.request_stop()
            coord.join(threads)
        sess.close()
