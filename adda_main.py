import os

import tensorflow as tf
import numpy as np
from network.loss_functions import huber_loss, mse_loss
from network.network import C3F2, C3F2_REINFORCE_with_baseline
import network.adda_tf as adda
from numpy import linalg as LA

import numpy as np
def read_npz(x):
    npzs = np.load(x)

    xs = npzs['xs']
    ys = npzs['ys']
    return xs,ys

def dataset_itr(Source=True):
    if Source:
        path = 'data_collection/Target/'
    else:
        path = 'data_collection/Source/'
    data_list = [path+x for x in os.listdir(path)[1:100]]
    data_list = list(map(read_npz ,data_list))
    return data_list

class initialize_network_DeepQLearning():
    def __init__(self, models_path,  vehicle_name='drone0'):
        self.g = tf.Graph()
        self.vehicle_name = vehicle_name
        self.iter=0
        self.models_path = models_path
        self.first_frame = True
        self.last_frame = []
        with self.g.as_default():
            # stat_writer_path = './adda/return_plot/'
            # loss_writer_path = './adda/return_plot/'+ self.vehicle_name + '/loss/'
            # self.stat_writer = tf.summary.FileWriter(stat_writer_path)
            # name_array = 'D:/train/loss'+'/'+name
            # self.loss_writer = tf.summary.FileWriter(loss_writer_path)


            self.batch_size = tf.placeholder(tf.int32, shape=())
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.X1 = tf.placeholder(tf.float32, [None, 103,103, 3], name='States')
            self.X1T = tf.placeholder(tf.float32, [None, 103,103, 3], name='StatesT')


            self.X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1)
            self.XT = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1T)
            # self.X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1)
            self.Y = tf.placeholder(tf.float32, shape=[None], name='Qvals')
            self.YT = tf.placeholder(tf.float32, shape=[None], name='Qvals')
            self.Yactions = tf.placeholder(tf.int32, shape=[None], name='Actions')
            self.YTactions = tf.placeholder(tf.int32, shape=[None], name='Actions')

            # create graph
            nn = adda.ADDA(25)
            # for source domain
            self.feat_s = nn.s_encoder(self.X, reuse=False, trainable=False)
            self.logits_s = nn.classifier(self.feat_s, reuse=False, trainable=False)
            self.disc_s = nn.discriminator(self.feat_s, reuse=False)

            # for target domain
            self.feat_t = nn.t_encoder(self.XT, reuse=False)
            self.logits_t = nn.classifier(self.feat_t, reuse=True, trainable=False)
            self.disc_t = nn.discriminator(self.feat_t, reuse=True)

            # build loss
            self.g_loss, self.d_loss = nn.build_ad_loss(self.disc_s, self.disc_t)
            # g_loss,d_loss = nn.build_w_loss(disc_s,disc_t)

            # create optimizer for two task
            self.var_t_en = tf.trainable_variables(nn.t_e)
            self.optim_g = tf.train.AdamOptimizer(0.00001, beta1=0.5, beta2=0.999).minimize(self.g_loss, var_list=self.var_t_en)

            self.var_d = tf.trainable_variables(nn.d)
            self.optim_d = tf.train.AdamOptimizer(0.00001, beta1=0.5, beta2=0.999).minimize(self.d_loss, var_list=self.var_d)






            # create acuuracy op with training batch
            self.acc_tr_s = nn.eval( self.logits_s, self.Yactions)
            self.acc_tr_t = nn.eval( self.logits_t, self.YTactions)






            # create source saver for restore s_encoder
            encoder_path = tf.train.latest_checkpoint(models_path + "/encoder/")
            classifier_path = tf.train.latest_checkpoint(models_path + "/classifier/")

            if encoder_path is None:
                raise ValueError("Don't exits in this dir")
            if classifier_path is None:
                raise ValueError("Don't exits in this dir")


            self.source_var = tf.contrib.framework.list_variables(encoder_path)

            self.var_s_g = tf.global_variables(scope=nn.s_e)
            self.var_c_g = tf.global_variables(scope=nn.c)
            self. var_t_g = tf.trainable_variables(scope=nn.t_e)

            self.encoder_saver = tf.train.Saver(var_list=self.var_s_g)
            self.classifier_saver = tf.train.Saver(var_list=self.var_c_g)

            self.encoder_saver_target = tf.train.Saver(var_list=self.var_s_g)
            dict_var = {}
            for i in self.source_var:
                for j in self.var_t_g:
                    if i[0][1:] in j.name[1:]:
                        dict_var[i[0]] = j
                        # print(dict_var)
            self.fine_turn_saver = tf.train.Saver(var_list=dict_var)


            # dict_var2 = {}
            # for i in self.var_t_g:
            #     for j in self.source_var:
            #         if i[0][1:] in j.name[1:]:
            #             dict_var[i[0]] = j
                        # print(dict_var)
            # self.fine_turn_saver2= tf.train.Saver(var_list=dict_var2)
            # self.fine_turn_saver = tf.train.Saver(var_list=dict_var)
            # assert False
            # create this model saver
            self.best_saver = tf.train.Saver(max_to_keep=3)

            self.sess = tf.InteractiveSession()
            self.merge = tf.summary.merge_all()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            self.sess.graph.finalize()
            self.filewriter = tf.summary.FileWriter(logdir="./adda/logs/", graph=self.sess.graph)

        self.load_network()
        print("model init successfully!")

    def get_vars(self):
        return self.sess.run(self.all_vars)

    def train_step(self,i,x,y,xt,yt):


        if i % 100 == 0 :
            # self.sess.run([s_init, t_init])

            # _, d_loss,_, g_loss,\
            s_acc,t_acc, merge = self.sess.run(
                [
                    # self.optim_d, self.d_loss,self.optim_g, self.g_loss,
                    self.acc_tr_s,self.acc_tr_t, self.merge],feed_dict={self.X1:x,self.X1T:xt,self.Yactions:y,self.YTactions:yt })
            print("epoch: %d, source accuracy: %.4f, target accuracy: %.4f, " % (
                i, s_acc, t_acc))

        _, d_loss_, = self.sess.run([self.optim_d, self.d_loss],feed_dict={self.X1:x,self.X1T:xt,self.Yactions:y,self.YTactions:yt })
        _, g_loss_, merge_ = self.sess.run([self.optim_g, self.g_loss, self.merge],feed_dict={self.X1:x,self.X1T:xt,self.Yactions:y,self.YTactions:yt })
        if i % 20 == 0:
            print("step:{},g_loss:{:.4f},d_loss:{:.4f}".format(i, g_loss_, d_loss_))

        self.filewriter.add_summary(merge_, global_step=i)

    def Q_val(self, xs):
        target = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        pred =  self.sess.run(self.predict,
                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0, self.X1: xs,
                                        self.target: target, self.actions: actions})

        return pred


    def train_n(self, xs, ys, actions, batch_size, dropout_rate, lr, epsilon, iter):
        self.iter=iter
        _, loss, Q = self.sess.run([self.train, self.loss, self.predict],
                                   feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.X1: xs,
                                              self.target: ys, self.actions: actions})

        # np.savez('./data_collection/target/iter_'+str(self.iter),xs=xs,ys=ys,Q=Q)

        meanQ = np.mean(Q)
        maxQ = np.max(Q)
        # np.savez('./data_collection/iter_'+str(iter),xs=xs,ys=ys)

        # Log to tensorboard
        self.log_to_tensorboard(tag='Loss', group=self.vehicle_name, value=LA.norm(loss) / batch_size, index=iter)
        self.log_to_tensorboard(tag='Epsilon', group=self.vehicle_name, value=epsilon, index=iter)
        self.log_to_tensorboard(tag='Learning Rate', group=self.vehicle_name, value=lr, index=iter)
        self.log_to_tensorboard(tag='MeanQ', group=self.vehicle_name, value=meanQ, index=iter)
        self.log_to_tensorboard(tag='MaxQ', group=self.vehicle_name, value=maxQ, index=iter)


    def action_selection(self, state):
        target = np.zeros(shape=[state.shape[0]], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[state.shape[0]])
        qvals = self.sess.run(self.predict,
                              feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
                                         self.X1: state,
                                         self.target: target, self.actions: actions})
        # print("qvals",qvals.shape,qvals)
        #
        # print("actions",actions.shape,actions)
        #
        # print("target", target.shape, target)
        if qvals.shape[0] > 1:
            # Evaluating batch
            action = np.argmax(qvals, axis=1)
            # print("selected action", action.shape, action)

        else:
            # Evaluating one sample
            action = np.zeros(1)
            action[0] = np.argmax(qvals)
            # print("selected one action", action.shape, action)

        return action.astype(int)

    def log_to_tensorboard(self, tag, group, value, index):
        summary = tf.Summary()
        tag = group + '/' + tag
        summary.value.add(tag=tag, simple_value=value)
        self.stat_writer.add_summary(summary, index)

    def save_network(self, save_path="", episode=''):
        # save_path = save_path + self.vehicle_name + '/' + self.vehicle_name + '_' + str(episode)
        # self.saver.save(self.sess, save_path)
        # print('Model Saved: ', save_path)

        # self.encoder_saver.save(self.sess,  "./adda/encoder/encoder.ckpt")
        # self.classifier_saver.save(self.sess,  "./adda/classifier/classifier.ckpt")

        # self.best_saver.save(self.sess, "./adda/adda_model.ckpt")

        os.makedirs('./adda/adapted_target/encoder/',exist_ok=True)
        os.makedirs('./adda/adapted_target/classifier/',exist_ok=True)
        self.fine_turn_saver.save(self.sess, "./adda/adapted_target/encoder/encoder.ckpt")
        self.classifier_saver.save(self.sess, "./adda/adapted_target/classifier/classifier.ckpt")

    def load_network(self):
        # self.saver.restore(self.sess, load_path)
        # self.saver.restore(self.sess, load_path)
        # self.encoder_saver.restore(self.sess, "./adda/encoder/encoder.ckpt")
        # self.classifier_saver.restore(self.sess, "./adda/classifier/classifier.ckpt")


        self.encoder_saver.restore(self.sess, self.models_path + "/encoder/encoder.ckpt")
        self.classifier_saver.restore(self.sess,self.models_path + "/classifier/classifier.ckpt")
        # self.fine_turn_saver.restore(self.sess, "./adda/encoder/encoder.ckpt")


model = initialize_network_DeepQLearning("models/trained/Indoor/indoor_updown/Imagenet/e2e/drone0/drone0_user")

# print("te:",len(model.var_c))
# print(source_model.var_c)
num_itr = 100


dset_loaders_source = dataset_itr(True)
dset_loaders_target = dataset_itr(False)
len_train_source = len(dset_loaders_source)
len_train_target = len(dset_loaders_target)




for itr in range(num_itr):

    if itr % len_train_source == 0:
        iter_source = iter(dset_loaders_source)
    if itr % len_train_target == 0:
        iter_target = iter(dset_loaders_target)

    inputs_source, labels_source = next(iter_source)
    inputs_target, labels_target  = next(iter_target)




    model.train_step(itr,inputs_source,labels_source,inputs_target,labels_target)

model.save_network()







# # var_s_g = tf.global_variables(scope=model.all_vars)
# # var_c_g = tf.global_variables(scope=nn.c?)
# # var_t_g = tf.trainable_variables(scope=nn.t_e)
#
#
# print("+++++++++++++++")
# print("s:",len(source_model.all_vars))
# print(source_model.all_vars)
# print("saver_se:",len(source_model.var_s_en))
# print(source_model.var_s_en)
# print("s:",len(source_model.var_c))
# print(source_model.var_c)
#
# print("+++++++++++++++")
# print("var_c :" ,source_model.var_c[1].name, source_model.var_c[1].eval())
#
# source_model.load_network('models/trained/Indoor/indoor_updown_tf_fail_1/Imagenet/e2e/global/global_2204')
#
# print("var_c :" ,source_model.all_vars[9].name, source_model.all_vars[9].eval())
#
# source_model.save_network("./adda/all/all.ckpt")