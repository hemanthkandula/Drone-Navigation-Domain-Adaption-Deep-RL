from copy import deepcopy

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from numpy import linalg as LA

from network import adda
from network.loss_functions import huber_loss2, mse_loss


import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T

import torch.optim as optim
from network.pre_process import  image_train
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

transform_rs = T.Compose([T.ToPILImage(), T.ToTensor()])
# import lr_schedule


class ImageList(Dataset):
    def __init__(self, image_list, transform=None):



        self.imgs = image_list
        self.transform = transform


    def __getitem__(self, index):
        img = self.imgs[index]
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

def get_data(xs):
    xs_new = deepcopy(xs)
    #
    # ds = ImageList(xs,resize)
    # return next(iter(DataLoader(ds, batch_size=len(xs))))
    batch_input = []
    for x in xs_new:  # assuming batch_size=10
        # x.resize((224,224))

        # batch_input.append( cv2.resize(x,(224,224)))
        batch_input.append( np.transpose(x, ( 2, 0, 1)))

    return torch.tensor(batch_input)
class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

# resize = T.Compose([ResizeImage((256,256)),
#                     T.ToTensor()
#                     ])


class initialize_network_DeepQLearning():
    def __init__(self, cfg, name, vehicle_name):
        # self.g = tf.Graph()
        self.vehicle_name = vehicle_name

        self.first_frame = True
        self.last_frame = []
        # with self.g.as_default():
        stat_writer_path = cfg.network_path + self.vehicle_name + '/return_plot/'
        loss_writer_path = cfg.network_path + self.vehicle_name + '/loss' + name + '/'
        self.stat_writer = tf.summary.FileWriter(stat_writer_path)
        # name_array = 'D:/train/loss'+'/'+name
        self.loss_writer = tf.summary.FileWriter(loss_writer_path)
        self.env_type = cfg.env_type
        self.input_size = cfg.input_size
        self.num_actions = cfg.num_actions

        # Placeholders
        # self.batch_size = tf.placeholder(tf.int32, shape=())
        # self.learning_rate = tf.placeholder(tf.float32, shape=())
        # self.X1 = tf.placeholder(tf.float32, [None, cfg.input_size, cfg.input_size, 3], name='States')

        self.source_encoder = adda.ResNetEncoder()
        self.source_classifier = adda.ResNetClassifier(self.num_actions)
        self.source_encoder, self.source_classifier = self.source_encoder.cuda(), self.source_classifier.cuda()

        self.parameter_list = self.source_encoder.get_parameters() + self.source_classifier.get_parameters()
        # self.parameter_list = self.source_encoder.get_parameters() + self.source_classifier.get_parameters()

        self.optimizer = optim.SGD(self.parameter_list, lr= cfg.learning_rate, momentum=0.9,
                                                            weight_decay=0.0005,
                                                            nesterov=True)


        # self.optimizer = optim.Adam(self.parameter_list, lr= cfg.learning_rate, betas=(0.9,0.99))



        self.transform = image_train

        self.optimizer.zero_grad()
        # self.X = tf.image.resize_images(self.X1, (227, 227))

        # self.X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1)
        # self.target = tf.placeholder(tf.float32, shape=[None], name='Qvals')
        # self.actions = tf.placeholder(tf.int32, shape=[None], name='Actions')

        # self.model = AlexNetDuel(self.X, cfg.num_actions, cfg.train_fc)
        # self.model = C3F2(self.X, cfg.num_actions, cfg.train_fc)

        # self.predict = self.model.output
        # ind = tf.one_hot(self.actions, cfg.num_actions)
        # pred_Q = tf.reduce_sum(tf.multiply(self.model.output, ind), axis=1)
        # self.loss = huber_loss(pred_Q, self.target)
        # self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(
        #     self.loss, name="train")

        # self.sess = tf.InteractiveSession()
        # tf.global_variables_initializer().run()
        # tf.local_variables_initializer().run()
        # self.saver = tf.train.Saver()
        # self.all_vars = tf.trainable_variables()

        # self.sess.graph.finalize()

        # Load custom weights from custom_load_path if required
        if cfg.custom_load:
            print('Loading weights from: ', cfg.custom_load_path)
            self.load_network(cfg.custom_load_path)

    # def get_vars(self):
    #     return self.sess.run(self.all_vars)

    # def initialize_graphs_with_average(self, agent, agent_on_same_network):
    #     values = {}
    #     var = {}
    #     all_assign = {}
    #     for name_agent in agent_on_same_network:
    #         values[name_agent] = agent[name_agent].network_model.get_vars()
    #         var[name_agent] = agent[name_agent].network_model.all_vars
    #         all_assign[name_agent] = []
    #
    #     for i in range(len(values[name_agent])):
    #         val = []
    #         for name_agent in agent_on_same_network:
    #             val.append(values[name_agent][i])
    #         # Take mean here
    #         mean_val = np.average(val, axis=0)
    #         for name_agent in agent_on_same_network:
    #             # all_assign[name_agent].append(tf.assign(var[name_agent][i], mean_val))
    #             var[name_agent][i].load(mean_val, agent[name_agent].network_model.sess)

    def Q_val(self, xs):
        # print("Q_val", xs.shape,type(xs))

        # xs = np.array(xs)
        # xs =torch.tensor([resize(x) for x in xs ]).cuda()
        # xs = tuple([image_train(x) for x in xs ]).cuda()
        # xs = torch.tensor( map(image_train,xs)).cuda()
        # xs = get_data(xs).cuda()

        # xs = torch.tensor(np.transpose(xs,(1, 2,0,1))).cuda()

        # xs = image_train(cl).cuda()

        xs  = get_data(xs ).float().cuda()
        # print("Q_val", xs.shape,type(xs))


        with torch.no_grad():
            q_vals = outputs_source = self.source_classifier(self.source_encoder(xs))
            return q_vals.cpu().numpy()

        # target = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
        # actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        # return self.sess.run(self.predict,
        #                      feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0, self.X1: xs,
        #                                 self.target: target, self.actions: actions})

    def train_n(self, xs, ys, actions, batch_size, dropout_rate, lr, epsilon, iter):
        # xs = torch.tensor( map(image_train,xs)).cuda()
        # xs =torch.tensor(np.array([image_train(x) for x in xs ])).cuda()

        xs = get_data(xs).float().cuda()
        ys = torch.tensor(np.array(ys)).cuda()
        actions = torch.tensor(actions).long().cuda()
        # xs = torch.tensor([image_train(x) for x in xs]).cuda()
        # print("train_n xs",xs.shape)
        # print("train_n ys",ys.shape, ys)
        # print("train_n actions",actions.shape,actions)
        self.lr = lr
        self.source_encoder.train(True)
        self.source_classifier.train(True)

        self.optimizer.zero_grad()

        pred_q_vals = outputs_source = self.source_classifier(self.source_encoder(xs))

        # ind = tf.one_hot(self.actions, cfg.num_actions)
        # pred_Q = tf.reduce_sum(tf.multiply(pred_q_vals, ind), axis=1)
        #
        #

        ind  = F.one_hot(actions,num_classes=self.num_actions).float().cuda()
        # print("pred_q_vals", pred_q_vals.shape)
        # print("ind", ind.shape)
        # print("actions", actions.shape)
        # print("mul", torch.mul(pred_q_vals,ind).shape)

        pred_Q = torch.sum(torch.mul(pred_q_vals,ind), 1)
        self.loss = huber_loss2(pred_Q.double(), ys.double())

        # state_action_values = pred_q_vals.gather(1, actions)
        # self.loss = huber_loss(pred_Q, self.target)

        # loss = F.smooth_l1_loss(state_action_values, ys.unsqueeze(1))
        #
        self.loss.backward()
        self.optimizer.step()
        # pred_Q = tf.reduce_sum(tf.multiply(self.model.output, ind), axis=1)

        # self.predict = self.model.output
        # ind = tf.one_hot(self.actions, cfg.num_actions)
        # pred_Q = tf.reduce_sum(tf.multiply(self.model.output, ind), axis=1)
        # self.loss = huber_loss(pred_Q, self.target)
        #
        #
        #
        # _, loss, Q = self.sess.run([self.train, self.loss, self.predict],
        #                            feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.X1: xs,
        #                                       self.target: ys, self.actions: actions})

        pred_q_vals_numpy = pred_q_vals.clone().cpu().detach().numpy()
        loss_numpy = self.loss.clone().cpu().detach().numpy()
        meanQ = np.mean(pred_q_vals_numpy)

        maxQ = np.max(pred_q_vals_numpy)
        # Log to tensorboard
        self.log_to_tensorboard(tag='Loss', group=self.vehicle_name, value=LA.norm(loss_numpy) / batch_size, index=iter)
        self.log_to_tensorboard(tag='Epsilon', group=self.vehicle_name, value=epsilon, index=iter)
        self.log_to_tensorboard(tag='Learning Rate', group=self.vehicle_name, value=lr, index=iter)
        self.log_to_tensorboard(tag='MeanQ', group=self.vehicle_name, value=meanQ, index=iter)
        self.log_to_tensorboard(tag='MaxQ', group=self.vehicle_name, value=maxQ, index=iter)

    def action_selection(self, state):
        with torch.no_grad():
            # xs = image_train(state.cuda())
            state = get_data(state).float().cuda()
            qvals = outputs_source = self.source_classifier(self.source_encoder(state))

            qvals_numpy = qvals.clone().cpu().detach().numpy()

            # target = np.zeros(shape=[state.shape[0]], dtype=np.float32)
            # actions = np.zeros(dtype=int, shape=[state.shape[0]])
            # qvals = self.sess.run(self.predict,
            #                       feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
            #                                  self.X1: state,
            #                                  self.target: target, self.actions: actions})
            # print("qvals", qvals.shape, qvals)

            if qvals_numpy.shape[0] > 1:
                # Evaluating batch
                action = np.argmax(qvals_numpy, axis=1)
                # print("selected action", action.shape, action)

            else:
                # Evaluating one sample
                action = np.zeros(1)
                action[0] = np.argmax(qvals_numpy)
                # print("selected one action", action.shape, action)

            return action.astype(int)

    def log_to_tensorboard(self, tag, group, value, index):
        summary = tf.Summary()
        tag = group + '/' + tag
        summary.value.add(tag=tag, simple_value=value)
        self.stat_writer.add_summary(summary, index)

    def save_network(self, save_path, episode=''):
        save_path = save_path + self.vehicle_name + '/' + self.vehicle_name + '_' + str(episode)
        # self.saver.save(self.sess, save_path)
        torch.save([self.source_encoder, self.source_classifier],
                  save_path + "_model.pth.tar")
        print('Model Saved: ', save_path)

    def load_network(self, load_path):
        [self.source_encoder, self.source_classifier] = torch.load(load_path)


    # def get_weights(self):
    #     xs = np.zeros(shape=(32, 227, 227, 3))
    #     actions = np.zeros(dtype=int, shape=[xs.shape[0]])
    #     ys = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
    #     return self.sess.run(self.weights,
    #                          feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0,
    #                                     self.X1: xs,
    #                                     self.target: ys, self.actions: actions})


