import numpy as np
import tensorflow as tf

# import torch

def huber_loss(X, Y):
    err = X - Y
    loss = tf.where(tf.abs(err) < 1.0, 0.5 * tf.square(err), tf.abs(err) - 0.5)
    loss = tf.reduce_sum(loss)

    return loss

# def huber_loss2(X, Y):
#     err = X - Y
#     loss = torch.where(torch.abs(err) < 1.0, 0.5 * err **2, torch.abs(err) - 0.5)
#     loss = torch.sum(loss)
#
#     return loss


def mse_loss(X, Y):
    err = X - Y
    return tf.reduce_sum(tf.square(err))
