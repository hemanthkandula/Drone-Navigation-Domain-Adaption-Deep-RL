import os

import numpy as np


data_dir = "../data_collection/"


for file in os.listdir(data_dir):
    file_path = data_dir+file

    # np.savez('./data_collection/iter_' + str(iter) + '.npy', xs=xs, ys=ys)
    npzfile = np.load(file_path)
    # print(npzfile)
    print(npzfile['xs'].shape)
    print(npzfile['ys'].shape)
    break
