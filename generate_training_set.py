# Deep Learning for Robust Normal Estimation in Unstructured Point Clouds
# Copyright (c) 2016 Alexande Boulch and Renaud Marlet
#
# This program is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street,
# Fifth Floor, Boston, MA 02110-1301  USA
#
# PLEASE ACKNOWLEDGE THE ORIGINAL AUTHORS AND PUBLICATION:
# "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
# by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016,
# Computer Graphics Forum

import python.NormalEstimatorHoughCNN as Estimator
import numpy as np
from tqdm import *
import torch
from torch.autograd import Variable
from torch.utils.serialization import load_lua
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import pickle
import os

torch.backends.cudnn.benchmark = True

K = 100
scale_number = 1
batch_size = 256
USE_CUDA = True

create_dataset = True
train = True
dataset_size = 100000
dataset_directory = "diretory_for_saving_dataset"
result_directory = "directory_for_saving_trained_model"

drop_learning_rate = 0.5
learning_rate = 0.1
epoch_max = 40
decrease_step = 4

# create thestimator
estimator = Estimator.NormalEstimatorHoughCNN()

Ks = None
if scale_number == 1:
    Ks=np.array([K], dtype=np.int)
elif scale_number == 3:
    Ks=np.array([K,K/2,K*2], dtype=np.int)
elif scale_number == 5:
    Ks=np.array([K,K/4,K/2,K*2,K*4], dtype=np.int)

estimator.set_Ks(Ks)

if create_dataset:
    print("creating dataset")
    count = 0
    dataset = np.zeros((dataset_size, scale_number, 33,33))
    targets = np.zeros((dataset_size, 2))
    for i in tqdm(range(0,dataset_size, batch_size), ncols=80):
        nbr, batch, batch_targets = estimator.generate_training_accum_random_corner(batch_size)
        if count+nbr > dataset_size:
            nbr = dataset_size - count
        dataset[count:count+nbr] = batch[0:nbr]
        targets[count:count+nbr] = batch_targets[0:nbr]
        count += nbr
        if(count >= dataset_size):
            break
    # save the dataset
    mean = dataset.mean(axis=0)
    print(mean.shape)
    dataset = {"input":dataset, "targets":targets, "mean":mean}
    print("  saving")
    pickle.dump( dataset, open( os.path.join(dataset_directory, "dataset.p"), "wb" ) )
    print("-->done")

if train:
    # create the model
    print("creating the model")
    if scale_number == 1:
        import python.model_1s as model_def
    elif scale_number == 3:
        import python.model_3s as model_def
    elif scale_number == 5:
        import python.model_5s as model_def
    net = model_def.create_model()

    # load the dataset
    print("loading the model")
    dataset = pickle.load( open( os.path.join(dataset_directory, "dataset.p"), "rb" ) )

    print("Creating optimizer")
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9)

    if USE_CUDA:
        net.cuda()
        criterion.cuda()

    dataset["input"] -= dataset["mean"][None,:,:,:]

    input_data = torch.from_numpy(dataset["input"]).float()
    target_data = torch.from_numpy(dataset["targets"]).float()
    ds = torch.utils.data.TensorDataset(input_data, target_data)
    ds_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    print("Training")

    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    np.savez(os.path.join(result_directory, "mean"),dataset["mean"])
    f = open(os.path.join(result_directory, "logs.txt"), "w")

    for epoch in range(epoch_max):

        if(epoch%decrease_step==0 and epoch>0):
            learning_rate *= drop_learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        total_loss = 0
        count = 0

        t = tqdm(ds_loader, ncols=80)
        for data in t:

            # set optimizer gradients to zero
            optimizer.zero_grad()

            # create variables
            batch = Variable(data[0])
            batch_target = Variable(data[1])
            if(USE_CUDA):
                batch = batch.cuda()
                batch_target = batch_target.cuda()

            # forward backward
            output = net.forward(batch)
            error = criterion(output, batch_target)
            error.backward()
            optimizer.step()

            b_loss = error.cpu().data.numpy()[0]
            count += batch.size(0)
            total_loss += b_loss

            t.set_postfix(Bloss= b_loss/batch.size(0), loss= total_loss/count)

        f.write(str(epoch)+" ")
        f.write(str(learning_rate)+" ")
        f.write(str(total_loss))
        f.write("\n")
        f.flush()

        # save the model
        torch.save(net.state_dict(), os.path.join(result_directory, "state_dict.pth"))

    f.close()
