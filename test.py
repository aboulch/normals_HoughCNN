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

import python.lib.python.NormalEstimatorHoughCNN as Estimator
import numpy as np
from tqdm import *
import torch
from torch.autograd import Variable

K = 100
scale_number = 3
batch_size = 512
USE_CUDA = True
input_filename = "test/cube_100k.xyz"
output_filename = "out.xyz"

# create thestimator
estimator = Estimator.NormalEstimatorHoughCNN()

# load the file
estimator.loadXYZ(input_filename)

Ks = None
model = None
if scale_number == 1:
    Ks=np.array([K], dtype=np.int)
    import models.model_1s as model_1s
    model = model_1s.load_model("/media/data/research/normals_HoughCNN/model_1s_boulch_SGP2016/model.pth")
    mean = np.load("/media/data/research/normals_HoughCNN/model_1s_boulch_SGP2016/mean.npz")["arr_0"]
elif scale_number == 3:
    Ks=np.array([K,K/2,K*2], dtype=np.int)
    import models.model_3s as model_3s
    model = model_3s.load_model("/media/data/research/normals_HoughCNN/model_3s_boulch_SGP2016/model.pth")
    mean = np.load("/media/data/research/normals_HoughCNN/model_3s_boulch_SGP2016/mean.npz")["arr_0"]
elif scale_number == 5:
    Ks=np.array([K,K/4,K/2,K*2,K*4], dtype=np.int)
    import models.model_5s as model_5s
    model = model_5s.load_model("/media/data/research/normals_HoughCNN/model_5s_boulch_SGP2016/model.pth")
    mean = np.load("/media/data/research/normals_HoughCNN/model_5s_boulch_SGP2016/mean.npz")["arr_0"]

# set the neighborhood size
estimator.set_Ks(Ks)
print(estimator.get_Ks())

# initialize
estimator.initialize()

# convert model to cuda if needed
if USE_CUDA:
    model.cuda()
print(model)

model.eval()
# iterate over the batches
with torch.no_grad():
    for pt_id in tqdm(range(0,estimator.size(), batch_size)):
        bs = batch_size
        batch = estimator.get_batch(pt_id, bs) - mean[None,:,:,:]
        batch_th = torch.Tensor(batch)
        if USE_CUDA:
            batch_th = batch_th.cuda()
        estimations = model.forward(batch_th)
        estimations = estimations.cpu().data.numpy()
        estimator.set_batch(pt_id,bs,estimations.astype(np.float64))

# save the estimator
estimator.saveXYZ(output_filename)
