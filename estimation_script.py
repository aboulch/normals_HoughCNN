import python.NormalEstimatorHoughCNN as Estimator
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
    import python.model_1s as model_1s
    model = model_1s.load_model("path_to_model_1_scale.pth")
elif scale_number == 3:
    Ks=np.array([K/2,K,K*2], dtype=np.int)
    import python.model_3s as model_3s
    model = model_3s.load_model("path_to_model_3_scales.pth")
elif scale_number == 5:
    Ks=np.array([K/4,K/2,K,K*2,K*4], dtype=np.int)
    import python.model_5s as model_5s
    model = model_5s.load_model("path_to_model_5_scales.pth")

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
for pt_id in tqdm(range(0,estimator.size(), batch_size)):
    bs = batch_size
    batch = estimator.get_batch(pt_id, bs)
    batch_th = Variable(torch.Tensor(batch), volatile=True)
    if USE_CUDA:
        batch_th = batch_th.cuda()
    estimations = model.forward(batch_th)
    estimations = estimations.cpu().data.numpy()
    estimations = np.zeros((bs,2))
    estimator.set_batch(pt_id,bs,estimations)

# save the estimator
estimator.saveXYZ(output_filename)
