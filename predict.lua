-- License Information
--
--  Copyright (C) ONERA, The French Aerospace Lab
--  Author: Alexandre BOULCH
--
--  Permission is hereby granted, free of charge, to any person obtaining a copy of this 
--  software and associated documentation files (the "Software"), to deal in the Software 
--  without restriction, including without limitation the rights to use, copy, modify, merge,
--  publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons 
--  to whom the Software is furnished to do so, subject to the following conditions:
--  
--  The above copyright notice and this permission notice shall be included in all copies or
--  substantial portions of the Software.
--
--  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
--  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
--  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
--  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
--  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
--  OR OTHER DEALINGS IN THE SOFTWARE.
--
--
--  Note that this library relies on external libraries subject to their own license.
--  To use this software, you are subject to the dependencies license, these licenses 
--  applies to the dependency ONLY  and NOT this code.
--  Please refer below to the web sites for license informations:
--       PCL, BOOST,NANOFLANN, EIGEN, LUA TORCH
-- 
-- When using the software please aknowledge the  corresponding publication:
-- "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
-- by Alexandre Boulch and Renaud Marlet
-- Symposium of Geometry Processing 2016, Computer Graphics Forum
--

print("HoughCNN normal estimation")
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image'


print("TORCH --> START")
function load_model(path)
    print("TORCH --> load model")
    model = torch.load(paths.concat(path,"net.t7"))
    mean = torch.load(paths.concat(path,"mean.t7")):float()
    print(model)
    print("TORCH --> Done")
    model:evaluate()
end

function estimate(t)
    print("TORCH --> estimate")
    local outputs = torch.FloatTensor(t:size(1),2)
    local batchSize = 256
    for i=1, t:size(1) do
        t[i] = t[i] - mean
    end
    for i=1,t:size(1),batchSize do
        local s = math.min(batchSize,t:size(1)-i+1)
        outputs:narrow(1,i,s):copy(model:forward(t:narrow(1,i,s):cuda()))
    end
    print("TORCH --> Done")
    return outputs;
end

function estimate_batch(t)
  -- create the output
  local outputs = torch.FloatTensor(t:size(1),2)
  
  -- substract mean
  for i=1, t:size(1) do
      t[i] = t[i] - mean
  end

  -- forward
  outputs:copy(model:forward(t:cuda()):float())

  return outputs;
end

print("HoughCNN normal estimation : END")
