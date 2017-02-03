# normals_HoughCNN
Deep Learning for Robust Normal Estimation in Unstructured Point Clouds

# Paper

Please acknowledge our the reference paper :

"Deep Learning for Robust Normal Estimation in Unstructured Point Clouds " by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016, Computer Graphics Forum

# Dependencies

Eigen and nanoflann (assumed to be in the source folder)

Torch

The code is NVIDIA GPU dependent.

# Pretrained networks

Pretrained networks can be found at [webpage](https://sites.google.com/view/boulch/publications/2016_cgf_sgp_deepnormals).
3 models are proposed for download, 1, 3 and 5 scales (the models of the paper).

# Usage

HoughCNN_Exec [options] -m path_to_the_torch_model -i input_file.xyz -c <number of scales>

Note: the input file must currently be at xyz format, it is possible to generate such file with Meshlab.

Note: the file predict.lua should be next to the executable.

Note: number of scales has to be consistent with the used model (there are separate models for different scales).

# Example

A file cube_100k is located in the test directory.

HoughCNN_Exec [options] -m path_to_the_torch_model -i test/cube_100k.xyz -c <scale (1,3,5)>

# Author

[Alexandre Boulch](https://sites.google.com/view/boulch)
