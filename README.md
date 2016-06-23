# normals_HoughCNN
Deep Learning for Robust Normal Estimation in Unstructured Point Clouds

# Paper

Please acknowledge our the reference paper :

"Deep Learning for Robust Normal Estimation in Unstructured Point Clouds " by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016, Computer Graphics Forum


# Dependencies

Eigen and nanoflann (assumed to be in the source folder)

Torch and TH++ (from FbLuaLib), for installation see the FbLuaLib repository

The code is NVIDIA GPU dependent.

# Usage

HoughCNN_Exec [options] -m path_to_the_torch_model -i input_file.ply -o output_file.ply

Note: the file predict.lua should be next to the executable

# Author

[Alexandre Boulch](https://sites.google.com/site/boulchalexandre)
