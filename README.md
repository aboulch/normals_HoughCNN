# normals_HoughCNN
Deep Learning for Robust Normal Estimation in Unstructured Point Clouds

# Paper

Please acknowledge our the reference paper :

"Deep Learning for Robust Normal Estimation in Unstructured Point Clouds " by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016, Computer Graphics Forum

# Dependencies

- Eigen and nanoflann (assumed to be in the include folder)
- CMake
- Cython
- PyTorch

GPU support: NVIDIA GPU

# Pretrained networks

Pretrained networks can be found at [webpage](https://sites.google.com/view/boulch/publications/2016_cgf_sgp_deepnormals).
3 models are proposed for download, 1, 3 and 5 scales (the models of the paper).

# Building the python library

    cd path_to_repository
    mkdir build
    cd build
    cmake ..
    make

It will build a library in the python folder of the repository.

# Usage

Once the library is built. You can use the **estimation_script.py** to test the estimation.
The **cube_100k.xyz** file is located in the test directory.

**Note:** the input file must currently be at xyz format, it is possible to generate such file with Meshlab.

**Note:** number of scales has to be consistent with the used model (there are separate models for different scales).

# License

The code is released under GPLv3 license. For commercial utilisation please contact the authors.
The license is [here](LICENSE.md).

# Previous versions

# Author

[Alexandre Boulch](https://sites.google.com/view/boulch)
