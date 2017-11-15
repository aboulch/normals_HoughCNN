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



# Building the python library

    cd path_to_repository
    mkdir build
    cd build
    cmake ..
    make

It will build a library in the python folder of the repository.

# Pretrained networks

Pretrained networks can be downloaded for 1, 3 and 5 scales (the models of the paper):
* [1 scale](https://drive.google.com/open?id=0B6IogDVqG75WTWQ3cVZOdHBPTDA)
* [3 scales](https://drive.google.com/open?id=0B6IogDVqG75WclV4czgtVDBoNkE)
* [5 scales](https://drive.google.com/open?id=0B6IogDVqG75WMVltdDYybS1VNGM)
The models for previous versions (Lua Torch) can be downloaded here: [1 scale](https://drive.google.com/open?id=0B6IogDVqG75WOFlQNVVtc1lfNW8), [3 scales](https://drive.google.com/open?id=0B6IogDVqG75WR2Z4NlJhclIzTjA) and [5 scales](https://drive.google.com/open?id=0B6IogDVqG75WMVltdDYybS1VNGM).

### Usage

Once the library is built. You can use the **estimation_script_pretrained.py** to test the estimation.
The **cube_100k.xyz** file is located in the test directory.

**Note:** the input file must currently be at xyz format, it is possible to generate such file with Meshlab.

**Note:** number of scales has to be consistent with the used model (there are separate models for different scales).

# Training from scratch

We provide the scripts for generating a training set and training a new model.
The script **generate_training_set.py** performs theses task.
You can choose the scale number (1, 3 or 5) as in the paper.
At inference, please use the **estimation_script_trained.py** instead of **estimation_script_pretrained.py**, the only difference is that the mean is now saved in a numpy format.

*Note*: This is not the original training script from the related paper, but it should be similar. If you spot malfunctioning code or unexpected behavior, please contact the author.

# License

The code is released under GPLv3 license. For commercial utilisation please contact the authors.
The license is [here](LICENSE.md).

# Author

[Alexandre Boulch](https://sites.google.com/view/boulch)
