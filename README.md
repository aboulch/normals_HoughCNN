# normals_HoughCNN
Deep Learning for Robust Normal Estimation in Unstructured Point Clouds

# Paper

Please acknowledge our the reference paper :

"Deep Learning for Robust Normal Estimation in Unstructured Point Clouds " by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016, Computer Graphics Forum

# Dependencies

- Eigen and nanoflann (in the third_party_includes folder)
- TQDM
- Cython
- PyTorch

GPU support: NVIDIA GPU

The code has been tested on Ubuntu 18.04 with Anaconda.

# Building the python library

    cd path_to_repository/python
    python setup.py install --home "."

It will build a library in the python folder of the repository.

# Pretrained networks

Pretrained networks can be downloaded for 1, 3 and 5 scales (the models of the paper):
* [1 scale](https://drive.google.com/open?id=0B6IogDVqG75WTWQ3cVZOdHBPTDA)
* [3 scales](https://drive.google.com/open?id=0B6IogDVqG75WclV4czgtVDBoNkE)
* [5 scales](https://drive.google.com/open?id=0B6IogDVqG75WMVltdDYybS1VNGM)

The models for previous versions (Lua Torch) can be downloaded here: [1 scale](https://drive.google.com/open?id=0B6IogDVqG75WOFlQNVVtc1lfNW8), [3 scales](https://drive.google.com/open?id=0B6IogDVqG75WR2Z4NlJhclIzTjA) and [5 scales](https://drive.google.com/open?id=0B6IogDVqG75WMVltdDYybS1VNGM).

### Usage

Once the library is built. You can use the **test.py** to test the estimation.
The **cube_100k.xyz** file is located in the test directory.

**Note:** the input file must currently be at xyz format, it is possible to generate such file with Meshlab.

**Note:** number of scales has to be consistent with the used model (there are separate models for different scales).

# Training from scratch

We provide the scripts for generating a training set and training a new model.
The script **train.py** performs theses task.
You can choose the scale number (1, 3 or 5) as in the paper.

*Note*: This is not the original training script from the related paper, but it should be similar. If you spot malfunctioning code or unexpected behavior, please contact the author.

# License

The code is released under GPLv3 license. For commercial utilisation please contact the authors.
The license is [here](LICENSE.md).

# Author

[Alexandre Boulch](https://sites.google.com/view/boulch)
