# otbtf keras tutorial

How to work with OTBTF and Keras / Tensorflow 2?

This tutorial is a revamp of the "Semantic Segmentation" part of 
[this book](https://www.routledge.com/Deep-Learning-for-Remote-Sensing-Images-with-Open-Source-Software/Cresson/p/book/9780367518981), 
using Keras.
Everything is coded in python.

Materials are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

# Introduction 

This is a very simple tutorial showing how to create training/validation/test
datasets from geospatial images, training a model, evaluating the model, and 
applying it to map buildings from a Spot-7 product.
A slight variation from the original book is that the proposed model shows how 
to consume multiple sources of different resolution and caracteristics natively:
the XS and Pan images of the Spot-7 product are directly fed in the model,
without any pre-processing.

# Get started

## Docker image

Pull the latest **otbtf** image on dockerhub. Here is an example with version
4.1.0, CPU flavored:

```
docker pull mdl4eo/otbtf:4.1.0-cpu
```

Then start a container and follow the tutorial inside it.
Here we mount some local volume in `/data` as storage for the data.

```
docker run -ti -v /some/local/pth:/data mdl4eo/otbtf:4.1.0-cpu
```

## pyotb

Install [pyotb](https://github.com/orfeotoolbox/pyotb) >= 2.0.0 inside the 
container.
Since pyotb 2.0.0 is still in beta, do the following to install it:

```
pip install pyotb==2.0.0-dev6
```

## Data

We use the same data as in the "Semantic Segmentation" part of 
[this book](https://www.routledge.com/Deep-Learning-for-Remote-Sensing-Images-with-Open-Source-Software/Cresson/p/book/9780367518981).

- One Spot-7 product, that can be freely downloaded from the 
[Airbus Defense and Space website](https://www.intelligence-airbusds.com/en/9317-sample-imagery-detail?product=35862)
- One label image of buildings, with the same extent and spatial resolution as 
the panchromatic channel of the Spot-7 product, that can be downloaded 
[here](https://github.com/remicres/otbtf_tutorials_resources/blob/master/02_semantic_segmentation/amsterdam_dataset/terrain_truth/amsterdam_labelimage.tif)

For all steps, we suggest to use the following directories:

- Put `amsterdam_labelimage.tif` in the `/data/terrain_truth` directory,
- Decompress the Spot-7 product archive into `/data/spot`,
- We will use `/data/output` to generate all output files.

# Step 1: sampling

In this section, we extract patches into 3 images:

- the panchromatic channel of the Spot-7 product, 64x64 pixels, monoband, uint8
- the multispectral image of the Spot-7 product: 16x16 pixels, multiband, uint8
- the label image of buildings: 64x64 pixels, monoband, uint8

Please note that Spot-7 images are encoded in uint8 because they are free products 
but in the real world they are in uint16.
We extract the 3 sources simultaneously using the `PatchesExtraction` application.
The location of the patches centroids is trivialy done using the `PatchesSelection` 
application. Note that in practice you probably want to go with your own 
approach, however this simple case is good enough for our tutorial!

Run the following:

```
python sampling.py
```

After the execution, the following new files should have been created:

- For the training dataset:
  - `train_p_patches.tif`
  - `train_xs_patches.tif`
  - `train_labels_patches.tif`
- For the validation dataset:
  - `valid_p_patches.tif`
  - `valid_xs_patches.tif`
  - `valid_labels_patches.tif`
- For the testing dataset:
  - `test_p_patches.tif`
  - `test_xs_patches.tif`
  - `test_labels_patches.tif`

If you want to deploy training at production scale, you can use 
`otbtf.TFRecords` to convert the `otbtf.DatasetFromPatchesImage` into TFRecords 
then to feed the training process. However that's not covered in this tutorial 
(more info in the [otbtf documentation](https://otbtf.readthedocs.io/)).

# Step 2: training

We train a very lightweight U-Net from the previously extracted patches.
The training is done using Keras in python.
Instances of `otbtf.DatasetFromPatchesImage` are used to generate tensorflow 
datasets to train, valid, and test our network.

Run the following:

```
python training.py --model_dir /data/output/savedmodel
```

In the end the best network (wrt. validation dataset) is exported as a 
SavedModel in `/data/output/savedmodel`.
The model is evaluated against the test dataset and precision, recall metrics 
are displayed.

# Step 3: inference

Run the following to generate the nice buildings map:

```
python inference.py
```
