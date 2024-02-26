# README

## Recreating the Conda Environment
There is an `environment.yml` file in the zipped folder. You can use that easily to recreate the virtual envirnoment used in this study.

## Changes to the .env file
There is a file called .env which holds all the paths to important folders.

Go and check it out, you probably only need to change the `ROLLING_DATA_PATH`.

Also make sure other folders set in .env exists to avoid error.


## src/
The `src` folder has most of the utility functions in it. 

`datasets/` - This folder includes the image data preparation functions.

`models/ ` - This folder includes the base model in this paper and written in pytorch.

`gda.py` - Gaussian Discriminanct analysis for Mahalanobis distance based classification method.

`tree.py`- it contains most of the logic about constructing hierarchy trees from tree descriptions and getting soft or hard label transforms as layers to be used in PyTorch.

The tree description for the rolling dataset is `rolling_hierarchy_description.json` in the root folder, which should be easy to read and understand. It's just a key-value pair of coarse classes and fine-grained classes.


## hypothesis1_log.py
This is a file that runs Netflix's Metaflow in the background. The tutorial is [here](https://metaflow.org).

This file is responsible for running bunch of experiments and saving the models for later use in the testing stage.

In this file, both train and test steps are contained.

When models are trained, they should be under the folder specified by `CHECKPOINTS_DIR` in the `.env` file.




## hypothesis1_derivatives_log.py
This file precomputes and caches logit outputs of test samples on trained models for reuse and they are stored under the folder specified by 'DERIVATIVES_DIR'. Because logits are reused over and over again when calculating AUROCs, it would be too costly to run experiments through GPU over and over again. 

This will need all the trained model parameters contained in 'CHECKPOINTS_DIR', which is currently not included here since bunch of model checkpoints were stored and they are too large. 



## hypothesis1results_log.py
This is the file that produces most of the figures used in the study. It uses [Streamlit](https://docs.streamlit.io/en/stable/). To use streamlit in visualization, you need to 'streamlit run hypothesis1results_log.py. This is a personal choice. You can use the same code and transform it into a Jupyter Notebook or other platforms.


## `run_hypothesis1.py`
This file shows how to run model trainings in parallel using [GNU Parallel](https://www.gnu.org/software/parallel/). Make sure you know what GNU Parallel is doing before you proceed with this. Once you master GPU Parallel, it makes life quite easy but then you should also keep an eye on your GPU usage so that you don't block others from using it.




