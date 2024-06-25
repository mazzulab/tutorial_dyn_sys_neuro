#!/bin/bash

# conda create -n ssm_conda_mazzu python=3.8
conda create -n ssm_dynamax python=3.9
source activate ssm_dynamax

pip install --upgrade pip
conda install -y numpy cython scipy scikit-learn matplotlib imageio pandas seaborn ipywidgets jupyterlab multiprocess joblib networkx
pip install infomap community rastermap mat73 h5py
# conda install -c conda-forge multiprocess
jupyter nbextension enable --py widgetsnbextension

#pip install autograd

#pip install tensorflow==1.7.0
#conda install -c conda-forge jupyterlab

current_dir=$(pwd)
#conda install -c conda-forge gcc 
xcode-select --install #installs compiler g++ from Xcode
# git clone git@github.com:lindermanlab/ssm.git
git clone git@github.com:mazzulab/ssm.git
cd ssm
pip install -e .

cd "$current_dir"
# cloning a dynamax fork
git clone git@github.com:mazzulab/dynamax.git
cd dynamax
pip install -e '.[dev]'
# resinstall previous version of jax and jaxlib
# pip uninstall jax jaxlib
# pip install jax-metal
cd "$current_dir"

conda install pytorch torchvision torchaudio -c pytorch-nightly


conda deactivate
echo 'Done with setup'