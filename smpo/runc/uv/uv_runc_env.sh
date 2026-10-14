#!/bin/bash
# py_runc_env.sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv init

# cpu
uv add gymnasium gym stable-baselines3 matplotlib networkx notebook numpy pandas scikit-learn simpy tqdm seaborn plotly scipy pillow torch==2.5.1 torchvision==0.20.1

# gpu
uv add torch==2.5.1 torchvision==0.20.1 --find-links https://download.pytorch.org/whl/cu121
#uv add torch==2.5.1 torchvision==0.20.1 --find-links https://download.pytorch.org/whl/cu124
#uv add torch==2.10.0 torchvision==0.25.0 --find-links https://download.pytorch.org/whl/cu130


# jupyter
uv add numpy markupsafe gymnasium gym stable-baselines3 ipykernel jupyter jupyterlab ipywidgets matplotlib networkx notebook pandas simpy tqdm seaborn scikit-learn requests plotly scipy pillow tensorboard jupyter-notebook-toc jupyterlab-spellchecker variable-inspector jupyterlab-snippets jupyterlab-hide-cells
# 非jupyter
uv add numpy markupsafe gymnasium gym stable-baselines3 matplotlib networkx pandas simpy tqdm seaborn scikit-learn requests plotly scipy pillow tensorboard

