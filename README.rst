==============
DeePyMoD_torch
==============

DeepMod is a deep learning based model discovery algorithm which seeks the partial differential equation underlying a spatio-temporal data set. DeepMoD employs sparse regression on a library of basis functions and their corresponding spatial derivatives. This code is based on the paper: [arXiv:1904.09406](http://arxiv.org/abs/1904.09406) 

A feed-forward neural network approximates the data set and automatic differentiation is used to construct this function library and perform regression within the neural network. This construction makes it extremely robust to noise and applicable to small data sets and, contrary to other deep learning methods, does not require a training set and is impervious to overfitting. We illustrate this approach on several physical problems, such as the Burgers', Korteweg-de Vries, advection-diffusion and Keller-Segel equations. 


How to install 
==============
We currently provide two ways to use our software, either in a docker container or as a normal package. If you want to use it as a package, simply clone the repo and run:

```python setup.py install```

A GPU-ready Docker image can also be used. Clone the repo, go into the config folder and run:

```./start_notebook.sh```

This pulls our lab's standard docker image from dockerhub, mounts the project directory inside the container and starts a jupyterlab server which can be accessed through localhost:8888. You can stop the container by running the stop_notebook script.  This will stop the container; next time you run start_notebook.sh it will look if any containers from that project exist and restart them instead of building a new one, so your changes inside the container are maintained.

Dependencies
==============
We don't provide a requirements.txt, but we've tested it on the following versions

* Python - 3.6.*
* Torch - 1.1.*
* Numpy - 1.16.2
* Tensorboard - 1.14 

The tensorboard really must be 1.14 as it's the first version which doesnt require a full tensorflow install.
This project has been set up using PyScaffold 3.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
