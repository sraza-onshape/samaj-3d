# Samaj 3D

This is a follow-on to my other ML libraries, [`samaj`](https://github.com/UPstartDeveloper/samaj) and [`samaj-vision`](https://github.com/sraza-onshape/samaj-vision/). I originally completed them as part of CS 532 (3D Computer Vision) at Stevens Institute of Technology. The code is `src.util` package is meant to be reusable, and I hope are helpful to more researchers/engineers in the future.

## Running the Code in the Notebooks

Please see this tutorial to set up things up: [How to Add/Delete Kernels from Jupyter Notebook](https://janakiev.com/blog/jupyter-virtual-envs/).

**TL;DR**:
```bash
$ python3 -m venv env
$ source env/bin/activate
(env) $ python -m pip install -r requirementx.txt
$ python -m ipykernel install --name=env
```

And know you should be able to select `"env"` as a kernel to use when running your Jupyter server :)

## References
1. Useful equation for finding the dimensions of a feature map outputted by a convolution - inspired the formula I used to compute the `padding_distance` in `ops.py`: [Stanford CS 231](https://cs231n.github.io/convolutional-networks/)
