# Homework 3

#### Name: Syed Zain Raza
#### CWID: 20011917

## What Was Done

As outlined in the homework description, I utilized a volumetric stereo approach to carry out a voxel-based reconstruction of a 3D scene into a PLY file, given 8 calibrated images and their silhouettes. I used a grid of dimensions 50x50x50 due to memory constraints. The code is mostly implemented in NumPy, and relies heavily on vectorization. There is a slight bug in step 7 which causes the output model to be all black (no color is found), which I have not yet root-caused. For visualizing the 3D model, I used Open3D.

## How to Reproduce Results

1. Make a virtual environment with Python. I've tested the following with Python 3.11:
```bash
$ python3 -m venv env  # environment setup - virtualenv
$ source env/bin/activate
(env) $ python -m pip install -r requirements.txt
```

1. The code is written in a Jupyter notebook. Please [add the virtual you just created as a Jupyter kernel](https://janakiev.com/blog/jupyter-virtual-envs/), so you can use it to run the code.

```bash
$ python -m ipykernel install --name=env
```

1. Finally, you can navigate to `experiments_hw3.ipynb` to run the code.


## Source Code and PLY Files

1. `src/experiments_hw3.ipynb` - this is my report. I provide code and docstrings thereof where appropiate.


1. `src/util` - this directory has a collection of my own helper functions, which I commonly use to complete assignments.

1. `Syed_Raza_PtCloudFalseColors.ply` - the final PLY I created (with a 50x50x50 voxel grid) in Step 6.

1. `Syed_Raza_PtCloudTrueColors.ply` - the final PLY I created (with a 50x50x50 voxel grid) in Step 8.

## Screenshots

### False Colors
![](./hw3_screenshots/falseColors/Screenshot%202024-04-07%20at%2010.43.35 PM.png)

![](./hw3_screenshots/falseColors/Screenshot%202024-04-07%20at%2010.43.40 PM.png)

![](./hw3_screenshots/falseColors/Screenshot%202024-04-07%20at%2010.43.46 PM.png)

### (Attempted) True Colors

![](./hw3_screenshots/trueColors/Screenshot%202024-04-07%20at%2010.44.39 PM.png)

![](./hw3_screenshots/trueColors/Screenshot%202024-04-07%20at%2010.44.42 PM.png)

![](./hw3_screenshots/trueColors/Screenshot%202024-04-07%20at%2010.44.48 PM.png)
