 # THIS IS A LINUX-ONLY compatible Object Detection training code for producing Tensorflow Lite model (.tflite) why? [check here](https://github.com/google-research/google-research/issues/779)
 #### *** ONLY WORKS ON PYTHON VERSION 3.9 or 3.8 ***
 #### SOLUTION: use [virtualbox](https://www.oracle.com/ph/virtualization/technologies/vm/downloads/virtualbox-downloads.html) and install Ubuntu on it and do the training there instead. Check the "commands.txt" (located in this repo) for an updated guide


### This Tensorflow Lite code DOES NOT WORK ON OLD HARDWARES 
## (CPUs that DO NOT support AVX / AVX2 (Advanced Vector Extensions), FMA (Fused Multiply-Add), SSE4.1 / SSE4.2 WILL NOT WORK)

![image](https://github.com/user-attachments/assets/a369e1c0-3f69-4713-ae4a-6004e6e64a8c)

### PROOF THAT IT STILL WORKS:
![image](https://github.com/user-attachments/assets/ef01a7ce-5f42-477f-ab03-0f32ab083e3e)

---

Hello, this is my first try to make something generalised, with the Python code in this repository you can develop an Object Detection model on your custom dataset [must be annotated, you can use [labelImg](https://github.com/tzutalin/labelImg) to get Pascal VOC annotation], train it and test it. This work is based on Tensorflow, and it's library [TFLITE-MODEL-MAKER](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker), you can watch my explaination on this repository [here](https://youtu.be/Yp5kglvEIZ4).

## Content of this file

* Introduction
* Requirements
* Usage

### Introduction
---

This is a simple Object detection wraper around Tensorflow Lite Model Maker, you can read more about it [here](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker).

With the help of this repository you can train an Object Detection model and save it in `.tflite` format.

### Requirements
---

Here are some basic requirements...

> Python 3.8 or greater

> Python packages Tensorflow Lite Model Maker and Pillow

> there is a `requirements.txt` file that you can use to install the required packages

### Before you start
---

There is a `config.py` file in the repository, make sure to change that according to the requirements.

### Usage
---

There are two main part of the project...
    
1. training a model
    
2. testing on a single file

> before running any file make sure you take a look at the `config.py` file and change the variables according to your need.

> the dataset must have separate training and validation folders

#### Training

make sure you go thorugh the `config.py` file, then run the file `train_model.py`, this will train the model and evaluate the model on the validation dataset and create a model in `model` folder

#### Testing

provide the path of the test image in the `test.py` folder to the variable `INPUT_IMAGE_PATH` and then run the file.

this will create a `result` folder and inside it, two files, one `input.jpeg` and annotated `output.jpeg`

## Help

If this repository is useful to you, please consider giving a start to it.

You can contact me to build any kind of Chatbot/AI/ML work.

[My Fiverr profile](https://www.fiverr.com/rajkkapadia​)

[My Upwork profile](https://www.upwork.com/freelancers/~0176aeacfcff7f1fc2)

[My LinkedIn profile](https://www.linkedin.com/in/rajkkapadia/)

Enjoy the life, Feel the music.
Peace.
