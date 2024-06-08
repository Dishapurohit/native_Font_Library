# native_Font_Library
This is small project of Font Libraries for native languages ( Hindi , Malayalam, Tamil etc). By using GAN model to calculate AL/ML model be trained to generate entire library from a (chosen) few characters?

C:\Users\Admin>python --version
Python 3.11.5

Libraries:
inltk = natural language toolkit for indic (Indian native language)
import pandas as pd     # for data collection (.csv file format to load data)
import numpy as np      # for data analysis
import matplotlib.pyplot as plt   # for data visualization
import tensorflow as tf   # for machine learning operations


# Monkey patch for collections.abc import issue
import collections.abc   # I found error like in 'setup language' so that I use this library to
# iterate 'collection' file
collections.Iterable = collections.abc.Iterable

[2] Data Preprocessing
use 'setup' for language selection. it is a methos in NLP or inltk

[3] Predict Next Character = Use inltk's predict_next_words to predict the next character in a sequence.

[4]Generate Sequence = Function to generate a sequence of characters based on the predicted next character.

[5] Model Selection = Use a GAN model to learn the style and generate new characters. 
2024-06-08 18:46:34.045774: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags. 
I found this warning/ error due my system settings.

[6] Training = Train the model with the collected data.

[7] Inference = Generate the entire font library from a few input characters.
