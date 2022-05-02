#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# License: © 2022 Achille-Tâm GUILCHARD All Rights Reserved
# Author: Achille-Tâm GUILCHARD

import os
import json
import uuid
import shutil
import argparse
import itertools

import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
from scipy.spatial import distance

import time
import random
import logging
from functools import wraps
from termcolor import colored
from collections import Counter
from alive_progress import alive_bar

# Logging stuff
LOG_FORMAT = "(%(levelname)s) %(asctime)s - %(message)s"
# create and configure logger
logging.basicConfig(level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='w')
logger = logging.getLogger()

# Function profiling
PROF_DATA = {}

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.perf_counter()

        ret = fn(*args, **kwargs)

        elapsed_time = time.perf_counter() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling

def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        logger.info(
            colored('Function {:s} called {:d} times. Execution time max: {:.6f} seconds, average: {:.6f} seconds'.format(
                fname,
                data[0],
                max_time,
                avg_time), 'red'))

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    """Parse input args"""                                                                                                                                                                                                                            
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_dir', type=str, default="./imgs", help='Path where images to sort are stored.', required=True)
    parser.add_argument('--use_tfhub_model', type=str2bool, nargs='?', const=True, default=False, help="Download model from TFHUB.")
    parser.add_argument('--fast_mode', type=str2bool, nargs='?', const=True, default=False, help="Use fast mode.")
                                                                                                                                                     
    return parser.parse_args()

def list_directories_only(rootdir):
    listOfDir = list()
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            listOfDir.append(d)
    return listOfDir

def list_of_files_in_directory(path_directory):
    """Retrieve all filenames in a directory and its subdirectories"""
    res = list()
    for path, subdirs, files in os.walk(path_directory):
        for name in files:
            res.append(os.path.join(path, name))
    return res

args            = parse_arguments()
start_time      = time.time()
IMAGE_SHAPE     = None
layer           = None
USE_TFHUB_MODEL = args.use_tfhub_model
FAST_MODE       = args.fast_mode

if USE_TFHUB_MODEL:
    print(colored("Downloading model...", 'red'))
    IMAGE_SHAPE = (512, 512)
    model_url   = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2"
    layer       = hub.KerasLayer(model_url)
else:
    print(colored("Loading model from disk...", 'red'))
    IMAGE_SHAPE  = (224, 224)
    loaded_model = tf.keras.models.load_model('saved_model/my_model_keras')
    # Check its architecture
    loaded_model.summary()
    layer = loaded_model.get_layer(name="keras_layer")
model = tf.keras.Sequential([layer])
print(colored("...Done!", 'red'))

@profile
def extract(file):
    file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
    file = np.stack((file,)*3, axis=-1)
    file = np.array(file)/255.0

    embedding           = model.predict(file[np.newaxis, ...])
    features            = np.array(embedding)
    flattended_features = features.flatten()
    return flattended_features

@profile
def computeSimilarity(img1Name, img2Name):
    metric = 'cosine'
    img1   = extract(img1Name)
    img2   = extract(img2Name)
    return distance.cdist([img1], [img2], metric)[0]

# Sort the folder by similiar images
folderToSort = args.input_dir
sortedFolder = "imageSorted"

shutil.rmtree(sortedFolder, ignore_errors=True)
os.makedirs(sortedFolder, exist_ok=True)

images       = list_of_files_in_directory(folderToSort)
images.sort()
imageCount   = 1

with alive_bar(len(images), force_tty=True) as bar:
    for image in images:
        if imageCount == 1:
            subfolder = sortedFolder + "/" + str(imageCount)
            shutil.rmtree(subfolder, ignore_errors=True)
            os.makedirs(subfolder, exist_ok=True)
            source      = image
            destination = subfolder + "/" + os.path.basename(image)
            shutil.copy(image, destination)
            imageCount  = imageCount + 1
        else:
            if FAST_MODE:
                subDirectories = list_directories_only(sortedFolder)
                foundSimilar   = False
                for subDir  in subDirectories:
                    subImages = list_of_files_in_directory(subDir)
                    if len(subImages) > 0:
                        randomSubImage = random.choice(subImages)
                        similarity     = computeSimilarity(randomSubImage, image)
                        if similarity < 0.25:
                            destination2 = subDir + "/" + os.path.basename(image)
                            shutil.copy(image, destination2)
                            foundSimilar = True
                            break
            else:
                subImages    = list_of_files_in_directory(sortedFolder)
                foundSimilar = False
                for subImage in subImages:
                    similarity = computeSimilarity(subImage, image)
                    if similarity < 0.25:
                        subDir       = os.path.dirname(subImage)
                        destination2 = subDir + "/" + os.path.basename(image)
                        shutil.copy(image, destination2)
                        foundSimilar = True
                        break      

            if foundSimilar == False:
                subfolder = sortedFolder + "/" + str(imageCount)
                shutil.rmtree(subfolder, ignore_errors=True)
                os.makedirs(subfolder, exist_ok=True)
                source      = image
                destination = subfolder + "/" + os.path.basename(image)
                shutil.copy(image, destination)
            imageCount = imageCount + 1
        bar()

print_prof_data()
elapsed_time = time.time() - start_time    
logger.info(colored("Elapsed time: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))), 'green'))