#!/usr/bin/env python3
# License: © 2022 Achille-Tâm GUILCHARD All Rights Reserved
# Author: Achille-Tâm GUILCHARD

import os
import shutil
import argparse

import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial import distance

import time
import pprint
import pickle
import logging
from termcolor import colored
from operator import itemgetter
from alive_progress import alive_bar
pp = pprint.PrettyPrinter(indent=4)
# Logging stuff
LOG_FORMAT = "(%(levelname)s) %(asctime)s - %(message)s"
# create and configure logger
logging.basicConfig(level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='w')
logger = logging.getLogger()


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
    parser.add_argument('--input_dir', type=str, default="./imgs", help='Path where images to sort are stored.',
                        required=True)
    parser.add_argument('--use_tfhub_model', type=str2bool, nargs='?', const=True, default=False,
                        help="Download model from TFHUB.")
    parser.add_argument('--load_data', type=str2bool, nargs='?', const=True, default=False, help="Load image features.")
    return parser.parse_args()


def list_directories_only(rootdir):
    list_of_dir = list()
    for f in os.listdir(rootdir):
        directory = os.path.join(rootdir, f)
        if os.path.isdir(directory):
            list_of_dir.append(directory)
    return list_of_dir


def list_of_files_in_directory(path_directory):
    """Retrieve all filenames in a directory and its sub_directories"""
    res = list()
    for path, sub_dirs, files in os.walk(path_directory):
        for name in files:
            res.append(os.path.join(path, name))
    return res


def extract(image_path, image_shape, model):
    image = Image.open(image_path).convert('L').resize(image_shape)
    image = np.stack((image,) * 3, axis=-1)
    image = np.array(image) / 255.0

    embedding = model.predict(image[np.newaxis, ...])
    features = np.array(embedding)
    flattened_features = features.flatten()
    return flattened_features


def compute_similarity(img1_features, img2_features):
    metric = 'cosine'
    return distance.cdist([img1_features], [img2_features], metric)[0]


def main():
    start_time = time.time()

    args = parse_arguments()
    folder_to_sort = args.input_dir
    load_data = args.load_data
    use_tfhub_model = args.use_tfhub_model

    image_shape = None
    layer = None
    sorted_folder = "imageSorted"
    pat_to_features_binary = "./image_features.bin"
    data = {}
    
    shutil.rmtree(sorted_folder, ignore_errors=True)
    os.makedirs(sorted_folder, exist_ok=True)
    
    if use_tfhub_model:
        print(colored("Downloading model from Tensorflow hub...", 'red'))
        image_shape = (512, 512)
        model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2"
        layer = hub.KerasLayer(model_url)
    else:
        print(colored("Loading model from disk...", 'red'))
        image_shape = (224, 224)
        loaded_model = tf.keras.models.load_model('saved_model/my_model_keras')
        # Check its architecture
        loaded_model.summary()
        layer = loaded_model.get_layer(name="keras_layer")
    model = tf.keras.Sequential([layer])
    print(colored("...Done!", 'red'))
    
    if load_data:
        print(colored("Loading image features from disk...", 'red'))
        file = open(pat_to_features_binary, 'rb')
        data = pickle.load(file)
        file.close()
    else:
        print(colored("Extracting image features...", 'red'))
        list_of_images = list_of_files_in_directory(folder_to_sort)
        list_of_images.sort()
        with alive_bar(len(list_of_images), force_tty=True) as bar:
            for image in list_of_images:
                image_features = extract(image, image_shape, model)
                data[image] = image_features
                bar()
        p = "./image_features.bin"
        with open(p, 'wb') as file:
            pickle.dump(data, file)
    
    data_with_basename_as_key = {}
    images = list()
    for d in data:
        images.append([d, data[d]])
        data_with_basename_as_key[os.path.basename(d)] = data[d]
    
    print(colored("Comparing image features and matching them to one another...", 'red'))
    with alive_bar(len(data), force_tty=True) as bar:
        image_count = 1
        for img in images:
            image = img[0]
            basename_image = os.path.basename(image)
            if image_count == 1:  # For the first image we just copy it in its folder
                subfolder = sorted_folder + "/" + str(image_count)
                shutil.rmtree(subfolder, ignore_errors=True)
                os.makedirs(subfolder, exist_ok=True)
                destination = subfolder + "/" + basename_image
                shutil.copy(image, destination)
                image_count = image_count + 1
            else:
                sub_images = list_of_files_in_directory(sorted_folder)
                tmp_list = list()
                for subImage in sub_images:
                    basename_image2 = os.path.basename(subImage)
                    similarity = compute_similarity(data_with_basename_as_key[basename_image],
                                                    data_with_basename_as_key[basename_image2])
                    tmp_list.append((similarity, subImage))
    
                tmp_list = sorted(tmp_list, key=itemgetter(0))
    
                if tmp_list[0][0] < 0.15:
                    sub_dir = os.path.dirname(tmp_list[0][1])
                    destination2 = sub_dir + "/" + basename_image
                    shutil.copy(image, destination2)
                else:
                    subfolder = sorted_folder + "/" + str(image_count)
                    shutil.rmtree(subfolder, ignore_errors=True)
                    os.makedirs(subfolder, exist_ok=True)
                    destination = subfolder + "/" + os.path.basename(image)
                    shutil.copy(image, destination)
                image_count = image_count + 1
            bar()
    
    folders = list_directories_only("imageSorted")
    
    shutil.rmtree("imageSorted/cannotFindSimilarImages", ignore_errors=True)
    os.makedirs("imageSorted/cannotFindSimilarImages", exist_ok=True)
    
    for folder in folders:
        files = list_of_files_in_directory(folder)
        if len(files) == 1:
            source = files[0]
            destination = "imageSorted/cannotFindSimilarImages" + "/" + os.path.basename(source)
            shutil.copy(source, destination)
            shutil.rmtree(folder, ignore_errors=True)
    
    elapsed_time = time.time() - start_time
    logger.info(colored("Elapsed time: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))), 'green'))


if __name__ == "__main__":
    main()
