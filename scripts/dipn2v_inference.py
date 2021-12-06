import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'predict')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'source')))

import pathlib
import logging
import multiprocessing

import tensorflow as tf
from source.utils import JsonSaver, ClassLoader

logging.getLogger("tensorflow").setLevel(logging.ERROR)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

MODEL_NAME = "20211109-095033_DIPN2VGenerator_replace_0.1_mse_with_mask_pre_post_frame_3_multi"
INPUT_FNAME = "2b.tif"
INPUT_FNAME_NO_EXT = os.path.splitext(INPUT_FNAME)[0]  # "2b"

generator_param = {}
inferrence_param = {}

# We are reusing the data generator for training here.
generator_param["type"] = "generator"
generator_param["name"] = "DIPN2VGenerator"
generator_param["pre_post_frame"] = 3
generator_param["pre_post_omission"] = 0
generator_param[
    "steps_per_epoch"
] = -1  # No steps necessary for inference as epochs are not relevant. -1 deactivate it.

generator_param["train_path"] = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "..",
    "datasets",
    INPUT_FNAME,
)

generator_param["batch_size"] = 1
generator_param[
    "randomize"
] = False  # This is important to keep the order and avoid the randomization used during training
generator_param["blind_pixel_ratio"] = 0.1
generator_param["blind_pixel_method"] = "replace"
generator_param["cell_mask_path"] = None

inferrence_param["type"] = "inferrence"
inferrence_param["name"] = "core_inferrence"
inferrence_param["rescale"] = True
inferrence_param["nb_workers"] = multiprocessing.cpu_count()
print("num_workers:", inferrence_param["nb_workers"])

# Replace this path to where you stored your model
base_path = os.path.join(
    "..",
    "results",
    MODEL_NAME,
)

inferrence_param["model_path"] = os.path.join(
    base_path,
    MODEL_NAME + "_model.h5"
)

# Replace this path to where you want to store your output file
inferrence_param["output_file"] = os.path.join(
    base_path,
    "_".join((MODEL_NAME, "result", INPUT_FNAME_NO_EXT + ".tiff")),
)

jobdir = os.path.join(
    base_path,
    "_".join(("inference", INPUT_FNAME_NO_EXT)),  # suggest to append the name of image in the end (i.e. 2b here)
)

try:
    os.mkdir(jobdir)
except:
    print("folder already exists")

path_generator = os.path.join(jobdir, "generator.json")
json_obj = JsonSaver(generator_param)
json_obj.save_json(path_generator)

path_network = os.path.join(base_path, "network.json")

path_infer = os.path.join(jobdir, "inferrence.json")
json_obj = JsonSaver(inferrence_param)
json_obj.save_json(path_infer)

generator_obj = ClassLoader(path_generator)
data_generator = generator_obj.find_and_build()(path_generator)

network_obj = ClassLoader(path_network)
network_callback = network_obj.find_and_build()(path_network)

inferrence_obj = ClassLoader(path_infer)
inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator, network_callback)

# Except this to be slow on a laptop without GPU. Inference needs parallelization to be effective.
inferrence_class.run()
