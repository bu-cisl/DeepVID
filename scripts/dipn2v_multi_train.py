import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'predict')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'source')))

import glob
import pathlib
import datetime
import logging
import multiprocessing
from tqdm import tqdm

import tensorflow as tf
from source.utils import JsonSaver, ClassLoader
from source.generator_collection import CollectorGenerator

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

# run time for keeping record
run_uid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# initialize meta-parameter objects
training_param = {}
network_param = {}
generator_test_param = {}

steps_per_epoch = 360

train_paths = glob.glob(os.path.join(
    "..", "datasets", "training",
    "*.tif"
))
train_paths = sorted(train_paths)
print("# of Images:", len(train_paths))

# Those are parameters used for the Validation test generator. Here the test is done on the beginning of the data but
# this can be a separate file
generator_test_param["type"] = "generator"  # type of collection
generator_test_param["name"] = "DIPN2VGenerator"  # Name of object in the collection
generator_test_param["pre_post_frame"] = 3  # Number of frame provided before and after the predicted frame
generator_test_param["pre_post_omission"] = 0
generator_test_param["train_path"] = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "..",
    "datasets",
    "2b.tif",
)
generator_test_param["randomize"] = False
generator_test_param["batch_size"] = 4
generator_test_param[
    "steps_per_epoch"
] = -1  # No step necessary for testing as epochs are not relevant. -1 deactivate it.
generator_test_param["blind_pixel_ratio"] = 0.1
generator_test_param["blind_pixel_method"] = "replace"
generator_test_param["cell_mask_path"] = None

# Those are parameters used for the main data generator
generator_param_list = []
for indiv_path in train_paths:
    generator_param = {}
    generator_param["type"] = generator_test_param["type"]
    generator_param["name"] = generator_test_param["name"]
    generator_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
    generator_param["pre_post_omission"] = generator_test_param["pre_post_omission"]
    generator_param["train_path"] = indiv_path

    generator_param["randomize"] = True
    generator_param["batch_size"] = 4
    generator_param["steps_per_epoch"] = steps_per_epoch
    generator_param["blind_pixel_ratio"] = generator_test_param["blind_pixel_ratio"]
    generator_param["blind_pixel_method"] = generator_test_param["blind_pixel_method"]
    generator_param["cell_mask_path"] = generator_test_param["cell_mask_path"]

    generator_param_list.append(generator_param)

# Those are parameters used for the network topology
network_param["type"] = "network"
network_param["name"] = "fullyconv_mask"  # Name of network topology in the collection

# Those are parameters used for the training process
training_param["type"] = "trainer"
training_param["name"] = "core_trainer"
training_param["run_uid"] = run_uid
training_param["batch_size"] = generator_test_param["batch_size"]
training_param["steps_per_epoch"] = steps_per_epoch
training_param[
    "period_save"
] = 100  # network model is potentially saved during training between a regular nb epochs
training_param["save_image_period"] = 50
# training_param["nb_gpus"] = 0
training_param["apply_learning_decay"] = 0
training_param[
    "nb_times_through_data"
] = 1  # if you want to cycle through the entire data. Two many iterations will cause noise overfitting
training_param["learning_rate"] = 5e-6
# training_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
training_param["loss"] = "mse_with_mask"
training_param[
    "nb_workers"
] = multiprocessing.cpu_count()  # this is to enable multiple threads for data generator loading. Useful when this is slower than training
print("num_workers:", training_param["nb_workers"])

training_param["model_string"] = "_".join((
    str(training_param["run_uid"]),
    generator_test_param["name"],
    generator_test_param["blind_pixel_method"],
    str(generator_test_param["blind_pixel_ratio"]),
    training_param["loss"],
    "pre_post_frame",
    str(generator_test_param["pre_post_frame"]),
    "multi",
))

# Where do you store ongoing training progress
jobdir = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "..",
    "results",
    training_param["model_string"],
)
training_param["output_dir"] = jobdir

try:
    os.mkdir(jobdir)
except:
    print("folder already exists")

# Here we create all json files that are fed to the training. This is used for recording purposes as well as input to the
# training process
path_training = os.path.join(jobdir, "training.json")
json_obj = JsonSaver(training_param)
json_obj.save_json(path_training)

list_train_generator = []
print("Building generators")
for local_index, indiv_generator in tqdm(iterable=enumerate(generator_param_list), total=len(generator_param_list)):
    path_generator = os.path.join(jobdir, "generator" + str(local_index) + ".json")
    json_obj = JsonSaver(indiv_generator)
    json_obj.save_json(path_generator)
    generator_obj = ClassLoader(path_generator)
    train_generator = generator_obj.find_and_build()(path_generator)
    list_train_generator.append(train_generator)

path_test_generator = os.path.join(jobdir, "test_generator.json")
json_obj = JsonSaver(generator_test_param)
json_obj.save_json(path_test_generator)

path_network = os.path.join(jobdir, "network.json")
json_obj = JsonSaver(network_param)
json_obj.save_json(path_network)

# We find the generator obj in the collection using the json file
generator_test_obj = ClassLoader(path_test_generator)

# We find the network obj in the collection using the json file
network_obj = ClassLoader(path_network)

# We find the training obj in the collection using the json file
trainer_obj = ClassLoader(path_training)

# We build the generators object. This will, among other things, calculate normalizing parameters.
test_generator = generator_test_obj.find_and_build()(path_test_generator)

# We build the network object. This will, among other things, calculate normalizing parameters.
network_callback = network_obj.find_and_build()(path_network)

global_train_generator = CollectorGenerator(list_train_generator)

# We build the training object.
training_class = trainer_obj.find_and_build()(
    global_train_generator, test_generator, network_callback, path_training
)

# Start training. This can take very long time.
training_class.run()

# Finalize and save output of the training.
training_class.finalize()
