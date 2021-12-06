import os

import numpy as np
import matplotlib.pylab as plt

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger

import loss_collection as lc
from callbacks import SaveImageCallback
from source.utils import JsonLoader

from packaging import version


class core_trainer:
    # This is the generic trainer class
    # auto_compile can be set to False when doing an
    # hyperparameter search to allow modification of the model
    def __init__(
            self,
            generator_obj,
            test_generator_obj,
            network_obj,
            trainer_json_path,
            auto_compile=True,
    ):

        self.network_obj = network_obj
        self.generator_obj = generator_obj
        self.test_generator_obj = test_generator_obj
        self.trainer_json_path = trainer_json_path

        json_obj = JsonLoader(trainer_json_path)

        # the following line is to be backward compatible in case
        # new parameter logics are added.

        self.json_data = json_obj.json_data
        self.output_dir = self.json_data["output_dir"]
        self.run_uid = self.json_data["run_uid"]
        self.model_string = self.json_data["model_string"]
        self.batch_size = self.json_data["batch_size"]
        self.steps_per_epoch = self.json_data["steps_per_epoch"]
        self.loss_type = self.json_data["loss"]
        # self.nb_gpus = self.json_data["nb_gpus"]
        self.period_save = self.json_data["period_save"]
        self.learning_rate = self.json_data["learning_rate"]

        if 'checkpoints_dir' in self.json_data.keys():
            self.checkpoints_dir = self.json_data["checkpoints_dir"]
        else:
            self.checkpoints_dir = self.output_dir

        if "use_multiprocessing" in self.json_data.keys():
            self.use_multiprocessing = self.json_data["use_multiprocessing"]
        else:
            self.use_multiprocessing = False

        if "caching_validation" in self.json_data.keys():
            self.caching_validation = self.json_data["caching_validation"]
        else:
            self.caching_validation = True

        self.output_model_file_path = os.path.join(
            self.output_dir,
            self.model_string + "_model.h5"
        )

        if "nb_workers" in self.json_data.keys():
            self.workers = self.json_data["nb_workers"]
        else:
            self.workers = 8

        # These parameters are related to setting up the
        # behavior of learning rates
        self.apply_learning_decay = self.json_data["apply_learning_decay"]

        if self.apply_learning_decay == 1:
            self.initial_learning_rate = self.json_data["initial_learning_rate"]
            self.epochs_drop = self.json_data["epochs_drop"]

        self.nb_times_through_data = self.json_data["nb_times_through_data"]

        # Generator has to be initialized first to provide
        # input size of network
        self.initialize_generator()

        if auto_compile:
            self.initialize_network()

        self.initialize_callbacks()

        self.initialize_loss()

        self.initialize_optimizer()

        if auto_compile:
            self.compile()

    def compile(self):
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer
        )  # , metrics=['mae'])

    def initialize_optimizer(self):
        self.optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    def initialize_loss(self):
        self.loss = lc.loss_selector(self.loss_type)

    def initialize_callbacks(self):
        callbacks_list = []

        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            self.model_string +
            "-{epoch:04d}-{val_loss:.4f}.h5",
        )
        checkpoint_callback = ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
            period=self.period_save,
        )
        callbacks_list.append(checkpoint_callback)

        # Add on epoch_end callback

        epo_end = OnEpochEnd([self.generator_obj.on_epoch_end])

        tensorboard_callback = TensorBoard(self.output_dir, profile_batch=0)
        callbacks_list.append(tensorboard_callback)

        csv_logger_callback = CSVLogger(os.path.join(self.output_dir, 'training_log.csv'))
        callbacks_list.append(csv_logger_callback)

        save_image_callback = SaveImageCallback(self.test_generator_obj, self.trainer_json_path)
        callbacks_list.append(save_image_callback)

        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1,
                                               patience=200, min_delta=1e-3, min_lr=1e-6)
        callbacks_list.append(reduce_lr_callback)

        if version.parse(tensorflow.__version__) <= version.parse("2.1.0"):
            callbacks_list.append(epo_end)

        self.callbacks_list = callbacks_list

    def initialize_generator(self):
        # If feeeding a stepped generator,
        # we need to calculate the number of epochs accordingly
        if self.steps_per_epoch > 0:
            self.epochs = self.nb_times_through_data * int(
                np.floor(len(self.generator_obj) / self.steps_per_epoch)
            )
        else:
            self.epochs = self.nb_times_through_data * \
                          int(len(self.generator_obj))

    def initialize_network(self):
        local_size = self.generator_obj.get_input_size()

        input_img = Input(shape=local_size)
        self.model = Model(input_img, self.network_obj(input_img))
        self.model.summary()

    def cache_validation(self):
        # This is used to remove IO duplication,
        # leverage memory for validation and
        # avoid deadlocks that happens when
        # using keras.utils.Sequence as validation datasets
        num_sample_val = 12

        input_example = self.test_generator_obj.__getitem__(0)
        nb_object = int(len(self.test_generator_obj))

        input_shape = list(input_example[0].shape)
        nb_samples = input_shape[0]
        input_shape[0] = input_shape[0] * num_sample_val

        output_shape = list(input_example[1].shape)
        output_shape[0] = output_shape[0] * num_sample_val

        cache_input = np.zeros(shape=input_shape, dtype=input_example[0].dtype)
        cache_output = np.zeros(
            shape=output_shape, dtype=input_example[1].dtype)

        for local_index in range(num_sample_val):
            local_data = self.test_generator_obj.__getitem__(int(nb_object / num_sample_val * local_index))
            cache_input[
            local_index * nb_samples: (local_index + 1) * nb_samples, :
            ] = local_data[0]
            cache_output[
            local_index * nb_samples: (local_index + 1) * nb_samples, :
            ] = local_data[1]

        self.test_generator_obj = (cache_input, cache_output)

    def run(self):
        # we first cache the validation data
        if self.caching_validation:
            self.cache_validation()
        print('Finish cache validation')
        if self.steps_per_epoch > 0:
            self.model_train = self.model.fit(
                self.generator_obj,
                validation_data=self.test_generator_obj,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.epochs,
                max_queue_size=32,
                workers=self.workers,
                shuffle=False,
                use_multiprocessing=self.use_multiprocessing,
                callbacks=self.callbacks_list,
                initial_epoch=0,
            )
        else:
            self.model_train = self.model.fit(
                self.generator_obj,
                validation_data=self.test_generator_obj,
                epochs=self.epochs,
                max_queue_size=32,
                workers=self.workers,
                shuffle=False,
                use_multiprocessing=True,
                callbacks=self.callbacks_list,
                initial_epoch=0,
            )

    def finalize(self):
        draw_plot = True

        if "loss" in self.model_train.history.keys():
            loss = self.model_train.history["loss"]
            # save losses

            save_loss_path = os.path.join(
                self.checkpoints_dir,
                self.model_string + "_loss.npy"
            )
            np.save(save_loss_path, loss)
        else:
            print("Loss data was not present")
            draw_plot = False

        if "val_loss" in self.model_train.history.keys():
            val_loss = self.model_train.history["val_loss"]

            save_val_loss_path = os.path.join(
                self.checkpoints_dir,
                self.model_string + "_val_loss.npy"
            )
            np.save(save_val_loss_path, val_loss)
        else:
            print("Val. loss data was not present")
            draw_plot = False

        # save model
        self.model.save(self.output_model_file_path)

        print("Saved model to disk")

        if draw_plot:
            h = plt.figure()
            plt.plot(loss, label="loss " + self.run_uid)
            plt.plot(val_loss, label="val_loss " + self.run_uid)

            if self.steps_per_epoch > 0:
                plt.xlabel(
                    "number of epochs ("
                    + str(self.batch_size * self.steps_per_epoch)
                    + " samples/epochs)"
                )
            else:
                plt.xlabel(
                    "number of epochs ("
                    + str(self.batch_size * len(self.generator_obj))
                    + " samples/epochs)"
                )

            plt.ylabel("training loss")
            plt.legend()
            save_hist_path = os.path.join(
                self.checkpoints_dir,
                self.model_string + "_losses.png"
            )
            plt.savefig(save_hist_path)
            plt.close(h)


class transfer_trainer(core_trainer):
    def __init__(
            self,
            generator_obj,
            test_generator_obj,
            network_obj,
            trainer_json_path,
            auto_compile=True,
    ):
        super(transfer_trainer, self).__init__(
            generator_obj,
            test_generator_obj,
            network_obj,
            trainer_json_path,
            auto_compile,
        )

    def initialize_network(self):
        super(transfer_trainer, self).initialize_network()
        model_path = self.json_data['model_path']
        self.model.load_weights(model_path)


# This is a helper class to fix an issue in tensorflow 2.0.
# the on_epoch_end callback from sequences is not called.
class OnEpochEnd(tensorflow.keras.callbacks.Callback):
    def __init__(self, callbacks):
        super(OnEpochEnd, self).__init__()
        self.callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback()
