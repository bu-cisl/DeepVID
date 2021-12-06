import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from utils import JsonLoader


class SaveImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, generator_obj, trainer_json_path):
        super(SaveImageCallback, self).__init__()
        self.local_generator = generator_obj
        self.local_mean, self.local_std = generator_obj.__get_norm_parameters__()

        json_obj = JsonLoader(trainer_json_path)
        json_data = json_obj.json_data

        self.save_image_period = json_data["save_image_period"]
        self.output_dir = json_data["output_dir"]

        self.X_sample, self.y_sample = self.local_generator[20]
        self.num_sample = 4
        self.X_sample = self.X_sample[:self.num_sample, :, :, :]
        self.y_sample = self.y_sample[:self.num_sample, :, :, :]
        self.y_sample = self.y_sample * self.local_std + self.local_mean
        self.y_true = np.concatenate([self.y_sample[i, :, :, 0].squeeze() for i in np.arange(self.num_sample)], axis=1)
        self.local_max = np.max(self.y_sample)
        self.local_min = np.min(self.y_sample)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_image_period == 0:
            y_pred = self.model.predict(self.X_sample)
            y_pred = y_pred * self.local_std + self.local_mean
            y_pred = np.concatenate([y_pred[i, :, :, 0].squeeze() for i in np.arange(self.num_sample)], axis=1)
            img = np.concatenate([y_pred, self.y_true], axis=0).squeeze()
            plt.imsave(os.path.join(self.output_dir, 'epoch_{:d}.png'.format(epoch)), img,
                       vmin=self.local_min, vmax=self.local_max)
            # plt.imsave(os.path.join(self.output_dir, 'epoch_{:d}.png'.format(epoch)), img, vmin=0, vmax=1)
