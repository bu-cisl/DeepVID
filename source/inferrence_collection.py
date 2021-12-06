import os

import h5py
import numpy as np
from tqdm import tqdm
import source.loss_collection as lc
from source.utils import JsonLoader
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

import tifffile


class core_inferrence:
    # This is the generic inferrence class
    def __init__(self, inferrence_json_path, generator_obj, network_obj=None):
        self.inferrence_json_path = inferrence_json_path
        self.generator_obj = generator_obj
        self.network_obj = network_obj

        local_json_loader = JsonLoader(inferrence_json_path)
        local_json_loader.load_json()
        self.json_data = local_json_loader.json_data

        self.output_file = self.json_data["output_file"]

        if "save_raw" in self.json_data.keys():
            self.save_raw = self.json_data["save_raw"]
        else:
            self.save_raw = False

        if "rescale" in self.json_data.keys():
            self.rescale = self.json_data["rescale"]
        else:
            self.rescale = True

        if "nb_workers" in self.json_data.keys():
            self.workers = self.json_data["nb_workers"]
        else:
            self.workers = 8

        self.batch_size = self.generator_obj.batch_size
        self.nb_datasets = len(self.generator_obj)
        self.indiv_shape = self.generator_obj.get_output_size()

        self.__load_model()

    def __load_model(self):
        model_path = self.json_data['model_path']

        local_size = self.generator_obj.get_input_size()

        input_img = Input(shape=local_size)
        self.model = Model(input_img, self.network_obj(input_img))

        self.model.load_weights(model_path)

    def run(self):
        final_shape = [self.nb_datasets * self.batch_size]
        final_shape.extend(self.indiv_shape)

        chunk_size = [1]
        chunk_size.extend(self.indiv_shape)

        predictions_data = self.model.predict(
            self.generator_obj,
            batch_size=self.batch_size,
            workers=self.workers,
            verbose=1,
        )

        predictions_data = np.expand_dims(predictions_data[:, :, :, 0], axis=-1)
        local_mean, local_std = \
            self.generator_obj.__get_norm_parameters__()

        # restore normalization
        if self.rescale:
            corrected_data = predictions_data * local_std + local_mean
        else:
            corrected_data = predictions_data

        # restore detrend
        trend = self.generator_obj.trend[
                self.generator_obj.pre_frame:(self.generator_obj.num_frames - self.generator_obj.post_frame - 1)]
        corrected_data = corrected_data + trend[:, np.newaxis, np.newaxis, np.newaxis]

        # save to tiff
        tiff_filename = os.path.splitext(self.output_file)[0] + '.tiff'
        corrected_data = np.squeeze(corrected_data)
        print(corrected_data.shape)
        corrected_data = corrected_data.astype(np.uint16)
        print(np.min(corrected_data), np.max(corrected_data))
        tifffile.imsave(tiff_filename, corrected_data)
