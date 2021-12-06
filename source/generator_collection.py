import numpy as np
from skimage import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from tensorflow.keras.utils import Sequence

import tifffile

from utils import JsonLoader


class BaseGenerator(Sequence):
    def __init__(self, json_path):
        super(BaseGenerator, self).__init__()
        local_json_loader = JsonLoader(json_path)
        local_json_loader.load_json()
        self.json_data = local_json_loader.json_data
        self.local_mean = 1
        self.local_std = 1

    def get_input_size(self):
        """
        This function returns the input size of the
        generator, excluding the batching dimension
        Parameters:
        None
        Returns:
        tuple: list of integer size of input array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[0]

        return local_obj.shape[1:]

    def get_output_size(self):
        """
        This function returns the output size of
        the generator, excluding the batching dimension
        Parameters:
        None
        Returns:
        tuple: list of integer size of output array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[1]

        return local_obj.shape[1:]

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return [np.array([]), np.array([])]

    def __get_norm_parameters__(self):
        """
        This function returns the normalization parameters
        of the generator. This can potentially be different
        for each data sample
        Parameters:
        idx index of the sample
        Returns:
        local_mean
        local_std
        """
        local_mean = self.local_mean
        local_std = self.local_std

        return local_mean, local_std


class DIPGenerator(BaseGenerator):
    def __init__(self, json_path):
        super(DIPGenerator, self).__init__(json_path)

        self.raw_data_file = self.json_data["train_path"]
        self.batch_size = self.json_data["batch_size"]

        if "pre_post_frame" in self.json_data.keys():
            self.pre_frame = self.json_data["pre_post_frame"]
            self.post_frame = self.json_data["pre_post_frame"]
        else:
            self.pre_frame = self.json_data["pre_frame"]
            self.post_frame = self.json_data["post_frame"]

        if "pre_post_omission" in self.json_data.keys():
            self.pre_post_omission = self.json_data["pre_post_omission"]
        else:
            self.pre_post_omission = 0

        self.steps_per_epoch = self.json_data["steps_per_epoch"]

        if "randomize" in self.json_data.keys():
            self.randomize = self.json_data["randomize"]
        else:
            self.randomize = 1

        self.raw_data = io.imread((self.raw_data_file))
        self.preprocess()
        local_data = self.data.flatten()
        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)
        self.epoch_index = 0

        self.list_samples = np.arange(
            self.pre_frame + self.pre_post_omission,
            self.num_frames - self.post_frame - self.pre_post_omission - 1,
        )

        if self.randomize:
            np.random.shuffle(self.list_samples)

    def preprocess(self):
        self.detrend()
        self.num_frames, self.img_rows, self.img_cols = self.data.shape

    def detrend(self, order=2):
        trace = np.mean(self.raw_data, axis=(1, 2))
        X = np.arange(1, trace.shape[0] + 1)
        X = X.reshape(X.shape[0], 1)
        pf = PolynomialFeatures(order)
        Xp = pf.fit_transform(X)
        md = LinearRegression()
        md.fit(Xp, trace)
        self.trend = md.predict(Xp)
        self.data = self.raw_data - np.reshape(self.trend, (self.trend.shape[0], 1, 1))

    def __len__(self):
        "Denotes the total number of batches"
        return int(np.floor(float(len(self.list_samples)) / self.batch_size))

    def __getitem__(self, index):
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        indexes = np.arange(index * self.batch_size,
                            (index + 1) * self.batch_size)

        shuffle_indexes = self.list_samples[indexes]

        input_full = np.zeros(
            [
                self.batch_size,
                self.data.shape[1],
                self.data.shape[2],
                self.pre_frame + self.post_frame,
            ],
            dtype="float32",
        )
        output_full = np.zeros(
            [
                self.batch_size,
                self.data.shape[1],
                self.data.shape[2],
                1
            ],
            dtype="float32",
        )

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame):
        # X : (n_samples, *dim, n_channels)
        "Generates data containing batch_size samples"

        input_full = np.zeros(
            [
                1,
                self.data.shape[1],
                self.data.shape[2],
                self.pre_frame + self.post_frame,
            ],
            dtype="float32",
        )
        output_full = np.zeros(
            [
                1,
                self.data.shape[1],
                self.data.shape[2],
                1
            ], dtype="float32"
        )

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        sel = (input_index >= index_frame - self.pre_post_omission) \
              & (input_index <= index_frame + self.pre_post_omission)
        input_index = input_index[~sel]

        data_img_input = self.data[input_index, :, :]
        data_img_output = self.data[index_frame, :, :]

        data_img_input = np.swapaxes(data_img_input, 1, 2)
        data_img_input = np.swapaxes(data_img_input, 0, 2)

        img_in_shape = data_img_input.shape
        img_out_shape = data_img_output.shape

        data_img_input = (
                                 data_img_input.astype("float32") - self.local_mean
                         ) / self.local_std
        data_img_output = (
                                  data_img_output.astype("float32") - self.local_mean
                          ) / self.local_std

        input_full[0, : img_in_shape[0], : img_in_shape[1], :] = data_img_input
        output_full[0, : img_out_shape[0], : img_out_shape[1], 0] = data_img_output

        return input_full, output_full

    def on_epoch_end(self):
        # We only increase index if steps_per_epoch is set
        # to positive value. -1 will force the generator
        # to not iterate at the end of each epoch
        if self.steps_per_epoch > 0:
            if self.steps_per_epoch * (self.epoch_index + 2) < self.__len__():
                self.epoch_index = self.epoch_index + 1
            else:
                # if we reach the end of the data, we roll over
                self.epoch_index = 0


class DIPN2VGenerator(DIPGenerator):
    def __init__(self, json_path):
        super(DIPN2VGenerator, self).__init__(json_path)
        self.blind_pixel_ratio = self.json_data["blind_pixel_ratio"]
        self.blind_pixel_method = self.json_data["blind_pixel_method"]

        if "cell_mask_path" in self.json_data.keys():
            self.cell_mask_path = self.json_data["cell_mask_path"]
        else:
            self.cell_mask_path = None

        if "cell_mask_index" in self.json_data.keys():
            self.cell_mask_index = self.json_data["cell_mask_index"]
        else:
            self.cell_mask_index = None

        self.read_cell_mask()

    def get_output_size(self):
        """
        This function returns the output size of
        the generator, excluding the batching dimension
        Parameters:
        None
        Returns:
        tuple: list of integer size of output array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[1]
        local_shape = list(local_obj.shape[1:])
        local_shape[-1] = 1

        return tuple(local_shape)

    def read_cell_mask(self):
        """
        This function returns the cell mask for the specific mask index.
        If mask index is not specified, the mask for all cells is returned.
        Input mask shape: (num_cells, h, w)
        :return:
        """
        if self.cell_mask_path is not None:
            cell_mask = tifffile.imread(self.cell_mask_path)
            if self.cell_mask_index is not None:
                self.cell_mask = cell_mask[self.cell_mask_index]
            else:
                self.cell_mask = np.max(cell_mask, axis=0)
            self.cell_mask_pixel_rows, self.cell_mask_pixel_cols = np.nonzero(self.cell_mask)
            self.num_cell_mask_pixel = len(self.cell_mask_pixel_rows)
        else:
            self.cell_mask = None

    def __getitem__(self, index):
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        indexes = np.arange(index * self.batch_size,
                            (index + 1) * self.batch_size)

        shuffle_indexes = self.list_samples[indexes]

        input_full = np.zeros(
            [
                self.batch_size,
                self.data.shape[1],
                self.data.shape[2],
                self.pre_frame + self.post_frame + 2,
            ],
            dtype="float32",
        )
        output_full = np.zeros(
            [
                self.batch_size,
                self.data.shape[1],
                self.data.shape[2],
                2
            ],
            dtype="float32",
        )

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame):
        # X : (n_samples, *dim, n_channels)
        "Generates data containing batch_size samples"

        input_full = np.zeros(
            [
                1,
                self.data.shape[1],
                self.data.shape[2],
                self.pre_frame + self.post_frame + 2,
            ],
            dtype="float32",
        )
        output_full = np.zeros(
            [
                1,
                self.data.shape[1],
                self.data.shape[2],
                2
            ], dtype="float32"
        )

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        sel = (input_index >= index_frame - self.pre_post_omission) \
              & (input_index <= index_frame + self.pre_post_omission)
        input_index = input_index[~sel]

        data_img_central_frame = self.data[index_frame, :, :]
        if self.blind_pixel_ratio > 0:
            data_img_modified, mask = self.__generate_blind_image__(data_img_central_frame)
        else:
            data_img_modified = data_img_central_frame
            mask = np.full_like(data_img_central_frame, 0)

        data_img_input = np.concatenate((
            self.data[input_index, :, :],
            data_img_modified[np.newaxis, :, :],
            mask[np.newaxis, :, :]
        ), axis=0)
        data_img_output = np.concatenate((
            data_img_central_frame[np.newaxis, :, :],
            mask[np.newaxis, :, :]
        ), axis=0)

        data_img_input = np.swapaxes(data_img_input, 1, 2)
        data_img_input = np.swapaxes(data_img_input, 0, 2)

        data_img_output = np.swapaxes(data_img_output, 1, 2)
        data_img_output = np.swapaxes(data_img_output, 0, 2)

        img_in_shape = data_img_input.shape
        img_out_shape = data_img_output.shape

        data_img_input = (
                                 data_img_input.astype("float32") - self.local_mean
                         ) / self.local_std
        data_img_output = (
                                  data_img_output.astype("float32") - self.local_mean
                          ) / self.local_std

        input_full[0, : img_in_shape[0], : img_in_shape[1], :] = data_img_input
        output_full[0, : img_out_shape[0], : img_out_shape[1], :] = data_img_output

        return input_full, output_full

    def __generate_blind_image__(self, img):
        img_rows = img.shape[0]
        img_cols = img.shape[1]

        modified_img = np.copy(img)
        mask = np.zeros(img.shape)

        if self.cell_mask is not None:
            num_blind_pix = int(self.blind_pixel_ratio * self.num_cell_mask_pixel)
            indexes = np.arange(self.num_cell_mask_pixel)
            np.random.shuffle(indexes)
            indexes_target = indexes[:num_blind_pix]
            idx_target_rows = self.cell_mask_pixel_rows[indexes_target]
            idx_target_cols = self.cell_mask_pixel_cols[indexes_target]
        else:
            num_blind_pix = int(self.blind_pixel_ratio * img_rows * img_rows)
            indexes = np.arange(img_rows * img_cols - 1)
            np.random.shuffle(indexes)
            indexes_target = indexes[:num_blind_pix]
            idx_target_rows, idx_target_cols = np.unravel_index(indexes_target, shape=img.shape)

        if self.blind_pixel_method == "zeros":
            modified_img[idx_target_rows, idx_target_cols] = 0
        elif self.blind_pixel_method == "replace":
            if self.cell_mask is not None:
                indexes = np.arange(self.num_cell_mask_pixel)
                np.random.shuffle(indexes)
                indexes_source = indexes[:num_blind_pix]
                idx_source_rows = self.cell_mask_pixel_rows[indexes_source]
                idx_source_cols = self.cell_mask_pixel_cols[indexes_source]
            else:
                indexes = np.arange(img_rows * img_cols - 1)
                np.random.shuffle(indexes)
                indexes_source = indexes[:num_blind_pix]
                idx_source_rows, idx_source_cols = np.unravel_index(indexes_source, shape=img.shape)
            modified_img[idx_target_rows, idx_target_cols] = img[idx_source_rows, idx_source_cols]
        else:
            raise ValueError("Undefined blind pixel generation method. ")

        mask[idx_target_rows, idx_target_cols] = 1

        return modified_img, mask


class CollectorGenerator(BaseGenerator):
    """This class allows to create a generator of generators
    for the purpose of training across multiple files
    All generators must have idendical batch size and input,
    output size but can be different length
    """

    def __init__(self, generator_list):
        self.generator_list = generator_list
        self.nb_generator = len(self.generator_list)
        self.batch_size = self.generator_list[0].batch_size
        self.assign_indexes()
        self.shuffle_indexes()

    def __len__(self):
        "Denotes the total number of batches"
        total_len = 0
        for local_generator in self.generator_list:
            total_len = total_len + local_generator.__len__()

        return total_len

    def assign_indexes(self):
        self.list_samples = []
        current_count = 0

        for generator_index, local_generator in enumerate(self.generator_list):
            local_len = local_generator.__len__()
            for index in np.arange(0, local_len):
                self.list_samples.append(
                    {"generator": generator_index, "index": index})
                current_count = current_count + 1

    def shuffle_indexes(self):
        np.random.shuffle(self.list_samples)

    def __getitem__(self, index):
        # Generate indexes of the batch

        local_index = self.list_samples[index]

        local_generator = self.generator_list[local_index["generator"]]
        local_generator_index = local_index["index"]

        input_full, output_full = local_generator.__getitem__(
            local_generator_index)

        return input_full, output_full
