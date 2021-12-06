import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Conv2D,
    Concatenate,
    Add,
    BatchNormalization,
    PReLU,
    Lambda,
)

from tensorflow.keras.regularizers import l1_l2


def res_block_gen(model, kernel_size, filters, strides):
    gen = model
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                   kernel_regularizer=l1_l2(l1=0.01, l2=0.01), bias_regularizer=l1_l2(l1=0.01, l2=0.01))(model)
    model = BatchNormalization(momentum=0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                   kernel_regularizer=l1_l2(l1=0.01, l2=0.01), bias_regularizer=l1_l2(l1=0.01, l2=0.01))(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = Add()([gen, model])
    return model


def fullyconv_mask(json_path):
    def local_network_function(gen_input):
        print("inputs shape:", gen_input.shape)

        # separate into two channels
        true_input = gen_input[:, :, :, :-1]
        print("true input shape: ", true_input.shape)
        mask_layer = Lambda(lambda x: K.expand_dims(x[:, :, :, -1], axis=-1))(gen_input)
        print("mask shape: ", mask_layer.shape)

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                       bias_regularizer=l1_l2(l1=0.01, l2=0.01))(true_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
            model)
        gen_model = model

        # Using 16 Residual Blocks
        for index in range(16):
            model = res_block_gen(model, 3, 64, 1)

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same",
                       kernel_regularizer=l1_l2(l1=0.01, l2=0.01), bias_regularizer=l1_l2(l1=0.01, l2=0.01))(model)
        model = BatchNormalization(momentum=0.5)(model)
        model = Add()([gen_model, model])

        model = Conv2D(filters=1, kernel_size=3, strides=1, padding="same",
                       kernel_regularizer=l1_l2(l1=0.01, l2=0.01), bias_regularizer=l1_l2(l1=0.01, l2=0.01))(model)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
            model)

        model = Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(model)

        output_with_mask = Concatenate(axis=3)([model, mask_layer])
        print("output_with_mask shape:", output_with_mask.shape)

        return output_with_mask

    return local_network_function
