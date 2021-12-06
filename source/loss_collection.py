from tensorflow.keras import backend as K


def loss_selector(loss_type):
    if loss_type == 'mse_with_mask':
        return mse_with_mask
    else:
        return loss_type


def mse_with_mask(y_true, y_pred):
    gt = y_true[:, :, :, 0]
    pred = y_pred[:, :, :, 0]
    mask = y_true[:, :, :, 1]
    return K.sum(K.square(gt - pred) * mask) / K.sum(mask)
