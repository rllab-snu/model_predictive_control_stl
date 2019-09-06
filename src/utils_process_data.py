# UTILITY-FUNCTIONS (PROCESS DATA)

from __future__ import print_function

import numpy as np
from sklearn import preprocessing


# READ TRAIN-DATA (SINGLE) --------------------------------------------------------------------------------------------#
def read_train_data(filename2read, dim_p, h_prev, h_post, idx_f_use, idx_r_use, num_near, read_r=True,
                    sample_ratio=1.0):

    len_filename = len(filename2read)
    data_size = 0
    for nidx_d in range(0, len_filename):
        filename2read_sel = filename2read[nidx_d]
        data_read_tmp = np.load(filename2read_sel, allow_pickle=True)
        y_train_in = data_read_tmp[()]["data_y_sel_sp"]
        data_size = data_size + y_train_in.shape[0]

    print("data_size (before): {:d}".format(data_size))

    if len(idx_f_use) > 0:
        dim_f = idx_f_use.shape[0]
    else:
        dim_f = -1

    if read_r:
        if len(idx_r_use) > 0:
            dim_r = idx_r_use.shape[0]
        else:
            dim_r = -1

    dim_x, dim_y = dim_p * h_prev, dim_p * h_post
    dim_x_3, dim_y_3 = 3 * h_prev, 3 * h_post

    idx_xin_tmp, idx_yin_tmp = np.arange(0, dim_x_3), np.arange(0, dim_y_3)

    h_prev_ref, h_post_ref = h_prev, h_post

    x_train = np.zeros((data_size, dim_x), dtype=np.float32)
    y_train = np.zeros((data_size, dim_y), dtype=np.float32)
    if dim_f > 0:
        f_train = np.zeros((data_size, dim_f), dtype=np.float32)
    if read_r and dim_r > 0:
        r_train = np.zeros((data_size, dim_r), dtype=np.float32)
    else:
        r_train = []

    cnt_data = 0
    for nidx_d in range(0, len_filename):
        filename2read_sel = filename2read[nidx_d]
        data_read_tmp = np.load(filename2read_sel, allow_pickle=True)

        x_train_in = data_read_tmp[()]["data_x_sel_sp"]
        y_train_in = data_read_tmp[()]["data_y_sel_sp"]

        if nidx_d == 0:
            h_prev_ref, h_post_ref = x_train_in.shape[1], y_train_in.shape[1]

        if dim_p == 2:
            idx_xin_tmp = np.setdiff1d(idx_xin_tmp, np.arange(2, 3 * h_prev_ref, 3))
            idx_yin_tmp = np.setdiff1d(idx_yin_tmp, np.arange(2, 3 * h_post_ref, 3))

        x_train_in = x_train_in[:, idx_xin_tmp]
        y_train_in = y_train_in[:, idx_yin_tmp]

        if dim_f > 0:
            f_train_in = data_read_tmp[()]["data_f"]
            f_train_in = f_train_in[:, idx_f_use]

        if read_r and dim_r > 0:
            r_train_in = data_read_tmp[()]["data_r_sel"]
            r_train_in = r_train_in[:, idx_r_use]

        # Update
        len_before = x_train_in.shape[0]
        idx_rand_tmp_ = np.random.permutation(len_before)
        len_after = int(sample_ratio * len_before)
        idx_rand_tmp = idx_rand_tmp_[np.arange(0, len_after)]

        idx_update_tmp = np.arange(cnt_data, cnt_data + len_after)
        x_train[idx_update_tmp, :] = x_train_in[idx_rand_tmp, :]
        y_train[idx_update_tmp, :] = y_train_in[idx_rand_tmp, :]

        if dim_f > 0:
            f_train[idx_update_tmp, :] = f_train_in[idx_rand_tmp, :]

        if read_r and dim_r > 0:
            r_train[idx_update_tmp, :] = r_train_in[idx_rand_tmp, :]

        cnt_data = cnt_data + len_after

    print("data_size (after): {:d}".format(cnt_data))

    idx_update = np.arange(0, cnt_data)
    x_train = x_train[idx_update, :]
    y_train = y_train[idx_update, :]

    if dim_f > 0:
        f_train = f_train[idx_update, :]

    if read_r and  dim_r > 0:
        r_train = r_train[idx_update, :]

    return x_train, y_train, f_train, r_train


# READ TRAIN-DATA (MULTI) ---------------------------------------------------------------------------------------------#
def read_train_data_multi(filename2read, dim_p, h_prev, h_post, idx_f_use, idx_r_use, num_near, read_r=True,
                          sample_ratio=1.0):

    len_filename = len(filename2read)
    data_size = 0
    for nidx_d in range(0, len_filename):
        filename2read_sel = filename2read[nidx_d]
        data_read_tmp = np.load(filename2read_sel, allow_pickle=True)
        y_train_in = data_read_tmp[()]["data_y_sel_sp"]
        data_size = data_size + y_train_in.shape[0]

    print("data_size (before): {:d}".format(data_size))

    if len(idx_f_use) > 0:
        dim_f = idx_f_use.shape[0]
    else:
        dim_f = -1

    if read_r:
        if len(idx_r_use) > 0:
            dim_r = idx_r_use.shape[0]
        else:
            dim_r = -1

    dim_x, dim_y = dim_p * h_prev, dim_p * h_post
    dim_x_3, dim_y_3 = 3 * h_prev, 3 * h_post

    idx_xin_tmp, idx_yin_tmp = np.arange(0, dim_x_3), np.arange(0, dim_y_3)

    h_prev_ref, h_post_ref = h_prev, h_post

    x_train = np.zeros((data_size, dim_x), dtype=np.float32)
    y_train = np.zeros((data_size, dim_y), dtype=np.float32)
    xnear_train = np.zeros((data_size, num_near * dim_x), dtype=np.float32)
    ynear_train = np.zeros((data_size, num_near * dim_y), dtype=np.float32)
    if dim_f > 0:
        f_train = np.zeros((data_size, dim_f), dtype=np.float32)
    if read_r and dim_r > 0:
        r_train = np.zeros((data_size, dim_r), dtype=np.float32)
        rnear_train = np.zeros((data_size, num_near * dim_r), dtype=np.float32)
    else:
        r_train, rnear_train = [], []

    cnt_data = 0
    for nidx_d in range(0, len_filename):
        filename2read_sel = filename2read[nidx_d]
        data_read_tmp = np.load(filename2read_sel, allow_pickle=True)

        x_train_in = data_read_tmp[()]["data_x_sel_sp"]
        y_train_in = data_read_tmp[()]["data_y_sel_sp"]
        ynear_train_in = data_read_tmp[()]["data_ynear_sp"]
        xnear_train_in = data_read_tmp[()]["data_xnear_sp"]

        if nidx_d == 0:
            h_prev_ref, h_post_ref = x_train_in.shape[1], y_train_in.shape[1]

        if dim_p == 2:
            idx_xin_tmp = np.setdiff1d(idx_xin_tmp, np.arange(2, 3 * h_prev_ref, 3))
            idx_yin_tmp = np.setdiff1d(idx_yin_tmp, np.arange(2, 3 * h_post_ref, 3))

        x_train_in = x_train_in[:, idx_xin_tmp]
        xnear_train_in = xnear_train_in[:, :, idx_xin_tmp]
        xnear_train_in = xnear_train_in.reshape(-1, xnear_train_in.shape[1] * xnear_train_in.shape[2])

        y_train_in = y_train_in[:, idx_yin_tmp]
        ynear_train_in = ynear_train_in[:, :, idx_yin_tmp]
        ynear_train_in = ynear_train_in.reshape(-1, ynear_train_in.shape[1] * ynear_train_in.shape[2])

        if dim_f > 0:
            f_train_in = data_read_tmp[()]["data_f"]
            f_train_in = f_train_in[:, idx_f_use]

        if read_r and dim_r > 0:
            r_train_in = data_read_tmp[()]["data_r_sel"]
            r_train_in = r_train_in[:, idx_r_use]

            rnear_train_in = data_read_tmp[()]["data_rnear"]
            rnear_train_in = rnear_train_in[:, :, idx_r_use]
            rnear_train_in = rnear_train_in.reshape(-1, rnear_train_in.shape[1] * rnear_train_in.shape[2])

        # Update
        len_before = x_train_in.shape[0]
        idx_rand_tmp_ = np.random.permutation(len_before)
        len_after = int(sample_ratio * len_before)
        idx_rand_tmp = idx_rand_tmp_[np.arange(0, len_after)]

        idx_update_tmp = np.arange(cnt_data, cnt_data + len_after)
        x_train[idx_update_tmp, :] = x_train_in[idx_rand_tmp, :]
        y_train[idx_update_tmp, :] = y_train_in[idx_rand_tmp, :]
        xnear_train[idx_update_tmp, :] = xnear_train_in[idx_rand_tmp, :]
        ynear_train[idx_update_tmp, :] = ynear_train_in[idx_rand_tmp, :]

        if dim_f > 0:
            f_train[idx_update_tmp, :] = f_train_in[idx_rand_tmp, :]

        if read_r and dim_r > 0:
            r_train[idx_update_tmp, :] = r_train_in[idx_rand_tmp, :]
            rnear_train[idx_update_tmp, :] = rnear_train_in[idx_rand_tmp, :]

        cnt_data = cnt_data + len_after

    print("data_size (after): {:d}".format(cnt_data))

    idx_update = np.arange(0, cnt_data)
    x_train = x_train[idx_update, :]
    y_train = y_train[idx_update, :]
    xnear_train = xnear_train[idx_update, :]
    ynear_train = ynear_train[idx_update, :]

    if dim_f > 0:
        f_train = f_train[idx_update, :]

    if read_r and  dim_r > 0:
        r_train = r_train[idx_update, :]
        rnear_train = rnear_train[idx_update, :]

    return x_train, y_train, xnear_train, ynear_train, f_train, r_train, rnear_train


# OTHERS --------------------------------------------------------------------------------------------------------------#
# Normalize data: find mean, std for normalization
def normalize_data(do, x):
    # x: (ndarray) data-in

    if len(x) == 0:
        x = []
        _x_mean = []
        _x_scale = []
    else:
        if ~isinstance(x, np.ndarray):
            x = np.array(x)
        shape_x = x.shape
        if len(shape_x) == 1:
            x = np.reshape(x, (1, -1))

        _dim_x = x.shape[1]
        if do == 1:
            _scaler_x = preprocessing.StandardScaler().fit(x)
            x = _scaler_x.transform(x)
            _x_mean = _scaler_x.mean_
            _x_scale = _scaler_x.scale_
        else:
            _x_mean = np.zeros((_dim_x,))
            _x_scale = np.ones((_dim_x,))
    return x, _x_mean, _x_scale


# Modify data_train w.r.t. normalization
def modify_data_wrt_normal(x, x_mean, x_scale):
    if len(x) == 0 or len(x_mean) == 0 or len(x_scale) == 0:
        x = []
    else:
        _n = x.shape[0]
        x = x
        x = x - np.tile(x_mean, (_n, 1))
        x = x / np.tile(x_scale, (_n, 1))
    return x


# Apply normalization
def apply_normal(x, x_mean, x_scale):
    if len(x) == 0 or len(x_mean) == 0 or len(x_scale) == 0:
        x = []
    else:
        if ~isinstance(x, np.ndarray):
            x = np.array(x)
        shape_x = x.shape
        if len(shape_x) == 1:
            x = np.reshape(x, (1, -1))

        _n = x.shape[0]
        x = x
        x = x * np.tile(x_scale, (_n, 1))
        x = x + np.tile(x_mean, (_n, 1))
    return x
