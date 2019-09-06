# UTILITY-FUNCTIONS (LEARNING-TENSORFLOW)

from __future__ import print_function

import numpy as np
import tensorflow as tf


# (1) GENERAL FUNCTIONS
def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def xavier_init(size):
    in_dim = float(size[0])
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def convert_tuple_to_tensor2d(tuple_in):
    if isinstance(tuple_in, tuple):
        len_in = len(tuple_in)

        tuple_out = tuple_in[0]
        for nidx_i in range(len_in-1):
            tuple_out = tf.concat([tuple_out, tuple_in[nidx_i+1]], axis=1)

        return tuple_out
    else:
        return tuple_in


def convert_list_to_tensor2d(list_in):
    if isinstance(list_in, list):
        len_in = len(list_in)

        list_out = list_in[0]
        for nidx_i in range(len_in-1):
            list_out = tf.concat([list_out, list_in[nidx_i + 1]], axis=1)

        return list_out
    else:
        return list_in


def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


# (2) Functions: RNN
def dropout_rnn_cell(cell, keep_prob=1.0):
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    return cell


def stacked_rnn_cell(cell_class, num_units_list, keep_prob=1.0):
    return tf.nn.rnn_cell.MultiRNNCell([dropout_rnn_cell(cell_class(nidx_num), keep_prob=keep_prob) for nidx_num in num_units_list])


def project_to_rnn_initstate(cell, tensor_in, scope="initstate_projection"):
    _cell_state_size = cell.state_size
    return project_to_rnn_initstate_helper(_cell_state_size, tensor_in, scope)


def project_to_rnn_initstate_helper(cell_state_size, tensor_in, scope="initstate_projection"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as vs:
        if isinstance(cell_state_size, tf.nn.rnn_cell.LSTMStateTuple):
            init_c = tf.layers.dense(tensor_in, cell_state_size.c, activation=tf.nn.tanh, name="init_c")
            init_h = tf.layers.dense(tensor_in, cell_state_size.h, activation=tf.nn.tanh, name="init_h")
            init_state = tf.nn.rnn_cell.LSTMStateTuple(init_c, init_h)
        elif isinstance(cell_state_size, int):
            init_state = tf.layers.dense(tensor_in, cell_state_size, activation=tf.nn.tanh, name="initState")
        elif isinstance(cell_state_size, tuple):
            init_state = tuple(project_to_rnn_initstate_helper(cs, tensor_in, "cell_" + str(i))
                               for i, cs in enumerate(cell_state_size))
        else:
            raise(Exception("Unknown rnn_cell.state_size!"))
        return init_state


def project_to_tensor(rnn_state):
    if isinstance(rnn_state, tf.nn.rnn_cell.LSTMStateTuple):
        tensor_out = tf.concat([rnn_state.c, rnn_state.h], axis=1)
    elif isinstance(rnn_state, tuple):
        tensor_out_ = tuple(project_to_tensor(cs) for _, cs in enumerate(rnn_state))
        tensor_out = convert_tuple_to_tensor2d(tensor_out_)
    elif isinstance(rnn_state, tf.Tensor):
        tensor_out = rnn_state

    return tensor_out


# (3) Functions: Projection to Gaussian, GMM parameters
def get_gaussian_components(dim_x, tensor_in, stdmax=10, do_sample=1):
    indexes_split = [dim_x, dim_x]
    mu_g, std_g_tmp = tf.split(tensor_in, indexes_split, axis=1)
    std_g = stdmax * tf.nn.sigmoid(std_g_tmp) + float(1e-6)
    # std_g = tf.math.scalar_mul(scalar=stdmax, x=tf.nn.sigmoid(std_g_tmp))

    # Sample
    if do_sample == 1:
        sample_x = mu_g + tf.random_normal(tf.shape(mu_g)) * std_g
    else:
        sample_x = []

    logp_x = get_gaussian_likelihood(sample_x, mu_g, std_g, eps=float(1e-6))

    return mu_g, std_g, logp_x, sample_x


def get_gmm_components(dim_x, n_component_gmm, input_tensor, stdmax=10):
    indexes_split = [dim_x * n_component_gmm, dim_x * n_component_gmm, n_component_gmm]
    mu_gmm_tmp, std_gmm_tmp, frac_gmm_tmp = tf.split(input_tensor, indexes_split, axis=1)

    mu_gmm = mu_gmm_tmp

    # std_gmm = stdmax * tf.nn.sigmoid(std_gmm_tmp)
    std_gmm = tf.math.scalar_mul(scalar=stdmax, x=tf.nn.sigmoid(std_gmm_tmp))
    # std_gmm = tf.multiply(_sigmamax, tf.nn.sigmoid(std_gmm_tmp))

    frac_gmm_out_max = tf.reduce_max(frac_gmm_tmp, axis=1)
    frac_gmm_out_max_ext_ = tf.expand_dims(frac_gmm_out_max, 1)
    frac_gmm_out_max_ext = tf.tile(frac_gmm_out_max_ext_, [1, n_component_gmm])
    frac_gmm_ = frac_gmm_tmp - frac_gmm_out_max_ext
    frac_gmm = tf.nn.softmax(frac_gmm_)

    mu_gmm = tf.reshape(mu_gmm, [-1, dim_x, n_component_gmm])  # (num of data, dim_out, num of components)
    std_gmm = tf.reshape(std_gmm, [-1, dim_x, n_component_gmm])  # (num of data, dim_out, num of components)
    frac_gmm = tf.reshape(frac_gmm, [-1, n_component_gmm])  # (num of data, num of components)

    return mu_gmm, std_gmm, frac_gmm


def get_gaussian_likelihood(x, mu, std, eps=float(1e-6)):
    log_std = tf.math.log(std + eps)
    scaled_dist = tf.math.square((x - mu)/(std + eps))
    pre_sum = -0.5 * (scaled_dist + 2 * log_std + np.log(2 * np.pi))
    log_likelihood = tf.reduce_sum(pre_sum, axis=1)
    return log_likelihood


def get_kl_two_univariate_gaussians(z1_mean, z1_std, z2_mean, z2_std, eps=float(1e-6)):
    # KL Divergence: KL(p1|p2)
    mean_diff = z1_mean - z2_mean
    scaled_min_diff = tf.math.divide(tf.math.square(z1_std) + tf.math.square(mean_diff),
                                     2.0 * tf.math.square(z2_std) + eps)
    std_ratio = tf.math.divide(z2_std, z1_std + eps)
    log_exponent = tf.math.log(std_ratio + eps) + scaled_min_diff - 0.5
    KL_div_ = tf.math.reduce_sum(log_exponent, axis=1)
    KL_div = tf.math.reduce_mean(KL_div_, axis=0)
    return KL_div


# CHECK ---------------------------------------------------------------------------------------------------------------#
# (check) print sigma value
def check_sigma(sigma):
    # sigma: array(num of data_train, dim_out, num of components)
    _N_data = sigma.shape[0]
    _dim = sigma.shape[1]
    _n_c = sigma.shape[2]

    _sigma_min = np.zeros((_dim, ))
    _sigma_max = np.zeros((_dim, ))
    _sigma_mean = np.zeros((_dim, ))
    for nidx_dim in range(0, _dim):
        _sigma_tmp = sigma[:, nidx_dim, :]
        _sigma_tmp = np.reshape(_sigma_tmp, -1)
        _sigma_min[nidx_dim] = np.min(_sigma_tmp)
        _sigma_max[nidx_dim] = np.max(_sigma_tmp)
        _sigma_mean[nidx_dim] = np.mean(_sigma_tmp)

    _txt_sigma_min = 'sig_min: '
    _txt_sigma_max = 'sig_max: '
    _txt_sigma_mean = 'sig_mean: '
    for nidx_dim in range(0, _dim):
        _str_min_tmp = "{:.3f}".format(_sigma_min[nidx_dim])
        _str_max_tmp = "{:.3f}".format(_sigma_max[nidx_dim])
        _str_mean_tmp = "{:.3f}".format(_sigma_mean[nidx_dim])
        if nidx_dim < (_dim - 1):
            _txt_sigma_min = _txt_sigma_min + _str_min_tmp + ', '
            _txt_sigma_max = _txt_sigma_max + _str_max_tmp + ', '
            _txt_sigma_mean = _txt_sigma_mean + _str_mean_tmp + ', '
        else:
            _txt_sigma_min = _txt_sigma_min + _str_min_tmp
            _txt_sigma_max = _txt_sigma_max + _str_max_tmp
            _txt_sigma_mean = _txt_sigma_mean + _str_mean_tmp

    print(_txt_sigma_min + ', ' + _txt_sigma_max + ', ' + _txt_sigma_mean)


# (check) sample from Gaussian mixture model
def sample_from_gmm(mu, sigma, fracs, n_sample):
    # mu: np-array(num of data_train, dim_out, num of components)
    # sigma: np-array(num of data_train, dim_out, num of components)
    # fracs: np-array(num of data_train, num of components)
    # n_sample: scalar

    _N_data = mu.shape[0]
    _dim = mu.shape[1]
    _n_c = mu.shape[2]

    _sampled_data_out = np.zeros((_N_data,), dtype=np.object)
    for nidx_d in range(0, _N_data):
        _sampled_data = np.zeros((n_sample, _dim), dtype=np.float32)
        _fracs_tmp = np.reshape(fracs[nidx_d, :], (-1))
        for nidx_c in range(0, _n_c):
            _mu_sel = np.reshape(mu[nidx_d, :, nidx_c], -1)
            _sigma_sel = np.sqrt(np.reshape(sigma[nidx_d, :, nidx_c], -1))
            _sigma_diag_sel = np.diag(_sigma_sel)
            _s_out = np.random.multivariate_normal(_mu_sel, _sigma_diag_sel, n_sample)
            _sampled_data = _sampled_data + _fracs_tmp[nidx_c] * _s_out

        _sampled_data_out[nidx_d] = _sampled_data

    return _sampled_data_out


# (check) choose mu with the highest weight
def choose_max_mu(mu, fracs):
    _N_data = mu.shape[0]
    _dim = mu.shape[1]

    _max_mu_out = np.zeros((_N_data, _dim))
    for nidx_d in range(0, _N_data):
        _fracs_tmp = np.reshape(fracs[nidx_d, :], (-1))
        _idx_max_tmp = np.argmax(_fracs_tmp)
        _mu_max_tmp = np.reshape(mu[nidx_d, :, _idx_max_tmp], -1)
        _max_mu_out[nidx_d, :] = _mu_max_tmp

    return _max_mu_out


# (check) print error
def check_result_max_mu(y_pred, y_test, dim_y):
    _mean_error_maxmu = np.mean(np.abs(y_pred - y_test), axis=0)
    _max_error_maxmu = np.max(np.abs(y_pred - y_test), axis=0)
    _txt_mean_error_maxmu = 'mean_error (maxmu): '
    _txt_max_error_maxmu = 'max_error (maxmu): '
    for nidx_y in range(0, dim_y):
        _str_mean_tmp = "{:.4f}".format(_mean_error_maxmu[nidx_y])
        _str_max_tmp = "{:.4f}".format(_max_error_maxmu[nidx_y])
        if nidx_y < (dim_y - 1):
            _txt_mean_error_maxmu = _txt_mean_error_maxmu + _str_mean_tmp + ', '
            _txt_max_error_maxmu = _txt_max_error_maxmu + _str_max_tmp + ', '
        else:
            _txt_mean_error_maxmu = _txt_mean_error_maxmu + _str_mean_tmp
            _txt_max_error_maxmu = _txt_max_error_maxmu + _str_max_tmp

    print(_txt_mean_error_maxmu + ", " + _txt_max_error_maxmu)
