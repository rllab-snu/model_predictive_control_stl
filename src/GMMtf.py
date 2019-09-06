# GAUSSIAN MIXTURE MODEL (TENSORFLOW)

from __future__ import print_function

import numpy as np
import tensorflow as tf


class GMMtf(object):
    def __init__(self, mu, std, frac, n_component, dim_x):
        # mu: tensor(num of data, dim_out, num of components)
        # std: tensor(num of data, dim_out, num of components) <-- diagonal
        # frac: tensor(num of data, num of components)
        # n_component: scalar
        # dim_x: scalar

        self.mu = mu
        self.std = std
        self.cov = tf.math.square(std)
        self.frac = frac
        self.batch_shape = tf.shape(self.mu)  # [..., dim, num_of_component]

        self.n_component = n_component
        self.dim_x = dim_x

    # Sample from gmm
    def sample(self):
        log_frac = tf.math.log(self.frac + float(1e-6))
        z_t = tf.multinomial(logits=log_frac, num_samples=1)  # [..., 1]

        xz_mus_t = tf.transpose(self.mu, [0, 2, 1])  # [..., num_of_component, dim]
        xz_sigs_t = tf.transpose(self.std, [0, 2, 1])  # [..., num_of_component, dim]

        # Choose mixture component corresponding to the latent.
        mask_t = tf.one_hot(z_t[:, 0], depth=self.n_component, dtype=tf.bool, on_value=True, off_value=False)
        xz_mu_t = tf.boolean_mask(xz_mus_t, mask_t)  # [..., dim]
        xz_sig_t = tf.boolean_mask(xz_sigs_t, mask_t)  # [..., dim]

        # Sample x
        x_sample = xz_mu_t + xz_sig_t * tf.random_normal([self.batch_shape[0], self.batch_shape[1]])  # [..., dim]

        return x_sample

    # Get max-mean
    def get_meanmax(self):
        idxmax = tf.argmax(self.frac, axis=1)  # [...]
        selector = tf.expand_dims(tf.one_hot(idxmax, self.n_component), 1)  # [..., 1, num_of_component]
        selector = tf.tile(selector, [1, self.batch_shape[1], 1])  # [..., dim, num_of_component]

        x_max = tf.multiply(self.mu, selector)  # [..., dim, num_of_component]
        x_max = tf.reduce_sum(x_max, axis=2)  # [..., dim]

        return x_max

    # Get negative log likelihood
    def get_negloglikelihood(self, y, eps=float(1e-6)):
        # y: (target) tensor(num of data, dim_out)
        # eps: (scalar)

        cov_c = self.cov + eps
        frac_c = self.frac + eps

        # Define loss
        y_ext_ = tf.expand_dims(y, 2)
        y_ext = tf.tile(y_ext_, [1, 1, self.n_component])  # (num of data, dim_out, num of components)
        diff_y = y_ext - self.mu  # (num of data, dim_out, num of components)
        squared_diff = tf.math.square(diff_y)  # (num of data, dim_out, num of components)
        scaled_squared_diff = tf.math.divide(squared_diff, cov_c)  # (num of data, dim_out, num of components)
        scaled_dist = tf.math.reduce_sum(scaled_squared_diff, 1)  # (num of data, num of components)

        logprod_sigma = tf.math.reduce_sum(tf.math.log(cov_c), 1)  # (num of data, num of components)

        loss_exponent = tf.math.log(frac_c) - 0.5 * float(self.dim_x) * tf.math.log(np.pi * 2) - 0.5 * logprod_sigma \
                        - 0.5 * scaled_dist
        loss_negloglikelihood = tf.math.reduce_mean(-tf.reduce_logsumexp(loss_exponent, axis=1))

        return loss_negloglikelihood

    # Get negative log likelihood (array)
    def get_negloglikelihood_array(self, y, eps=float(1e-6)):
        # _y: (target) tensor(num of data, dim_out)
        # _eps: (scalar)

        cov_c = self.cov + eps
        frac_c = self.frac + eps

        # Define loss
        y_ext_ = tf.expand_dims(y, 2)
        y_ext = tf.tile(y_ext_, [1, 1, self.n_component])  # (num of data, dim_out, num of components)
        diff_y = y_ext - self.mu  # (num of data, dim_out, num of components)
        squared_diff = tf.math.square(diff_y)  # (num of data, dim_out, num of components)
        scaled_squared_diff = tf.math.divide(squared_diff, cov_c)  # (num of data, dim_out, num of components)
        scaled_dist = tf.math.reduce_sum(scaled_squared_diff, 1)  # (num of data, num of components)

        logprod_sigma = tf.math.reduce_sum(tf.math.log(cov_c), 1)  # (num of data, num of components)

        loss_exponent = tf.math.log(frac_c) - 0.5 * float(self.dim_x) * tf.math.log(np.pi * 2) - 0.5 * logprod_sigma \
                        - 0.5 * scaled_dist
        loss_negloglikelihood_array = -tf.math.reduce_logsumexp(loss_exponent, axis=1)

        return loss_negloglikelihood_array


if __name__ == '__main__':

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True

    # mu: tensor(num of data, dim_out, num of components)
    # cov: tensor(num of data, dim_out, num of components) <-- diagonal
    # fracs: tensor(num of data, num of components)
    mu = tf.constant([[-1, 0, 1]], dtype=tf.float32)
    cov = tf.constant([[0.1, 0.1, 0.1]], dtype=tf.float32)
    frac = tf.constant([1 / 3, 1 / 3, 1 / 3], dtype=tf.float32)
    std = tf.math.sqrt(cov)

    mu_r = tf.expand_dims(mu, 0)
    std_r = tf.expand_dims(std, 0)
    fracs_r = tf.expand_dims(frac, 0)

    y_test = tf.placeholder(tf.float32, [None, ])
    y_test_r = tf.reshape(y_test, (-1, 1))
    y_test_in = np.arange(-2, 2, 0.1)

    sim_gmm = GMMtf(mu_r, std_r, fracs_r, 3, 1)
    gmm_sample = sim_gmm.sample()
    log_gmm = -1 * sim_gmm.get_negloglikelihood_array(y_test_r, float(1e-5))

    init = tf.global_variables_initializer()

    N_sample = int(1e4)
    val_gmm_samples = np.zeros((N_sample, ), dtype=np.float32)
    with tf.Session(config=config) as sess:
        # Run the initializer
        sess.run(init)

        for nidx_d in range(0, N_sample):
            val_gmm_sample = sess.run(gmm_sample)
            val_gmm_samples[nidx_d] = val_gmm_sample[0]

        log_gmm_val = sess.run(log_gmm, {y_test: y_test_in})
        # print(log_gmm_val)

    # PLOT 1
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig1, axs1 = plt.subplots(1, 1, tight_layout=True)
    n_bins = 100
    # We can set the number of bins with the `bins` kwarg
    N, bins, patches = axs1.hist(val_gmm_samples, bins=n_bins)

    fracs_hist = N / N.max()
    norm = colors.Normalize(fracs_hist.min(), fracs_hist.max())
    for thisfrac, thispatch in zip(fracs_hist, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    # PLOT 2
    fig2, ax2 = plt.subplots(tight_layout=True)
    ax2.plot(log_gmm_val)

    plt.show()
