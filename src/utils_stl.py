# UTILITY-FUNCTIONS (STL-ROBUSTNESS)
# Function list

import numpy as np
import math
from src.utils_sim import *


# Compute robustness (lane-observation: down & up)
def compute_robustness_lane(traj_in, cp_l_d, cp_l_u, rad_l):
    # traj_in: (ndarray) ego-vehicle trajectory (dim = H x 2)
    # cp_l_d, cp_l_u: (ndarray) lane-point (dim = 2)
    # rad_l: (scalar) lane-angle

    traj_in = make_numpy_array(traj_in, keep_1dim=False)
    cp_l_d = make_numpy_array(cp_l_d, keep_1dim=True)
    cp_l_u = make_numpy_array(cp_l_u, keep_1dim=True)

    h = traj_in.shape[0]

    r_l_down_array = np.zeros((h,), dtype=np.float32)
    r_l_up_array = np.zeros((h,), dtype=np.float32)
    for nidx_h in range(0, h):
        y_down = math.tan(rad_l) * (traj_in[nidx_h, 0] - cp_l_d[0]) + cp_l_d[1]
        r_l_down_tmp = traj_in[nidx_h, 1] - y_down

        y_up = math.tan(rad_l) * (traj_in[nidx_h, 0] - cp_l_u[0]) + cp_l_u[1]
        r_l_up_tmp = y_up - traj_in[nidx_h, 1]

        r_l_down_array[nidx_h] = r_l_down_tmp
        r_l_up_array[nidx_h] = r_l_up_tmp

    r_l_down = np.min(r_l_down_array)
    r_l_up = np.min(r_l_up_array)

    return r_l_down, r_l_up, r_l_down_array, r_l_up_array


# Compute robustness (collision)
def compute_robustness_collision(traj_in, size_in, traj_oc_in, size_oc_in, param_x_1=1.0, param_x_2=0.125,
                                 param_x_3=1.25, param_y_1=0.1, param_y_2=0.125, param_y_3=0.1, param_y_4=0.2,
                                 lanewidth_cur=3.28, use_mod=1):
    # traj_in: (ndarray) ego-vehicle trajectory (dim = H x 2)
    # size_in: (ndarray or list) ego-vehicle size (dim = 2)
    # traj_oc_in: (ndarray) other-vehicle trajectory (dim = H x 2)
    # size_oc_in: (ndarray) other-vehicle size (dim = H x 2)

    traj_in = make_numpy_array(traj_in, keep_1dim=False)
    size_in = make_numpy_array(size_in, keep_1dim=True)
    traj_oc_in = make_numpy_array(traj_oc_in, keep_1dim=False)
    size_oc_in = make_numpy_array(size_oc_in, keep_1dim=False)

    h = traj_in.shape[0]

    if h > 1:
        diff_traj_tmp = traj_in[range(1, h), 0:2] - traj_in[range(0, (h - 1)), 0:2]
        dist_traj_tmp = np.sqrt(np.sum(diff_traj_tmp * diff_traj_tmp, axis=1))

    r_oc_array = np.zeros((h,), dtype=np.float32)
    dist_traj = 0.0
    for nidx_h in range(0, h):
        x_oc_tmp, y_oc_tmp = traj_oc_in[nidx_h, 0], traj_oc_in[nidx_h, 1]

        if use_mod == 1:
            # Modify size w.r.t. distance
            if h > 1:
                if nidx_h < dist_traj_tmp.shape[0]:
                    dist_traj = min(dist_traj + dist_traj_tmp[nidx_h] * lanewidth_cur / 3.28, 1000)
            mod_size_x = min(param_x_1 * (math.exp(param_x_2 * dist_traj) - 1.0), param_x_3)
            mod_size_y = param_y_1 + min(param_y_2 * (math.exp(param_y_3 * dist_traj) - 1.0), param_y_4)
        else:
            mod_size_x, mod_size_y = 0.0, 0.0
        rx_oc = (size_oc_in[nidx_h, 0] + size_in[0]) / 2.0 + mod_size_x
        ry_oc = (size_oc_in[nidx_h, 1] + size_in[1]) / 2.0 + mod_size_y

        r_b_oc0, r_b_oc1 = float(-x_oc_tmp - rx_oc), float(x_oc_tmp - rx_oc)
        r_b_oc2, r_b_oc3 = float(-y_oc_tmp - ry_oc), float(y_oc_tmp - ry_oc)

        r_oc0, r_oc1 = traj_in[nidx_h, 0] + r_b_oc0, -traj_in[nidx_h, 0] + r_b_oc1
        r_oc2, r_oc3 = traj_in[nidx_h, 1] + r_b_oc2, -traj_in[nidx_h, 1] + r_b_oc3

        r_oc_tmp = max([r_oc0, r_oc1, r_oc2, r_oc3])
        r_oc_array[nidx_h] = r_oc_tmp

    r_oc = np.min(r_oc_array)

    return r_oc, r_oc_array


# Compute robustness (speed)
def compute_robustness_speed(vtraj_in, v_th):
    # vtraj_in: (ndarray) ego-vehicle velocity trajectory (dim = H)

    vtraj_in = make_numpy_array(vtraj_in, keep_1dim=True)

    len_v_traj_in = vtraj_in.shape[0]
    r_speed_array = np.zeros((len_v_traj_in, ), dtype=np.float32)
    for nidx_h in range(0, len_v_traj_in):
        r_speed_array[nidx_h] = v_th - vtraj_in[nidx_h]

    r_speed = np.min(r_speed_array)

    return r_speed, r_speed_array


# Compute robustness (until)
def compute_robustness_until(traj_in, size_in, vtraj_in, traj_cf_in, size_cf_in, t_s, t_a, t_b, v_th, d_th):
    # traj_in: (ndarray) ego-vehicle trajectory (dim = H x 2)
    # size_in: (ndarray or list) ego-vehicle size (dim = 2)
    # vtraj_in: (ndarray) ego-vehicle velocity trajectory (dim = H)
    # traj_cf_in: (ndarray) centerlane-front vehicle trajectory (dim = H x 2)
    # size_cf_in: (ndarray) centerlane-front vehicle size (dim = H x 2)

    traj_in = make_numpy_array(traj_in, keep_1dim=False)
    size_in = make_numpy_array(size_in, keep_1dim=True)
    vtraj_in = make_numpy_array(vtraj_in, keep_1dim=True)
    traj_cf_in = make_numpy_array(traj_cf_in, keep_1dim=False)
    size_cf_in = make_numpy_array(size_cf_in, keep_1dim=False)

    len_1 = t_b - t_a + 1
    r_1 = np.zeros((len_1, ), dtype=np.float32)
    for idx_t1 in range(t_a, t_b + 1):
        r_phi_2 = traj_in[idx_t1, 0] - traj_cf_in[idx_t1, 0] + (size_in[0] + size_cf_in[idx_t1, 0]) / 2.0 + d_th

        len_3_tmp = idx_t1 - t_s + 1
        r_3_tmp = np.zeros((len_3_tmp, ), dtype=np.float32)
        for idx_t2 in range(t_s, idx_t1 + 1):
            r_phi_1 = v_th - vtraj_in[idx_t2]
            r_3_tmp[idx_t2 - t_s] = r_phi_1

        rmin_3_tmp = np.amin(r_3_tmp, axis=0)
        r_1[idx_t1 - t_a] = np.amin(np.array([r_phi_2, rmin_3_tmp]), axis=0)

    r_out = np.amax(r_1)
    return r_out


# Compute robustness of STL (part)
def compute_robustness_part(traj_in, size_in, cp_l_d, cp_l_u, rad_l, traj_cf_in, size_cf_in, traj_rest_in, size_rest_in,
                            idx_h_ov, v_th):
    # traj_in: (ndarray) ego-vehicle trajectory (dim = H x 2)
    # size_in: (ndarray or list) ego-vehicle size (dim = 2)
    #       1: Lane-observation (down, right)
    #       2: Lane-observation (up, left)
    #       3: Collision (front)
    #       4: Collision (others)
    #       5: Speed-limit
    #       6: Slow-down

    traj_in = make_numpy_array(traj_in, keep_1dim=False)
    size_in = make_numpy_array(size_in, keep_1dim=True)

    # Rule 1-2: lane-observation (down, up)
    r_l_down, r_l_up, r_l_down_array, r_l_up_array = compute_robustness_lane(traj_in, cp_l_d, cp_l_u, rad_l)

    # Rule 3: collision (front)
    r_c_cf, r_c_cf_array = compute_robustness_collision(traj_in, size_in, traj_cf_in[idx_h_ov, :], size_cf_in[idx_h_ov, :])

    # Rule 4: collision (rest)
    num_oc_rest = len(traj_rest_in)
    r_oc_rest_, r_oc_rest_array_ = [], []
    for nidx_oc in range(0, num_oc_rest):
        traj_rest_in_tmp = traj_rest_in[nidx_oc]
        size_rest_in_tmp = size_rest_in[nidx_oc]
        r_c_tmp, r_c_array_tmp = compute_robustness_collision(traj_in, size_in, traj_rest_in_tmp[idx_h_ov, :],
                                                              size_rest_in_tmp[idx_h_ov, :])
        r_oc_rest_.append(r_c_tmp)
        r_oc_rest_array_.append(r_c_array_tmp)
    r_oc_rest_ = np.asarray(r_oc_rest_)

    r_c_rest = np.min(r_oc_rest_)
    r_c_rest_array = r_oc_rest_

    # Rule 5: speed-limit
    r_speed, r_speed_array = compute_robustness_speed(traj_in[:, -1], v_th)

    return r_l_down, r_l_up, r_c_cf, r_c_rest, r_c_rest_array, r_speed


# Compute robustness of STL
def compute_robustness(traj_in, vtraj_in, size_in, cp_l_d, cp_l_u, rad_l, traj_cf_in, size_cf_in, traj_rest_in,
                       size_rest_in, idx_h_ov, v_th, until_t_s, until_t_a, until_t_b, until_v_th, until_d_th):
    # traj_in: (ndarray) ego-vehicle trajectory (dim = H x 2)
    # vtraj_in: (ndarray) ego-vehicle velocity trajectory (dim = H)
    # size_in: (ndarray or list) ego-vehicle size (dim = 2)
    #       1: Lane-observation (down, right)
    #       2: Lane-observation (up, left)
    #       3: Collision (front)
    #       4: Collision (others)
    #       5: Speed-limit
    #       6: Slow-down

    traj_in = make_numpy_array(traj_in, keep_1dim=False)
    vtraj_in = make_numpy_array(vtraj_in, keep_1dim=True)
    size_in = make_numpy_array(size_in, keep_1dim=True)

    # Rule 1-2: lane-observation (down, up)
    r_l_down, r_l_up, r_l_down_array, r_l_up_array = compute_robustness_lane(traj_in, cp_l_d, cp_l_u, rad_l)

    # Rule 3: collision (front)
    r_c_cf, r_c_cf_array = compute_robustness_collision(traj_in, size_in, traj_cf_in[idx_h_ov, :],
                                                        size_cf_in[idx_h_ov, :], use_mod=0)

    # Rule 4: collision (rest)
    num_oc_rest = len(traj_rest_in)
    r_oc_rest_, r_oc_rest_array_ = [], []
    for nidx_oc in range(0, num_oc_rest):
        traj_rest_in_tmp = traj_rest_in[nidx_oc]
        size_rest_in_tmp = size_rest_in[nidx_oc]
        r_c_tmp, r_c_array_tmp = compute_robustness_collision(traj_in, size_in, traj_rest_in_tmp[idx_h_ov, :],
                                                              size_rest_in_tmp[idx_h_ov, :])
        r_oc_rest_.append(r_c_tmp)
        r_oc_rest_array_.append(r_c_array_tmp)
    r_oc_rest_ = np.asarray(r_oc_rest_)

    r_c_rest = np.min(r_oc_rest_)
    r_c_rest_array = r_oc_rest_

    # Rule 5: speed-limit
    r_speed, r_speed_array = compute_robustness_speed(vtraj_in, v_th)

    # Rule 6: until-logic
    r_until = compute_robustness_until(traj_in[:, 0:2], size_in, vtraj_in, traj_cf_in[idx_h_ov, :],
                                       size_cf_in[idx_h_ov, :], until_t_s, until_t_a, until_t_b, until_v_th, until_d_th)

    return r_l_down, r_l_up, r_c_cf, r_c_rest, r_c_rest_array, r_speed, r_until


def compute_lane_constraints(pnt_down, pnt_up, ry, lane_angle, margin_dist, cp2rotate, theta2rotate):
    # pnt_down, pnt_up: (ndarray) point (dim = 2)
    # ry: (scalar) width-size
    # lane_angle: (scalar) lane-heading (rad)
    # margin_dist: (scalar) margin dist (bigger -> loosing constraints)
    # cp2rotate: (ndarray) center point to convert (dim = 2)
    # theta2rotate: (scalar) angle (rad) to convert

    pnt_down = make_numpy_array(pnt_down, keep_1dim=True)
    pnt_up = make_numpy_array(pnt_up, keep_1dim=True)

    pnt_down_r = np.reshape(pnt_down[0:2], (1, 2))
    pnt_up_r = np.reshape(pnt_up[0:2], (1, 2))
    pnt_down_conv_ = get_rotated_pnts_tr(pnt_down_r, -cp2rotate, -theta2rotate)
    pnt_up_conv_ = get_rotated_pnts_tr(pnt_up_r, -cp2rotate, -theta2rotate)
    pnt_down_conv, pnt_up_conv = pnt_down_conv_[0, :], pnt_up_conv_[0, :]
    lane_angle_r = angle_handle(lane_angle - theta2rotate)

    margin_dist = margin_dist - ry / 2.0
    cp_l_d = pnt_down_conv + np.array([+margin_dist*math.sin(lane_angle_r),
                                       -margin_dist*math.cos(lane_angle_r)], dtype=np.float32)
    cp_l_u = pnt_up_conv + np.array([-margin_dist * math.sin(lane_angle_r),
                                     +margin_dist * math.cos(lane_angle_r)], dtype=np.float32)
    rad_l = lane_angle_r

    return cp_l_d, cp_l_u, rad_l


def compute_collision_constraints(h, xinit_conv, ry, traj_ov_near_list, size_ov_near_list, id_near, cp2rotate, theta2rotate):
    # h: (scalar) horizon
    # xinit_conv: (ndarray) init state
    # ry: (scalar) width
    # traj_ov_near_list: (list)-> (ndarray) x y theta (dim = N x 3)
    #                  : [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
    # size_ov_near_list: (list)-> (ndarray) dx dy (dim = N x 2)
    #                  : [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
    # id_near: (ndarray) selected indexes [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
    # cp2rotate: (ndarray) center point to convert (dim = 2)
    # theta2rotate: (scalar) angle (rad) to convert

    xinit_conv = make_numpy_array(xinit_conv, keep_1dim=True)
    id_near = make_numpy_array(id_near, keep_1dim=True)
    cp2rotate = make_numpy_array(cp2rotate, keep_1dim=True)

    # Do reset
    traj_ov, size_ov = [], []

    len_near = len(traj_ov_near_list)

    for nidx_l in range(0, len_near):
        traj_ov_near_list_sel = traj_ov_near_list[nidx_l]
        size_ov_near_list_sel = size_ov_near_list[nidx_l]
        id_near_sel = id_near[nidx_l]

        traj_ov_near_list_sel = make_numpy_array(traj_ov_near_list_sel, keep_1dim=False)
        size_ov_near_list_sel = make_numpy_array(size_ov_near_list_sel, keep_1dim=True)

        if nidx_l == 4:  # id_cf
            if id_near_sel == -1:
                traj_ov_cf = traj_ov_near_list_sel
                size_ov_cf = np.zeros((h + 1, 2), dtype=np.float32)
            else:
                traj_tmp = np.zeros((h + 1, 3), dtype=np.float32)
                traj_tmp[:, 0:2] = get_rotated_pnts_tr(traj_ov_near_list_sel[:, 0:2], -cp2rotate, -theta2rotate)
                traj_tmp[:, 2] = angle_handle(traj_ov_near_list_sel[:, 2] - theta2rotate)

                diff_tmp = traj_tmp[:, 0:2] - np.reshape(xinit_conv[0:2], (1, 2))
                dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
                idx_tmp_ = np.where(dist_tmp > 100.0)
                idx_tmp_ = idx_tmp_[0]
                if len(idx_tmp_) > 0:
                    traj_tmp[idx_tmp_, 0:2] = [xinit_conv[0] - 200, xinit_conv[1] - 200]

                traj_ov_cf = traj_tmp

                size_ov_cf = np.zeros((h + 1, 2), dtype=np.float32)
                for nidx_h in range(0, h + 1):
                    traj_tmp_sel = traj_tmp[nidx_h, :]
                    size_ov_near_list_sel_new = np.array([size_ov_near_list_sel[0], size_ov_near_list_sel[1]],
                                                         dtype=np.float32)
                    if size_ov_near_list_sel_new[1] < 0.9 * ry:
                        size_ov_near_list_sel_new[1] = 0.9 * ry
                    size_tmp_sel = get_modified_size_linear(size_ov_near_list_sel_new, traj_tmp_sel[2])
                    size_ov_cf[nidx_h, :] = size_tmp_sel

        else:
            if id_near_sel == -1:
                traj_tmp = traj_ov_near_list_sel
                size_tmp = np.zeros((h + 1, 2), dtype=np.float32)
            else:
                traj_tmp = np.zeros((h + 1, 3), dtype=np.float32)
                traj_tmp[:, 0:2] = get_rotated_pnts_tr(traj_ov_near_list_sel[:, 0:2], -cp2rotate, -theta2rotate)
                traj_tmp[:, 2] = angle_handle(traj_ov_near_list_sel[:, 2] - theta2rotate)

                diff_tmp = traj_tmp[:, 0:2] - np.reshape(xinit_conv[0:2], (1, 2))
                dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
                idx_tmp_ = np.where(dist_tmp > 100.0)
                idx_tmp_ = idx_tmp_[0]
                if len(idx_tmp_) > 0:
                    traj_tmp[idx_tmp_, 0:2] = [xinit_conv[0] - 100, xinit_conv[1] - 100]

                size_tmp = np.zeros((h + 1, 2), dtype=np.float32)
                for nidx_h in range(0, h + 1):
                    traj_tmp_sel = traj_tmp[nidx_h, :]
                    size_tmp_sel = get_modified_size_linear(size_ov_near_list_sel, traj_tmp_sel[2])
                    size_tmp[nidx_h, :] = size_tmp_sel

            traj_ov.append(traj_tmp)
            size_ov.append(size_tmp)

    return traj_ov_cf, size_ov_cf, traj_ov, size_ov


# Get modified size for linearization
def get_modified_size_linear(size_in, heading, w=0.4):
    # size_in: (ndarray) dx dy (dim = 2)
    # heading: (scalar) heading
    # w: (scalar) weight

    size_in = make_numpy_array(size_in, keep_1dim=True)

    if abs(math.cos(heading)) <= 0.125:
        size_out = np.flipud(size_in)
    elif abs(math.sin(heading)) <= 0.125:
        size_out = size_in
    elif abs(math.cos(heading)) < 1 / math.sqrt(2):
        size_x = w * abs(size_in[0] * math.cos(heading)) + abs(size_in[1] * math.sin(heading))
        size_y = abs(size_in[0] * math.sin(heading)) + w * abs(size_in[1] * math.cos(heading))
        size_out = np.array([size_x, size_y], dtype=np.float32)
    else:
        size_x = abs(size_in[0] * math.cos(heading)) + w * abs(size_in[1] * math.sin(heading))
        size_y = w * abs(size_in[0] * math.sin(heading)) + abs(size_in[1] * math.cos(heading))
        size_out = np.array([size_x, size_y], dtype=np.float32)

    # size_out = size_in
    return size_out


# Convert state
def convert_state(x_in, cp2rotate, theta2rotate):
    # x_in: (ndarray) ego-vehicle system state (dim = 4)
    # cp2rotate: (ndarray) center point to convert (dim = 2)
    # theta2rotate: (scalar) angle (rad) to convert

    x_in = make_numpy_array(x_in, keep_1dim=True)
    cp2rotate = make_numpy_array(cp2rotate, keep_1dim=True)

    dim_x = 4

    # Convert state (ego-vehicle)
    xconv_in = np.zeros((dim_x,), dtype=np.float32)
    x_in_r = np.reshape(x_in[0:2], (1, 2))
    xconv_in_tmp1 = get_rotated_pnts_tr(x_in_r, -cp2rotate, -theta2rotate)
    xconv_in_tmp1 = xconv_in_tmp1.reshape(-1)
    xconv_in[0:2] = xconv_in_tmp1
    xconv_in[2] = angle_handle(x_in[2] - theta2rotate)
    xconv_in[3] = x_in[3]

    return xconv_in


# Convert trajectory
def convert_trajectory(traj_in, cp2rotate, theta2rotate):
    # traj_in: (ndarray) trajectory (dim = N x 4 or N x 3)
    # cp2rotate: (ndarray) center point to convert (dim = 2)
    # theta2rotate: (scalar) angle (rad) to convert

    traj_in = make_numpy_array(traj_in, keep_1dim=False)
    cp2rotate = make_numpy_array(cp2rotate, keep_1dim=True)

    len_traj = traj_in.shape[0]
    dim_x = traj_in.shape[1]

    traj_conv = np.zeros((len_traj, dim_x), dtype=np.float32)
    traj_conv[:, 0:2] = get_rotated_pnts_tr(traj_in[:, 0:2], -cp2rotate, -theta2rotate)
    traj_conv[:, 2] = angle_handle(traj_in[:, 2] - theta2rotate)
    if dim_x == 4:
        traj_conv[:, 3] = traj_in[:, 3]

    return traj_conv