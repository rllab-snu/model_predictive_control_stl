# UTILITY-FUNCTIONS (SIMULATOR)
# Function list
#       get_data_trimmed_t(data_v, t_cur, t_range)
#       get_data_trimmed_id_t(data_v, id_tv, t_cur, t_range)
#       get_mid_pnts_lr(pnts_l, pnts_r, num_intp)
#       get_index_seg_and_lane(pos_i, pnts_poly_track)
#       get_index_seg_and_lane_outside(pos_i, pnts_poly_track)
#       get_selected_vehicles(data_v, id_sel)
#       get_selected_vehicles_near(data_v, id_near)
#       get_vehicle_trajectory(data_v, time_in, horizon, handle_remain=0)
#       get_vehicle_trajectory_near(data_v_list, id_near, time_in, horizon, do_reverse=0, handle_remain=1)
#       get_vehicle_trajectory_per_id(data_v, time_in, id_in, horizon, do_reverse=0, handle_remain=1)
#       get_vehicle_vtrajectory_per_id(data_v, time_in, id_in, horizon, do_reverse=0, handle_remain=1)
#       check_collision(data_tv, data_ov)
#       check_collision_near(data_ev, data_ov, time_in)
#       get_lane_rad(pnt, pnts_poly_track, pnts_lr_border_track)
#       get_lane_cp_wrt_mtrack(sim_track, pos, seg, lane)
#       get_lane_angle_wrt_mtrack(sim_track, pos, seg, lane, delta=4)
#       get_feature_part(sim_track, data_t, use_intp=0)
#       get_feature(sim_track, data_ev, data_ov, use_intp=0)
#       get_control_set_naive(u0_set, u1_set)
#       get_dist2goal(pnt_t, seg_t, lane_t, indexes_goal, pnts_goal)
#       encode_trajectory(traj_in, traj_type, pnts_poly_track, pnts_lr_border_track, is_track_simple=0)
#       decode_trajectory(pos_in, val_in, h, traj_type, pnts_poly_track, pnts_lr_border_track, is_track_simple=0)


from src.utils import *


# Get trimmed data (w.r.t. time)
def get_data_trimmed_t(data_v, t_cur, t_range):
    # data_v: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)

    data_v = make_numpy_array(data_v, keep_1dim=False)

    t_min, t_max = t_cur - t_range, t_cur + t_range
    idx_found_t1 = np.where(t_min <= data_v[:, 0])
    idx_found_t2 = np.where(data_v[:, 0] <= t_max)
    idx_found_t = np.intersect1d(idx_found_t1, idx_found_t2)
    data_v = data_v[idx_found_t, :]

    return data_v


# Get trimmed data (w.r.t. id, time)
def get_data_trimmed_id_t(data_v, id_tv, t_cur, t_range):
    # data_v: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)

    data_v = make_numpy_array(data_v, keep_1dim=False)

    t_min, t_max = t_cur - t_range, t_cur + t_range
    idx_found_t1 = np.where(t_min <= data_v[:, 0])
    idx_found_t2 = np.where(data_v[:, 0] <= t_max)
    idx_found_t = np.intersect1d(idx_found_t1, idx_found_t2)
    data_v = data_v[idx_found_t, :]

    idx_tv_ = np.where(data_v[:, -1] == id_tv)
    idx_tv = idx_tv_[0]
    data_tv = data_v[idx_tv, :]

    idx_ov = np.setdiff1d(np.arange(0, data_v.shape[0]), idx_tv)
    data_ov = data_v[idx_ov, :]

    return data_v, data_tv, data_ov


# Get middle points from two points (track: left-right)
def get_mid_pnts_lr(pnts_l, pnts_r, num_intp):
    # pnts_l, pnts_r: (ndarray) points of left & right side (dim = N x 2)
    # num_intp: interpolation number

    pnts_l = make_numpy_array(pnts_l, keep_1dim=False)
    pnts_r = make_numpy_array(pnts_r, keep_1dim=False)

    pnt_x_l_tmp, pnt_y_l_tmp = pnts_l[:, 0].reshape(-1), pnts_l[:, 1].reshape(-1)
    pnt_x_r_tmp, pnt_y_r_tmp = pnts_r[:, 0].reshape(-1), pnts_r[:, 1].reshape(-1)

    do_flip = 0
    if (pnt_x_l_tmp[1] - pnt_x_l_tmp[0]) < 0:
        pnt_x_l_tmp, pnt_y_l_tmp = np.flip(pnt_x_l_tmp, axis=0), np.flip(pnt_y_l_tmp, axis=0)
        do_flip = 1

    if (pnt_x_r_tmp[1] - pnt_x_r_tmp[0]) < 0:
        pnt_x_r_tmp, pnt_y_r_tmp = np.flip(pnt_x_r_tmp, axis=0), np.flip(pnt_y_r_tmp, axis=0)
        do_flip = 1

    x_l_range = np.linspace(min(pnts_l[:, 0]), max(pnts_l[:, 0]), num=num_intp)
    x_r_range = np.linspace(min(pnts_r[:, 0]), max(pnts_r[:, 0]), num=num_intp)
    y_l_intp = np.interp(x_l_range, pnt_x_l_tmp, pnt_y_l_tmp)
    y_r_intp = np.interp(x_r_range, pnt_x_r_tmp, pnt_y_r_tmp)

    pnts_l_intp = np.zeros((num_intp, 2), dtype=np.float32)
    pnts_r_intp = np.zeros((num_intp, 2), dtype=np.float32)
    pnts_l_intp[:, 0], pnts_l_intp[:, 1] = x_l_range, y_l_intp
    pnts_r_intp[:, 0], pnts_r_intp[:, 1] = x_r_range, y_r_intp
    pnts_c_intp = (pnts_l_intp + pnts_r_intp) / 2

    if do_flip == 1:
        pnts_c_intp = np.flip(pnts_c_intp, axis=0)

    return pnts_c_intp


# Get indexes of segment and lane
def get_index_seg_and_lane(pos_i, pnts_poly_track):
    # pos_i: (ndarray) point (dim = 2)

    pos_i = make_numpy_array(pos_i, keep_1dim=True)

    # Check on which track this car is
    num_check = 3
    curseg = -1 * np.ones((num_check, ), dtype=np.int32)
    curlane = -1 * np.ones((num_check, ), dtype=np.int32)
    cnt_check = 0
    for segidx in range(0, len(pnts_poly_track)):
        pnts_poly_seg = pnts_poly_track[segidx]

        for laneidx in range(0, len(pnts_poly_seg)):
            pnts_poly_lane = pnts_poly_seg[laneidx]

            point_is_in_hull = inpolygon(pos_i[0], pos_i[1], pnts_poly_lane[:, 0], pnts_poly_lane[:, 1])
            if point_is_in_hull:
                cnt_check = cnt_check + 1
                curseg[cnt_check - 1] = int(segidx)
                curlane[cnt_check - 1] = int(laneidx)

                if cnt_check >= num_check:
                    break
    if cnt_check == 0:
        curseg, curlane = np.array([-1], dtype=np.int32), np.array([-1], dtype=np.int32)
    else:
        curseg, curlane = curseg[0:cnt_check], curlane[0:cnt_check]

    return curseg, curlane


# Get indexes of segment and lane when point is outside of track
def get_index_seg_and_lane_outside(pos_i, pnts_poly_track):
    # pos_i: (ndarray) point (dim = 2)

    pos_i = make_numpy_array(pos_i, keep_1dim=True)

    # Check on which track this car is
    curseg, curlane = np.array([-1], dtype=np.int32), np.array([-1], dtype=np.int32)
    dist_cur = 100000
    for segidx in range(0, len(pnts_poly_track)):
        pnts_poly_seg = pnts_poly_track[segidx]

        for laneidx in range(0, len(pnts_poly_seg)):
            pnts_poly_lane = pnts_poly_seg[laneidx]

            pnt_mean = np.mean(pnts_poly_lane[:, 0:2], axis=0)
            pnt_mean = pnt_mean.reshape(-1)

            vec_i2mean = np.array([pos_i[0] - pnt_mean[0], pos_i[1] - pnt_mean[1]], dtype=np.float32)
            dist_i2mean = norm(vec_i2mean)

            if dist_i2mean < dist_cur:
                dist_cur = dist_i2mean
                curseg[0] = segidx
                curlane[0] = laneidx

    return curseg, curlane


# Get selected vehicles
def get_selected_vehicles(data_v, id_sel):
    # data_v: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    # id_sel: (ndarray) selected indexes

    data_v = make_numpy_array(data_v, keep_1dim=False)

    if ~isinstance(id_sel, np.ndarray):
        id_sel = np.array(id_sel, dtype=np.int32)

    if len(data_v) > 0:
        dim_data = data_v.shape[1]

        id_sel_c = []
        if len(id_sel) > 0:
            idx_sel_ = np.where(id_sel != -1)
            idx_sel = idx_sel_[0]
            if len(idx_sel) > 0:
                id_sel_c = id_sel[idx_sel]

        data_vehicle_out = np.zeros((200000, dim_data), dtype=np.float32)
        cnt_data_vehicle_out = 0
        if len(id_sel_c) > 0:
            id_data_tmp = data_v[:, -1]
            id_data_tmp = id_data_tmp.astype(dtype=np.int32)

            for nidx_d in range(0, id_sel_c.shape[0]):
                id_sel_c_sel = id_sel_c[nidx_d]
                idx_sel_2_ = np.where(id_data_tmp == id_sel_c_sel)
                idx_sel_2 = idx_sel_2_[0]

                if len(idx_sel_2) > 0:
                    idx_update_tmp = np.arange(cnt_data_vehicle_out, cnt_data_vehicle_out + len(idx_sel_2))
                    data_vehicle_out[idx_update_tmp, :] = data_v[idx_sel_2, :]
                    cnt_data_vehicle_out = cnt_data_vehicle_out + len(idx_sel_2)

        if cnt_data_vehicle_out == 0:
            data_vehicle_out = []
        else:
            data_vehicle_out = data_vehicle_out[np.arange(0, cnt_data_vehicle_out), :]
    else:
        data_vehicle_out = []

    return data_vehicle_out


# Get selected vehicles (near)
def get_selected_vehicles_near(data_v, id_near):
    # data_v: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    # id_near: (ndarray) selected indexes [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]

    data_v = make_numpy_array(data_v, keep_1dim=False)
    id_near = make_numpy_array(id_near, keep_1dim=True)

    if len(data_v) > 0:
        dim_data = data_v.shape[1]

        data_vehicle_out = np.zeros((100000, dim_data), dtype=np.float32)
        cnt_data_vehicle_out = 0
        data_vehicle_out_list = []
        if len(id_near) > 0:
            id_data_tmp = data_v[:, -1]
            id_data_tmp = id_data_tmp.astype(dtype=np.int32)

            for nidx_d in range(0, id_near.shape[0]):
                id_near_sel = id_near[nidx_d]
                if id_near_sel == -1:
                    data_vehicle_out_tmp = -1
                else:
                    idx_near_sel_2_ = np.where(id_data_tmp == id_near_sel)
                    idx_near_sel_2 = idx_near_sel_2_[0]

                    if len(idx_near_sel_2) > 0:
                        idx_update_tmp = np.arange(cnt_data_vehicle_out, cnt_data_vehicle_out + len(idx_near_sel_2))
                        data_vehicle_out[idx_update_tmp, :] = data_v[idx_near_sel_2, :]
                        cnt_data_vehicle_out = cnt_data_vehicle_out + len(idx_near_sel_2)

                        data_vehicle_out_tmp = data_v[idx_near_sel_2, :]
                    else:
                        data_vehicle_out_tmp = -1
                data_vehicle_out_list.append(data_vehicle_out_tmp)

        if cnt_data_vehicle_out == 0:
            data_vehicle_out = []
        else:
            data_vehicle_out = data_vehicle_out[np.arange(0, cnt_data_vehicle_out), :]
    else:
        data_vehicle_out, data_vehicle_out_list = [], []

    return data_vehicle_out, data_vehicle_out_list


# Get vehicle-trajectory
def get_vehicle_trajectory(data_v, time_in, horizon, handle_remain=0):
    # data_v: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    # time_in: (scalar) time
    # horizon: (scalar) horizon or length of trajectory
    # handle_remain: (scalar) method to handle remain values

    data_v = make_numpy_array(data_v, keep_1dim=False)

    if len(data_v) > 0:
        id_unique_ = data_v[:, -1]
        id_unique = np.unique(id_unique_, axis=0)
        id_unique = id_unique.astype(dtype=np.int32)

        size_out = np.zeros((id_unique.shape[0], 2), dtype=np.float32)
        traj_out = []
        for nidx_i in range(0, id_unique.shape[0]):
            idx_sel_1_ = np.where(data_v[:, -1] == id_unique[nidx_i])
            idx_sel_1 = idx_sel_1_[0]

            data_vehicle_per_id = data_v[idx_sel_1, :]
            size_out[nidx_i, :] = [data_v[idx_sel_1[0], 6], data_v[idx_sel_1[0], 5]]

            traj_out_tmp = np.zeros((horizon + 1, 3), dtype=np.float32)
            for nidx_t in range(time_in, time_in + horizon + 1):
                idx_sel_2_ = np.where(data_vehicle_per_id[:, 0] == nidx_t)
                idx_sel_2 = idx_sel_2_[0]

                if len(idx_sel_2) > 0:
                    data_vehicle_sel = data_vehicle_per_id[idx_sel_2[0], :]
                    traj_out_tmp[nidx_t - time_in, :] = data_vehicle_sel[1:4]
                else:
                    if handle_remain == 0:
                        traj_out_tmp[nidx_t - time_in, :] = [np.nan, np.nan, np.nan]
                    elif handle_remain == 1:
                        traj_out_tmp[nidx_t - time_in, :] = traj_out_tmp[nidx_t - time_in - 1, :]
                    else:
                        traj_out_tmp[nidx_t - time_in, :] = [traj_out_tmp[0, 0] - 1000.0, traj_out_tmp[0, 1] - 1000.0,
                                                             0.0]

            traj_out.append(traj_out_tmp)
    else:
        id_unique, traj_out, size_out = [], [], []

    return id_unique, traj_out, size_out


# Get vehicle-trajectory (near)
def get_vehicle_trajectory_near(data_v_list, id_near, time_in, horizon, do_reverse=0, handle_remain=1):
    # data_v_list: (list)-> (ndarray) t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    #                     -> (scalar) -1
    #              : [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
    # id_near: (ndarray) id near vehicles [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
    # time_in: (scalar) time
    # horizon: (scalar) horizon or length of trajectory
    # do_reverse: (boolean) find reverse trajectory
    # handle_remain: (scalar) method to handle remain values

    id_near = make_numpy_array(id_near, keep_1dim=True)

    len_list = len(data_v_list)

    traj_out, size_out = [], []

    for nidx_l in range(0, len_list):
        data_vehicle_list_sel = data_v_list[nidx_l]
        data_vehicle_list_sel = make_numpy_array(data_vehicle_list_sel, keep_1dim=False)

        id_near_sel = id_near[nidx_l]
        if id_near_sel == -1:
            size_out_tmp = np.zeros((2,), dtype=np.float32)

            traj_out_tmp = -2000 * np.ones((horizon + 1, 3), dtype=np.float32)
            traj_out_tmp[:, 2] = 0
        else:
            size_out_tmp = np.array([data_vehicle_list_sel[0, 6], data_vehicle_list_sel[0, 5]], dtype=np.float32)

            traj_out_tmp = np.zeros((horizon + 1, 3), dtype=np.float32)

            iter_start = time_in
            if do_reverse == 0:
                iter_end = time_in + horizon + 1
                iter_step = +1
            else:
                iter_end = time_in - horizon - 1
                iter_step = -1

            for nidx_t in range(iter_start, iter_end, iter_step):
                idx_sel_2_ = np.where(data_vehicle_list_sel[:, 0] == nidx_t)
                idx_sel_2 = idx_sel_2_[0]

                if do_reverse == 1:
                    idx_cur_tmp = -1 * (nidx_t - time_in)
                else:
                    idx_cur_tmp = nidx_t - time_in

                if len(idx_sel_2) > 0:
                    traj_out_tmp[idx_cur_tmp, :] = data_vehicle_list_sel[idx_sel_2[0], 1:4]
                else:
                    if handle_remain == 0:
                        traj_out_tmp[idx_cur_tmp, :] = [np.nan, np.nan, np.nan]
                    elif handle_remain == 1:
                        traj_out_tmp[idx_cur_tmp, :] = traj_out_tmp[idx_cur_tmp - 1, :]
                    elif handle_remain == 2:
                        traj_out_tmp[idx_cur_tmp, :] = [traj_out_tmp[0, 0] - 1000.0, traj_out_tmp[0, 1] - 1000.0,
                                                             0.0]
                    else:
                        diff_tmp = traj_out_tmp[idx_cur_tmp - 1, :] - traj_out_tmp[idx_cur_tmp - 2, :]
                        traj_out_tmp[idx_cur_tmp, :] = traj_out_tmp[idx_cur_tmp - 1, :] + diff_tmp

            if do_reverse == 1:
                traj_out_tmp = np.flipud(traj_out_tmp)

        traj_out.append(traj_out_tmp)
        size_out.append(size_out_tmp)

    return traj_out, size_out


# Get vehicle-trajectory (per id)
def get_vehicle_trajectory_per_id(data_v, time_in, id_in, horizon, do_reverse=0, handle_remain=1):
    # data_v: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    # time_in: (scalar) time
    # id_in: (scalar) id
    # horizon: (scalar) horizon or length of trajectory
    # do_reverse: (boolean) find reverse trajectory
    # handle_remain: (scalar) method to handle remain values

    data_v = make_numpy_array(data_v, keep_1dim=False)

    traj_out, size_out = [], []

    if len(data_v) > 0:
        if ~isinstance(data_v, np.ndarray):
            data_v = np.array(data_v)

        idx_sel_1_ = np.where(data_v[:, -1] == id_in)
        idx_sel_1 = idx_sel_1_[0]

        if len(idx_sel_1) > 0:
            data_vehicle_per_id = data_v[idx_sel_1, :]
            size_out = [data_v[idx_sel_1[0], 6], data_v[idx_sel_1[0], 5]]

            traj_out = np.zeros((horizon + 1, 3), dtype=np.float32)

            iter_start = time_in
            if do_reverse == 0:
                iter_end = time_in + horizon + 1
                iter_step = +1
            else:
                iter_end = time_in - horizon - 1
                iter_step = -1

            for nidx_t in range(iter_start, iter_end, iter_step):
                idx_sel_2_ = np.where(data_vehicle_per_id[:, 0] == nidx_t)
                idx_sel_2 = idx_sel_2_[0]

                if do_reverse == 1:
                    idx_cur_tmp = -1 * (nidx_t - time_in)
                else:
                    idx_cur_tmp = nidx_t - time_in

                if len(idx_sel_2) > 0:
                    data_vehicle_sel = data_vehicle_per_id[idx_sel_2[0], :]
                    traj_out[idx_cur_tmp, :] = data_vehicle_sel[1:4]
                else:
                    if handle_remain == 0:
                        traj_out[idx_cur_tmp, :] = [np.nan, np.nan, np.nan]
                    elif handle_remain == 1:
                        traj_out[idx_cur_tmp, :] = traj_out[idx_cur_tmp - 1, :]
                    elif handle_remain == 2:
                        traj_out[idx_cur_tmp, :] = [traj_out[0, 0] - 1000.0, traj_out[0, 1] - 1000.0, 0.0]
                    else:
                        diff_tmp = traj_out[idx_cur_tmp - 1, :] - traj_out[idx_cur_tmp - 2, :]
                        traj_out[idx_cur_tmp, :] = traj_out[idx_cur_tmp - 1, :] + diff_tmp

            if do_reverse == 1:
                traj_out = np.flipud(traj_out)

    return traj_out, size_out


# Get vehicle-vtrajectory (velocity per id)
def get_vehicle_vtrajectory_per_id(data_v, time_in, id_in, horizon, do_reverse=0, handle_remain=1):
    # data_v: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    # time_in: (scalar) time
    # id_in: (scalar) id
    # horizon: (scalar) horizon or length of trajectory
    # do_reverse: (boolean) find reverse trajectory
    # handle_remain: (scalar) method to handle remain values

    data_v = make_numpy_array(data_v, keep_1dim=False)

    vtraj_out = []

    if len(data_v) > 0:
        if ~isinstance(data_v, np.ndarray):
            data_v = np.array(data_v)

        idx_sel_1_ = np.where(data_v[:, -1] == id_in)
        idx_sel_1 = idx_sel_1_[0]

        if len(idx_sel_1) > 0:
            data_vehicle_per_id = data_v[idx_sel_1, :]

            vtraj_out = np.zeros((horizon + 1, ), dtype=np.float32)

            iter_start = time_in
            if do_reverse == 0:
                iter_end = time_in + horizon + 1
                iter_step = +1
            else:
                iter_end = time_in - horizon - 1
                iter_step = -1

            for nidx_t in range(iter_start, iter_end, iter_step):
                idx_sel_2_ = np.where(data_vehicle_per_id[:, 0] == nidx_t)
                idx_sel_2 = idx_sel_2_[0]

                if do_reverse == 1:
                    idx_cur_tmp = -1 * (nidx_t - time_in)
                else:
                    idx_cur_tmp = nidx_t - time_in

                if len(idx_sel_2) > 0:
                    data_vehicle_sel = data_vehicle_per_id[idx_sel_2[0], :]
                    vtraj_out[idx_cur_tmp] = data_vehicle_sel[4]
                else:
                    if handle_remain == 0:
                        vtraj_out[nidx_t - time_in] = np.nan
                    elif handle_remain == 1:
                        vtraj_out[idx_cur_tmp] = vtraj_out[idx_cur_tmp - 1]
                    else:
                        vtraj_out[nidx_t - time_in, :] = 0

            if do_reverse == 1:
                vtraj_out = np.flipud(vtraj_out)

    return vtraj_out


# Check collision
def check_collision(data_tv, data_ov):
    # data_tv: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = 10, width > length) <- target vehicle
    # data_ov: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = N x 10, width > length) <- others

    data_tv = make_numpy_array(data_tv, keep_1dim=True)
    data_tv_r = np.reshape(data_tv[1:3], (1, 2))

    data_ov = make_numpy_array(data_ov, keep_1dim=False)
    num_ov = data_ov.shape[0]

    # Near distance threshold (30, -1)
    if num_ov >= 10:
        dist_r = 12.5
    else:
        dist_r = -1

    if dist_r > 0:
        # Get near distance other vehicle data
        diff_array = np.repeat(data_tv_r, data_ov.shape[0], axis=0) - data_ov[:, 1:3]
        dist_array = np.sqrt(np.sum(diff_array * diff_array, axis=1))

        idx_sel_ = np.where(dist_array <= dist_r)
        idx_sel = idx_sel_[0]
        data_oc_sel = data_ov[idx_sel, :]
    else:
        data_oc_sel = data_ov

    if len(data_oc_sel.shape) == 1:
        data_oc_sel = np.reshape(data_oc_sel, (1, -1))

    # Get pnts of box (target vehicle)
    # pnts_out_ = get_box_pnts(data_t[1], data_t[2], data_t[3], data_t[6], data_t[5])

    nx_col = max(int(data_tv[6]/0.15), 20)
    ny_col = max(int(data_tv[5]/0.15), 10)
    pnts_out_ = get_box_pnts_precise(data_tv[1], data_tv[2], data_tv[3], data_tv[6], data_tv[5], nx=nx_col, ny=ny_col)
    pnts_m_ = get_m_pnts(data_tv[1], data_tv[2], data_tv[3], data_tv[6], nx=nx_col)
    pnts_out = np.concatenate((pnts_out_, pnts_m_), axis=0)

    len_oc = data_oc_sel.shape[0]
    is_collision = 0
    for nidx_d1 in range(0, len_oc):  # For all other vehicles
        # Get pnts of box (other vehicle)
        data_oc_sel_tmp = data_oc_sel[nidx_d1, :]
        pnts_oc_out = get_box_pnts(data_oc_sel_tmp[1], data_oc_sel_tmp[2], data_oc_sel_tmp[3],
                                   data_oc_sel_tmp[6], data_oc_sel_tmp[5])

        is_out = 0
        for nidx_d2 in range(0, pnts_out.shape[0]):  # For all pnts (target vehicle)
            pnts_out_sel = pnts_out[nidx_d2, :]
            is_in = inpolygon(pnts_out_sel[0], pnts_out_sel[1], pnts_oc_out[:, 0], pnts_oc_out[:, 1])
            if is_in == 1:
                is_out = 1
                break

        if is_out == 1:
            is_collision = 1
            break

    return is_collision


# Check collision (near)
def check_collision_near(data_ev, data_ov, time_in):
    # data_ev, data_ov: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    # time_in: (scalar) time

    if len(data_ov) > 0:
        data_ev = make_numpy_array(data_ev, keep_1dim=True)
        data_ov = make_numpy_array(data_ov, keep_1dim=False)

        idx_sel_ = np.where(data_ov[:, 0] == time_in)
        idx_sel = idx_sel_[0]
        if len(idx_sel) > 0:
            data_ov_cur = data_ov[idx_sel, :]
            is_collision = check_collision(data_ev, data_ov_cur)
        else:
            is_collision = 0
    else:
        is_collision = 0

    return is_collision


# Get lane rad
def get_lane_rad(pnt, pnts_poly_track, pnts_lr_border_track):
    # pnt: (ndarray) point in (dim = 2)
    pnt = make_numpy_array(pnt, keep_1dim=True)

    curseg_, curlane_ = get_index_seg_and_lane(pnt[0:2], pnts_poly_track)
    curseg, curlane = curseg_[0], curlane_[0]
    if curseg == -1 or curlane == -1:
        curseg_, curlane_ = get_index_seg_and_lane_outside(pnt[0:2], pnts_poly_track)
        curseg, curlane = curseg_[0], curlane_[0]

    if curseg == -1 or curlane == -1:
        rad_center = 0.0
    else:
        pnts_lr_border_lane = pnts_lr_border_track[curseg][curlane]
        pnts_left = pnts_lr_border_lane[0]
        pnts_right = pnts_lr_border_lane[1]
        pnt_minleft, _ = get_closest_pnt(pnt[0:2], pnts_left)
        pnt_minright, _ = get_closest_pnt(pnt[0:2], pnts_right)
        # pnt_center = (pnt_minleft + pnt_minright) / 2
        vec_l2r = pnt_minright[0:2] - pnt_minleft[0:2]
        rad = math.atan2(vec_l2r[1], vec_l2r[0])
        rad_center = (rad + math.pi / 2)

    return rad_center


# Get lane-angle w.r.t. mtrack
def get_lane_cp_wrt_mtrack(sim_track, pos, seg, lane):
    # pos: (ndarray) x, y (dim = 2)
    # seg, lane: (scalar) indexes of segment & lane

    # Find indexes of seg & lane
    if seg == -1 or lane == -1:
        seg_, lane_ = get_index_seg_and_lane(pos, sim_track.pnts_poly_track)
        seg, lane = seg_[0], lane_[0]

    pnts_c_tmp = sim_track.pnts_m_track[seg][lane]  # [0, :] start --> [end, :] end

    # Find middle points
    if seg < (sim_track.num_seg - 1):
        child_tmp = sim_track.idx_child[seg][lane]
        for nidx_tmp in range(0, len(child_tmp)):
            pnts_c_next_tmp = sim_track.pnts_m_track[seg + 1][sim_track.idx_child[seg][lane][nidx_tmp]]
            # [0, :] start --> [end, :] end
            pnts_c_tmp = np.concatenate((pnts_c_tmp, pnts_c_next_tmp), axis=0)
        pnts_c = pnts_c_tmp
    else:
        pnts_c = pnts_c_tmp

    # Find lane cp w.r.t. middle points
    pnt_c_out, dist_c_out = get_closest_pnt_intp(pos[0:2], pnts_c, num_intp=100)

    return pnt_c_out


# Get lane-angle w.r.t. mtrack
def get_lane_angle_wrt_mtrack(sim_track, pos, seg, lane, delta=4):
    # pos: (ndarray) x, y (dim = 2)
    # seg, lane: (scalar) indexes of segment & lane
    # delta: (scalar) delta timestep

    # Find indexes of seg & lane
    if seg == -1 or lane == -1:
        seg_, lane_ = get_index_seg_and_lane(pos, sim_track.pnts_poly_track)
        seg, lane = seg_[0], lane_[0]

    pnts_c_tmp = sim_track.pnts_m_track[seg][lane]  # [0, :] start --> [end, :] end

    # Find middle points
    if seg < (sim_track.num_seg - 1):
        child_tmp = sim_track.idx_child[seg][lane]
        for nidx_tmp in range(0, len(child_tmp)):
            pnts_c_next_tmp = sim_track.pnts_m_track[seg + 1][sim_track.idx_child[seg][lane][nidx_tmp]]
            # [0, :] start --> [end, :] end
            pnts_c_tmp = np.concatenate((pnts_c_tmp, pnts_c_next_tmp), axis=0)
        pnts_c = pnts_c_tmp
    else:
        pnts_c = pnts_c_tmp

    # Find lane angle w.r.t. middle points
    len_pnts_c = pnts_c.shape[0]
    pos_r = np.reshape(pos, (1, 2))
    diff_tmp = np.tile(pos_r, (len_pnts_c, 1)) - pnts_c[:, 0:2]
    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
    idx_cur = np.argmin(dist_tmp, axis=0)

    pnt_c_cur = pnts_c[idx_cur, 0:2]
    if (idx_cur + delta) <= (len_pnts_c - 1):
        idx_next = idx_cur + delta
    else:
        idx_next = len_pnts_c - 1

    if (idx_cur - delta) >= 0:
        idx_prev = idx_cur - delta
    else:
        idx_prev = 0

    if idx_cur == 0:
        pnt_c_next = pnts_c[idx_next, 0:2]
        angle_c = math.atan2(pnt_c_next[1] - pnt_c_cur[1], pnt_c_next[0] - pnt_c_cur[0])
    elif idx_cur == pnts_c.shape[0]:
        pnt_c_prev = pnts_c[idx_prev, 0:2]
        angle_c = math.atan2(pnt_c_cur[1] - pnt_c_prev[1], pnt_c_cur[0] - pnt_c_prev[0])
    else:
        pnt_c_next = pnts_c[idx_next, 0:2]
        pnt_c_prev = pnts_c[idx_prev, 0:2]
        angle_c_f = math.atan2(pnt_c_next[1] - pnt_c_cur[1], pnt_c_next[0] - pnt_c_cur[0])
        angle_c_b = math.atan2(pnt_c_cur[1] - pnt_c_prev[1], pnt_c_cur[0] - pnt_c_prev[0])

        # if abs(angle_c_f) < 0.01:
        #     angle_c = angle_c_b
        # elif abs(angle_c_b) < 0.01:
        #     angle_c = angle_c_f
        # else:
        #     angle_c = (angle_c_f + angle_c_b) / 2.0

        if abs(angle_c_f) < 0.01:
            angle_c = angle_c_b
        else:
            angle_c = angle_c_f

    return angle_c


# Get feature part
def get_feature_part(sim_track, data_t, use_intp=0):
    # data_t: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    # use_intp: (scalar) whether to use interpolation (pnt_minleft, pnt_minright, pnt_center, lane_angle)

    data_t = make_numpy_array(data_t, keep_1dim=True)

    # Get car-info
    pos_i = data_t[1:4]  # x, y, theta(rad)
    # carlen = data_t[6]
    carwidth = data_t[5]
    seg_t = int(data_t[7])
    lane_t = int(data_t[8])

    # Check on which track this car is
    if seg_t == -1 or lane_t == -1:
        curseg, curlane = get_index_seg_and_lane(pos_i, sim_track.pnts_poly_track)
    else:
        curseg, curlane = np.array([seg_t], dtype=np.int32), np.array([lane_t], dtype=np.int32)

    curseg_sel, curlane_sel = curseg[0], curlane[0]

    # Find feature
    # A. GET LANE DEVIATION DIST
    if curseg_sel == -1 or curlane_sel == -1:
        # Out of track
        # print("Out of track")
        lane_dev_rad, lane_dev_dist, lane_dev_rdist, lane_width = 0, 0, 0.5, 1
        pnt_center = np.array([0, 0], dtype=np.float32)
        rad_center = 0
        pnt_minleft = np.array([0, 0], dtype=np.float32)
        pnt_minright = np.array([0, 0], dtype=np.float32)
    else:
        pnts_lr_border_lane = sim_track.pnts_lr_border_track[curseg_sel][curlane_sel]
        pnts_left = pnts_lr_border_lane[0]
        pnts_right = pnts_lr_border_lane[1]
        if use_intp == 0:
            pnt_minleft, _ = get_closest_pnt(pos_i[0:2], pnts_left)
            pnt_minright, _ = get_closest_pnt(pos_i[0:2], pnts_right)
            vec_l2r = pnt_minright[0:2] - pnt_minleft[0:2]
            rad = math.atan2(vec_l2r[1], vec_l2r[0])
            rad_center = (rad + math.pi / 2)
        else:
            pnt_minleft, _ = get_closest_pnt_intp(pos_i[0:2], pnts_left, num_intp=100)
            pnt_minright, _ = get_closest_pnt_intp(pos_i[0:2], pnts_right, num_intp=100)
            vec_l2r = pnt_minright[0:2] - pnt_minleft[0:2]

            if sim_track.lane_type[curseg_sel][curlane_sel] == "Straight":
                delta_sel = 4  # 4
            else:
                delta_sel = 1

            rad_center = get_lane_angle_wrt_mtrack(sim_track, pos_i[0:2], curseg_sel, curlane_sel, delta=delta_sel)

        lane_dev_rad = pos_i[2] - rad_center
        lane_dev_rad = angle_handle(lane_dev_rad)
        pnt_center = (pnt_minleft + pnt_minright) / 2

        vec_c2i = pos_i[0:2] - pnt_center[0:2]

        lane_width = norm(vec_l2r[0:2])
        lane_dev_dist = norm(vec_c2i[0:2]) * np.sign(vec_c2i[0]*vec_l2r[0] + vec_c2i[1]*vec_l2r[1])
        lane_dev_rdist = norm(pos_i[0:2] - pnt_minright[0:2])
        lane_dev_rdist = lane_dev_rdist - carwidth / 2

    return curseg, curlane, lane_dev_rad, lane_dev_dist, lane_dev_rdist, lane_width, \
           pnt_center, rad_center, pnt_minleft, pnt_minright


# Get feature
def get_feature(sim_track, data_ev, data_ov, use_intp=0):
    # data_ev: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    # data_ov: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = N x 10, width > length)
    # th_lane_connected_lower = 2.5: (scalar) threshold for whether two lanes are connected (lower)
    # th_lane_connected_upper = 3.8: (scalar) threshold for whether two lanes are connected (upper)
    # use_intp: (scalar) whether to use interpolation (pnt_minleft, pnt_minright, pnt_center, lane_angle)

    data_ev = make_numpy_array(data_ev, keep_1dim=True)
    data_ov = make_numpy_array(data_ov, keep_1dim=False)

    # Get car-info
    pos_i = data_ev[1:4]  # x, y, theta(rad)
    carlen = data_ev[6]
    carwidth = data_ev[5]

    num_car = data_ov.shape[0]

    # Check on which track this car is
    if int(data_ev[7]) == -1 or int(data_ev[8]) == -1:
        curseg, curlane = get_index_seg_and_lane(pos_i, sim_track.pnts_poly_track)
        data_ev[7], data_ev[8] = curseg[0], curlane[0]

    # FIND FEATURE
    # A. GET LANE DEVIATION DIST
    curseg, curlane, lane_dev_rad, lane_dev_dist, lane_dev_rdist, lane_width, pnt_center, rad_center, \
        pnt_minleft, pnt_minright = get_feature_part(sim_track, data_ev, use_intp)
    seg_i, lane_i = curseg[0], curlane[0]

    lane_dev_dist_scaled = lane_dev_dist / lane_width
    lane_dev_rdist_scaled = lane_dev_rdist / lane_width

    # B. FRONTAL AND REARWARD DISTANCES (LEFT, CENTER, RIGHT)
    max_dist = 40  # maximum-distance
    id_lf, id_lr, id_rf, id_rr, id_cf, id_cr, idx_lf, idx_lr, idx_cf, idx_cr, idx_rf, idx_rr, left_frontal, \
    left_rearward, right_frontal, right_rearward, center_frontal, center_rearward, left_frontal_pos, \
    left_rearward_pos, right_frontal_pos, right_rearward_pos, center_frontal_pos, center_rearward_pos = \
        get_feature_sub(sim_track, pos_i, seg_i, lane_i, data_ov, num_car, max_dist)

    # Set lr-dist
    idx_array = np.array([idx_lf, idx_lr, idx_cf, idx_cr, idx_rf, idx_rr], dtype=np.int32)
    lr_dist_array = np.ones((6,), dtype=np.float32)
    for nidx_sr in range(0, idx_array.shape[0]):
        idx_tmp = idx_array[nidx_sr]
        if idx_tmp >= 0:
            _, _, _, _, ld_rdist_tmp, lw_tmp, _, _, _, _ = \
                get_feature_part(sim_track, data_ov[idx_tmp, :], use_intp=use_intp)
            lr_dist_array[nidx_sr] = (ld_rdist_tmp - data_ov[idx_tmp, 5] / 2) / lw_tmp

    left_frontal_scaled, left_rearward_scaled = left_frontal / lane_width, left_rearward / lane_width
    center_frontal_scaled, center_rearward_scaled = center_frontal / lane_width, center_rearward / lane_width
    right_frontal_scaled, right_rearward_scaled = right_frontal / lane_width, right_rearward / lane_width

    f_out = np.array([lane_dev_rad, lane_dev_dist, lane_dev_rdist, lane_width,
                      lane_dev_dist_scaled, lane_dev_rdist_scaled,
                      left_frontal_scaled, left_rearward_scaled, center_frontal_scaled, center_rearward_scaled,
                      right_frontal_scaled, right_rearward_scaled,
                      lr_dist_array[0], lr_dist_array[1], lr_dist_array[2], lr_dist_array[3], lr_dist_array[4],
                      lr_dist_array[5], data_ev[4]], dtype=np.float32)

    id_lr = -1 if id_lf == id_lr else id_lr
    id_rr = -1 if id_rf == id_rr else id_rr
    id_cr = -1 if id_cf == id_cr else id_cr

    id_near = np.array([id_lf, id_lr, id_rf, id_rr, id_cf, id_cr], dtype=np.int32)

    pnts_feature_ev = np.zeros((3, 2), dtype=np.float32)
    pnts_feature_ev[0, :] = pnt_center
    pnts_feature_ev[1, :] = pnt_minleft
    pnts_feature_ev[2, :] = pnt_minright

    pnts_feature_ov = np.zeros((6, 2), dtype=np.float32)
    pnts_feature_ov[0, :] = left_frontal_pos
    pnts_feature_ov[1, :] = left_rearward_pos
    pnts_feature_ov[2, :] = center_frontal_pos
    pnts_feature_ov[3, :] = center_rearward_pos
    pnts_feature_ov[4, :] = right_frontal_pos
    pnts_feature_ov[5, :] = right_rearward_pos

    return f_out, id_near, pnts_feature_ev, pnts_feature_ov, rad_center, center_frontal


# Get feature (sub)
def get_feature_sub(sim_track, pos_i, seg_i, lane_i, data_ov, num_car, max_dist):
    th_lane_connected_lower = sim_track.th_lane_connected_lower
    th_lane_connected_upper = sim_track.th_lane_connected_upper

    left_frontal, left_rearward = max_dist, max_dist
    left_frontal_pos = np.array([0, 0], dtype=np.float32)
    left_rearward_pos = np.array([0, 0], dtype=np.float32)

    right_frontal, right_rearward = max_dist, max_dist
    right_frontal_pos = np.array([0, 0], dtype=np.float32)
    right_rearward_pos = np.array([0, 0], dtype=np.float32)

    center_frontal, center_rearward = max_dist, max_dist
    center_frontal_pos = np.array([0, 0], dtype=np.float32)
    center_rearward_pos = np.array([0, 0], dtype=np.float32)

    # Near vehicle id
    id_lf, id_lr, id_rf, id_rr, id_cf, id_cr = -1, -1, -1, -1, -1, -1

    # Near vehicle indexes
    idx_lf, idx_lr, idx_rf, idx_rr, idx_cf, idx_cr = -1, -1, -1, -1, -1, -1

    if seg_i == -1 or lane_i == -1:
        # Out of track
        # print("Out of track")
        pass
    else:
        track_dir_cur = sim_track.lane_dir[seg_i][lane_i]

        # B-0. FIND 'cpnt_c'
        pnts_lr_border_lane_c = sim_track.pnts_lr_border_track[seg_i][lane_i]
        pnts_left_c = pnts_lr_border_lane_c[0]
        pnts_right_c = pnts_lr_border_lane_c[1]

        lpnt_c, _ = get_closest_pnt(pos_i[0:2], pnts_left_c)
        rpnt_c, _ = get_closest_pnt(pos_i[0:2], pnts_right_c)
        cpnt_c = (lpnt_c + rpnt_c) / 2

        # B-1. LEFT FRONTAL AND REARWARD DISTANCE
        cond_not_leftmost = (lane_i > 0) if track_dir_cur == +1 else (lane_i < (sim_track.num_lane[seg_i] - 1))
        if cond_not_leftmost:  # NOT LEFTMOST LANE
            lane_i_left = (lane_i - 1) if track_dir_cur == +1 else (lane_i + 1)
            pnts_lr_border_lane_l = sim_track.pnts_lr_border_track[seg_i][lane_i_left]
            pnts_left_l = pnts_lr_border_lane_l[0]
            pnts_right_l = pnts_lr_border_lane_l[1]

            lpnt_l, _ = get_closest_pnt(pos_i[0:2], pnts_left_l)
            rpnt_l, _ = get_closest_pnt(pos_i[0:2], pnts_right_l)

            cpnt_l = (lpnt_l + rpnt_l) / 2
            vec_l2c = cpnt_c[0:2] - cpnt_l[0:2]
            norm_vec_l2c = norm(vec_l2c)

            if (norm_vec_l2c < th_lane_connected_upper) and (norm_vec_l2c > th_lane_connected_lower):
                # CHECK IF LEFT LANE IS CONNECTED
                vec_l2r = rpnt_l - lpnt_l
                rad = math.atan2(vec_l2r[1], vec_l2r[0]) + math.pi / 2
                forwardvec = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)

                for j in range(0, num_car):
                    pos_j = data_ov[j, 1:4]  # x, y, theta
                    rx_j = data_ov[j, 6] / 2
                    lane_j = data_ov[j, 8]
                    id_j = data_ov[j, -1]

                    if abs(lane_j - lane_i_left) < 0.5:  # FOR THE LANE LEFT
                        vec_c2j = pos_j[0:2] - cpnt_l
                        dist_c2j = norm(vec_c2j)
                        dot_tmp = vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]
                        if dot_tmp > 0:  # FRONTAL
                            if dist_c2j < left_frontal:
                                alpha_tmp = (vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]) - rx_j
                                left_frontal_pos = cpnt_l[0:2] + alpha_tmp * forwardvec[0:2]
                                left_frontal = norm(cpnt_l[0:2] - left_frontal_pos[0:2]) * np.sign(alpha_tmp)
                                left_frontal = min(left_frontal, max_dist)
                                id_lf = id_j
                                idx_lf = j

                        else:  # REARWARD
                            if dist_c2j < left_rearward:
                                alpha_tmp = -(vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]) - rx_j
                                left_rearward_pos = cpnt_l[0:2] + alpha_tmp * (-forwardvec[0:2])
                                left_rearward = norm(cpnt_l[0:2] - left_rearward_pos[0:2]) * np.sign(alpha_tmp)
                                left_rearward = min(left_rearward, max_dist)
                                id_lr = id_j
                                idx_lr = j
            else:
                left_frontal, left_rearward = 0, 0
        else:
            left_frontal, left_rearward = 0, 0

        # B-2. RIGHT FRONTAL AND REARWARD DISTANCE
        cond_not_rightmost = (lane_i < (sim_track.num_lane[seg_i] - 1)) if track_dir_cur == +1 else \
            (lane_i > 0)
        if cond_not_rightmost:  # NOT RIGHTMOST LANE
            lane_i_right = (lane_i + 1) if track_dir_cur == +1 else (lane_i - 1)
            pnts_lr_border_lane_r = sim_track.pnts_lr_border_track[seg_i][lane_i_right]
            pnts_left_r = pnts_lr_border_lane_r[0]
            pnts_right_r = pnts_lr_border_lane_r[1]

            lpnt_r, _ = get_closest_pnt(pos_i[0:2], pnts_left_r)
            rpnt_r, _ = get_closest_pnt(pos_i[0:2], pnts_right_r)
            cpnt_r = (lpnt_r + rpnt_r) / 2
            vec_r2c = cpnt_c[0:2] - cpnt_r[0:2]
            norm_vec_r2c = norm(vec_r2c)

            if (norm_vec_r2c < th_lane_connected_upper) and (norm_vec_r2c > th_lane_connected_lower):
                # CHECK IF RIGHT LANE IS CONNECTED
                vec_l2r = rpnt_r[0:2] - lpnt_r[0:2]
                rad = math.atan2(vec_l2r[1], vec_l2r[0]) + math.pi / 2
                forwardvec = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)
                for j in range(0, num_car):  # FOR ALL OTHER CARS
                    pos_j = data_ov[j, 1:4]  # x, y, theta
                    rx_j = data_ov[j, 6] / 2
                    lane_j = data_ov[j, 8]
                    id_j = data_ov[j, -1]

                    if abs(lane_j - lane_i_right) < 0.5:  # FOR THE LANE RIGHT
                        vec_c2j = pos_j[0:2] - cpnt_r[0:2]
                        dist_c2j = norm(vec_c2j)
                        dot_tmp = vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]
                        if dot_tmp > 0:  # FRONTAL
                            if dist_c2j < right_frontal:
                                alpha_tmp = vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1] - rx_j
                                right_frontal_pos = cpnt_r[0:2] + alpha_tmp * forwardvec[0:2]
                                right_frontal = norm(cpnt_r[0:2] - right_frontal_pos[0:2]) * np.sign(alpha_tmp)
                                right_frontal = min(right_frontal, max_dist)
                                id_rf = id_j
                                idx_rf = j

                        else:  # REARWARD
                            if dist_c2j < right_rearward:
                                alpha_tmp = -(vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]) - rx_j
                                right_rearward_pos = cpnt_r[0:2] + alpha_tmp * (-forwardvec[0:2])
                                right_rearward = norm(cpnt_r[0:2] - right_rearward_pos[0:2]) * np.sign(alpha_tmp)
                                right_rearward = min(right_rearward, max_dist)
                                id_rr = id_j
                                idx_rr = j
            else:
                right_frontal, right_rearward = 0, 0
        else:
            right_frontal, right_rearward = 0, 0

        # B-3. CENTER FRONTAL AND REARWARD DISTANCE
        vec_l2r = rpnt_c[0:2] - lpnt_c[0:2]
        rad = math.atan2(vec_l2r[1], vec_l2r[0]) + math.pi / 2
        forwardvec = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)

        for j in range(0, num_car):  # FOR ALL OTHER CARS
            pos_j = data_ov[j, 1:4]  # x, y, theta
            rx_j = data_ov[j, 6] / 2
            lane_j = data_ov[j, 8]
            id_j = data_ov[j, -1]

            if abs(lane_j - lane_i) < 0.5:  # FOR THE SAME LANE
                vec_c2j = pos_j[0:2] - cpnt_c[0:2]
                dist_c2j = norm(vec_c2j)
                dot_tmp = vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]
                if dot_tmp > 0:  # FRONTAL
                    if dist_c2j < center_frontal:
                        alpha_tmp = vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1] - rx_j
                        center_frontal_pos = cpnt_c[0:2] + alpha_tmp * forwardvec[0:2]
                        center_frontal = norm(cpnt_c[0:2] - center_frontal_pos[0:2]) * np.sign(alpha_tmp)
                        center_frontal = min(center_frontal, max_dist)
                        id_cf = id_j
                        idx_cf = j

                else:  # REARWARD
                    if dist_c2j < center_rearward:
                        alpha_tmp = -(vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]) - rx_j
                        center_rearward_pos = cpnt_c[0:2] + alpha_tmp * (-forwardvec[0:2])
                        center_rearward = norm(cpnt_c[0:2] - center_rearward_pos[0:2]) * np.sign(alpha_tmp)
                        center_rearward = min(center_rearward, max_dist)
                        id_cr = id_j
                        idx_cr = j

    return id_lf, id_lr, id_rf, id_rr, id_cf, id_cr, idx_lf, idx_lr, idx_cf, idx_cr, idx_rf, idx_rr, left_frontal, \
           left_rearward, right_frontal, right_rearward, center_frontal, center_rearward, left_frontal_pos, \
           left_rearward_pos, right_frontal_pos, right_rearward_pos, center_frontal_pos, center_rearward_pos


# Get control set (naive)
def get_control_set_naive(u0_set, u1_set):
    # u0_set, u1_set: (ndarray) set of controls for each dimension (dim = N)

    u0_set = make_numpy_array(u0_set, keep_1dim=True)
    u1_set = make_numpy_array(u1_set, keep_1dim=True)

    len_lv = u0_set.shape[0]
    len_av = u1_set.shape[0]

    u_set = np.zeros((len_lv * len_av, 2), dtype=np.float32)

    lv_range_ext = np.repeat(u0_set, len_av)
    av_range_ext = np.tile(u1_set, len_lv)

    u_set[:, 0] = lv_range_ext
    u_set[:, 1] = av_range_ext

    return u_set


# Get dist to goal
def get_dist2goal(pnt_t, seg_t, lane_t, indexes_goal, pnts_goal):
    # pnt_t: (ndarray) target point (dim = 2)
    # seg_t, lane_t: (scalar) target seg & lane indexes
    # indexes_goal: (ndarray) goal indexes - seg, lane (dim = N x 2)
    # pnts_goal: (ndarray) goal points (dim = N x 2)

    pnt_t = make_numpy_array(pnt_t, keep_1dim=True)

    max_dist = 10000

    num_goal_pnts = indexes_goal.shape[0]

    dist2goal_array = max_dist * np.ones((num_goal_pnts,), dtype=np.float32)
    reach_goal_array = np.zeros((num_goal_pnts,), dtype=np.int32)

    if int(seg_t) == -1 or int(lane_t) == -1:
        # print("Invalid segment & lane indexes")
        pass
    else:
        for nidx_d in range(0, num_goal_pnts):
            index_goal_sel = indexes_goal[nidx_d, :]
            pnt_goal_sel = pnts_goal[nidx_d, :]
            if seg_t == index_goal_sel[0] and lane_t == index_goal_sel[1]:
                diff_sel = pnt_t[0:2] - pnt_goal_sel[0:2]
                dist_sel = norm(diff_sel)

                dist2goal_array[nidx_d] = min(dist_sel, max_dist)
                if dist_sel < 6:
                    reach_goal_array[nidx_d] = 1

    return dist2goal_array, reach_goal_array


# Encode trajectory
def encode_trajectory(traj_in, traj_type, pnts_poly_track, pnts_lr_border_track, is_track_simple=0):
    # traj_in: (ndarray) trajectory (dim = (H+1) x 3)
    # traj_type: (boolean)  0 --> prev: (1):t-H --> (end):t
    #                       1 --> post: (1):t --> (end):t+H
    # is_track_simple: (boolean) whether track is simple (highD)

    traj_in = make_numpy_array(traj_in, keep_1dim=False)
    h = traj_in.shape[0] - 1

    traj_encoded = np.zeros((3 * h, ), dtype=np.float32)

    for nidx_h in range(0, h):
        if traj_type == 0:  # prev
            t_cur = h - nidx_h
            pnt_cur = traj_in[t_cur, 0:3]
            pnt_next = traj_in[t_cur - 1, 0:3]
        else:  # post
            t_cur = nidx_h
            pnt_cur = traj_in[t_cur, 0:3]
            pnt_next = traj_in[t_cur + 1, 0:3]

        if is_track_simple == 1:
            if abs(pnt_cur[2]) > math.pi / 2:
                rad_center_cur = math.pi
            else:
                rad_center_cur = 0.0
        else:
            rad_center_cur = get_lane_rad(pnt_cur, pnts_poly_track, pnts_lr_border_track)

        diff_vec = pnt_next[0:2] - pnt_cur[0:2]
        dist_vec = norm(diff_vec)
        angle_diff = math.atan2(diff_vec[1], diff_vec[0]) - rad_center_cur
        dist_horiz = dist_vec * math.cos(angle_diff)
        dist_vertical = dist_vec * math.sin(angle_diff)
        diff_rad = pnt_next[2] - pnt_cur[2]

        idx_update = np.arange(3 * nidx_h, 3 * nidx_h + 3)
        traj_encoded[idx_update] = [dist_horiz, dist_vertical, diff_rad]

    return traj_encoded


# Decode trajectory
def decode_trajectory(pos_in, val_in, h, traj_type, pnts_poly_track, pnts_lr_border_track, is_track_simple=0):
    # pos_in: (ndarray) start pose (dim = 3)
    # val_in: (ndarray) encoded val (dim = 3 * h)
    # h: (scalar) horizon-length
    # traj_type: (boolean)  0 --> prev: (1):t-H --> (end):t
    #                       1 --> post: (1):t --> (end):t+H
    # is_track_simple: (boolean) whether track is simple (highD)

    pos_in = make_numpy_array(pos_in, keep_1dim=True)
    val_in = make_numpy_array(val_in, keep_1dim=True)

    traj_decoded = np.zeros((h + 1, 3), dtype=np.float32)

    pos_init = np.copy(pos_in)
    if is_track_simple == 1:
        if abs(pos_init[2]) > math.pi / 2:
            lanerad_init = math.pi
        else:
            lanerad_init = 0.0
    else:
        lanerad_init = get_lane_rad(pos_init, pnts_poly_track, pnts_lr_border_track)
    devrad_init = pos_in[2] - lanerad_init

    traj_decoded[0, :] = pos_init
    pos_cur, devrad_cur, lanerad_cur = np.copy(pos_init), np.copy(devrad_init), np.copy(lanerad_init)
    pos_cur_tmp, devrad_cur_tmp, lanerad_cur_tmp = np.copy(pos_cur), np.copy(devrad_cur), np.copy(lanerad_cur)

    for nidx_h in range(0, h):
        pos_next = np.zeros((3, ), dtype=np.float32)

        idx_h = np.arange(3 * nidx_h, 3 * nidx_h + 3)
        diff2next = val_in[idx_h]
        dist2next = norm(diff2next[0:2])

        # Find next position (x, y)
        angle_tmp1 = math.atan2(diff2next[1], diff2next[0])
        angle_tmp2 = lanerad_cur_tmp + angle_tmp1
        dx_tmp = dist2next * math.cos(angle_tmp2)
        dy_tmp = dist2next * math.sin(angle_tmp2)
        pos_next[0:2] = pos_cur_tmp[0:2] + np.array([dx_tmp, dy_tmp], dtype=np.float32)

        # Find next angle (theta)
        devrad_next = devrad_cur_tmp + diff2next[2]
        if is_track_simple == 1:
            lanerad_next = lanerad_init
        else:
            lanerad_next = get_lane_rad(pos_next[0:2], pnts_poly_track, pnts_lr_border_track)
        pos_next[2] = devrad_next + lanerad_next
        traj_decoded[nidx_h + 1, :] = pos_next

        # Update pos_cur_tmp, devrad_cur_tmp, lanerad_cur_tmp
        pos_cur_tmp, devrad_cur_tmp, lanerad_cur_tmp = np.copy(pos_next), np.copy(devrad_next), np.copy(lanerad_next)

    if traj_type == 0:
        traj_decoded = np.flipud(traj_decoded)

    return traj_decoded


# Get current info (multi)
def get_current_info_multi(t_cur, id_ev, data_ev, data_ov, id_near_cur, num_near, h_post, h_prev, idx_y_sp, idx_x_sp,
                           dim_p, sim_track, is_track_simple):

    if h_post > 0:
        y_ev, _ = get_vehicle_trajectory_per_id(data_ev, t_cur, id_ev, h_post, do_reverse=0, handle_remain=1)
        vy_ev = get_vehicle_vtrajectory_per_id(data_ev, t_cur, id_ev, h_post, do_reverse=0, handle_remain=1)
        y_ev_enc = encode_trajectory(y_ev, 1, sim_track.pnts_poly_track, sim_track.pnts_lr_border_track,
                                     is_track_simple)
    else:
        y_ev, vy_ev, y_ev_enc = [], [], []

    if h_prev > 0:
        x_ev, _ = get_vehicle_trajectory_per_id(data_ev, t_cur, id_ev, h_prev, do_reverse=1, handle_remain=1)
        x_ev_enc = encode_trajectory(x_ev, 0, sim_track.pnts_poly_track, sim_track.pnts_lr_border_track,
                                     is_track_simple)

        x_ov_near_enc = np.zeros((1, num_near, dim_p * h_prev), dtype=np.float32)
    else:
        x_ev, x_ev_enc, x_ov_near_enc = [], [], []

    if len(idx_y_sp) > 0 and h_post > 0:
        y_ev_sp = y_ev[idx_y_sp, :]
        y_ev_enc_sp = encode_trajectory(y_ev_sp, 1, sim_track.pnts_poly_track, sim_track.pnts_lr_border_track,
                                        is_track_simple)
    else:
        y_ev_sp, y_ev_enc_sp = [], []

    if len(idx_x_sp) > 0 and h_prev > 0:
        x_ev_sp = x_ev[idx_x_sp, :]
        x_ev_enc_sp = encode_trajectory(x_ev_sp, 0, sim_track.pnts_poly_track, sim_track.pnts_lr_border_track,
                                        is_track_simple)

        x_ov_near_sp_enc = np.zeros((1, num_near, dim_p * len(idx_x_sp)), dtype=np.float32)
    else:
        x_ev_sp, x_ev_enc_sp, x_ov_near_sp_enc = [], [], []

    y_ov_near, vy_ov_near, x_ov_near = [], [], []
    y_ov_near_sp, x_ov_near_sp = [], []
    s_ov_near_cur = np.zeros((len(id_near_cur), 4), dtype=np.float32)
    size_ov_near_cur = np.zeros((len(id_near_cur), 2), dtype=np.float32)

    for nidx_near in range(0, len(id_near_cur)):
        id_near_sel = id_near_cur[nidx_near]
        if id_near_sel == -1:
            y_ov_tmp = -1000 * np.ones((h_post + 1, 3), dtype=np.float32)
            x_ov_tmp = -1000 * np.ones((h_prev + 1, 3), dtype=np.float32)
            vy_ov_tmp = np.zeros((h_prev + 1,), dtype=np.float32)
            y_ov_sp_tmp = y_ov_tmp[idx_y_sp, :]
            x_ov_sp_tmp = x_ov_tmp[idx_x_sp, :]
        else:
            if h_post > 0:
                y_ov_tmp, size_tmp = get_vehicle_trajectory_per_id(data_ov, t_cur, id_near_sel, h_post, do_reverse=0,
                                                                   handle_remain=1)
                vy_ov_tmp = get_vehicle_vtrajectory_per_id(data_ov, t_cur, id_near_sel, h_post, do_reverse=0,
                                                           handle_remain=1)
                size_ov_near_cur[nidx_near, :] = size_tmp

            if h_prev > 0:
                x_ov_tmp, _ = get_vehicle_trajectory_per_id(data_ov, t_cur, id_near_sel, h_prev, do_reverse=1,
                                                            handle_remain=1)
                x_ov_enc_tmp = encode_trajectory(x_ov_tmp, 0, sim_track.pnts_poly_track, sim_track.pnts_lr_border_track,
                                                 is_track_simple)
                x_ov_near_enc[0, nidx_near, :] = x_ov_enc_tmp

            if len(idx_x_sp) > 0 and h_prev > 0:
                x_ov_sp_tmp = x_ov_tmp[idx_x_sp, :]
                x_ov_sp_enc_tmp = encode_trajectory(x_ov_sp_tmp, 0, sim_track.pnts_poly_track,
                                                    sim_track.pnts_lr_border_track, is_track_simple)
                x_ov_near_sp_enc[0, nidx_near, :] = x_ov_sp_enc_tmp

        # Save
        if h_post > 0:
            s_ov_near_cur[nidx_near, :] = [y_ov_tmp[0, 0], y_ov_tmp[0, 1], y_ov_tmp[0, 2], vy_ov_tmp[0]]
            y_ov_near.append(y_ov_tmp)
            vy_ov_near.append(vy_ov_tmp)

        if h_prev > 0:
            x_ov_near.append(x_ov_tmp)

        if len(idx_y_sp) > 0 and h_post > 0:
            y_ov_near_sp.append(y_ov_sp_tmp)

        if len(idx_x_sp) > 0 and h_prev > 0:
            x_ov_near_sp.append(x_ov_sp_tmp)

    if h_prev > 0:
        x_ov_near_enc = x_ov_near_enc.reshape(1, -1)

    return y_ev, vy_ev, y_ev_sp, x_ev, x_ev_sp, y_ev_enc, x_ev_enc, y_ev_enc_sp, x_ev_enc_sp, s_ov_near_cur, \
           y_ov_near, vy_ov_near, x_ov_near, y_ov_near_sp, x_ov_near_sp, x_ov_near_enc, x_ov_near_sp_enc, \
           size_ov_near_cur
