# SET EGO-VEHICLE FOR SIMULATOR
# System Dynamic: 4-state unicycle dynamics
#   state: [x, y, theta, v]
#   control: [w, a]
#       dot(x) = v * cos(theta)
#       dot(y) = v * sin(theta)
#       dot(theta) = v * kappa_1 * w
#       dot(v) = kappa_2 * a

from __future__ import print_function

import sys
sys.path.insert(0, "../")

from src.utils_sim import *


class SimControl(object):
    def __init__(self, sim_track, dt_in, rx_in, ry_in, kappa, v_ref, v_range):
        self.dim_x, self.dim_u = 4, 2  # Set dimension for state & control

        self.track = sim_track  # track (class)

        self.dt = dt_in
        self.rx, self.ry = rx_in, ry_in
        self.kappa = kappa
        self.v_range = v_range  # [min, max]
        self.v_ref = v_ref

        # (Initial) Ego-state
        self.x_ego_init, self.y_ego_init, self.theta_ego_init, self.v_ego_init = 0.0, 0.0, 0.0, v_ref

        # Ego state
        self.x_ego, self.y_ego, self.theta_ego, self.v_ego = 0.0, 0.0, 0.0, v_ref

        # Ego control
        self.w_ego, self.a_ego = 0.0, 0.0

        # Initial points
        self.pnts_init = []

        # Reward components
        self.r_weight = []
        self.dev_rad_th, self.dist_cf_th, self.w_max = 0.0, 0.0, 0.0

    # Set reward components
    def set_reward_components(self, r_weight, dev_rad_th, dist_cf_th, w_max):
        # Reward weights
        # outside, collision, dev_rad, dev_dist, dist_cf, linear_velocity, angular_velocity, reach_goal (dim = 8)
        self.r_weight = r_weight

        self.dev_rad_th, self.dist_cf_th, self.w_max = dev_rad_th, dist_cf_th, w_max

    # Set track
    def set_track(self, sim_track):
        self.track = sim_track

    # Set initial states
    def set_initial_state(self, data_vehicle_ov, time_in, seg_in, lane_in, margin_rx, margin_ry, vel_rand=False):
        # data_vehicle_ov: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = N x 10)
        # time_in: (scalar) time
        # seg_in, lane_in: (scalar) indexes of seg & lane
        # margin_rx, margin_ry: (scalar) margin
        # vel_rand: (boolean) whether to set random velocity

        if ~isinstance(data_vehicle_ov, np.ndarray):
            data_vehicle_ov = np.array(data_vehicle_ov)

        if seg_in == -1:
            seg_in = 0
        if lane_in == -1:
            lane_in = np.random.randint(3)

        # Select data (w.r.t time)
        idx_sel_ = np.where(data_vehicle_ov[:, 0] == time_in)
        idx_sel = idx_sel_[0]
        data_vehicle_ov_sel = data_vehicle_ov[idx_sel, :]

        # Select data (w.r.t lane)
        idx_sel_2 = []
        idx_sel_c_ = np.where(data_vehicle_ov_sel[:, 8] == lane_in)
        if len(idx_sel_c_) > 0:
            idx_sel_c = idx_sel_c_[0]
            idx_sel_2 = idx_sel_c

        if lane_in > 0:
            idx_sel_l_ = np.where(data_vehicle_ov_sel[:, 8] == (lane_in - 1))
            if len(idx_sel_l_) > 0:
                idx_sel_l = idx_sel_l_[0]
                idx_sel_2 = np.concatenate((idx_sel_2, idx_sel_l), axis=0)

        if lane_in < (self.track.num_lane[seg_in] - 1):
            idx_sel_r_ = np.where(data_vehicle_ov_sel[:, 8] == (lane_in + 1))
            if len(idx_sel_r_) > 0:
                idx_sel_r = idx_sel_r_[0]
                idx_sel_2 = np.concatenate((idx_sel_2, idx_sel_r), axis=0)

        if len(idx_sel_2) > 0:
            idx_sel_2 = np.unique(idx_sel_2, axis=0)
            data_vehicle_ov_sel = data_vehicle_ov_sel[idx_sel_2, :]

        # Points (middle)
        pnts_c_intp = self.track.pnts_m_track[seg_in][lane_in]

        # Modify points (middle)
        if self.track.track_name == "US101" or self.track.track_name == "I80":
            if seg_in == 0:
                len_pnts = pnts_c_intp.shape[0]
                idx_2_update = range(math.floor(len_pnts * 1 / 5), len_pnts)
                pnts_c_intp = pnts_c_intp[idx_2_update, :]
        elif "highD" in self.track.track_name:
            len_pnts = pnts_c_intp.shape[0]
            idx_2_update = range(4, math.floor(len_pnts * 1 / 2))
            pnts_c_intp = pnts_c_intp[idx_2_update, :]
        else:
            pass

        # Check collision
        rad_c = 0
        idx_valid = np.zeros((pnts_c_intp.shape[0], ), dtype=np.int32)
        rad_valid = np.zeros((pnts_c_intp.shape[0], ), dtype=np.float32)
        cnt_valid = 0
        for nidx_p in range(0, pnts_c_intp.shape[0]):
            pnt_c_cur = pnts_c_intp[nidx_p, :]
            pnt_c_cur = pnt_c_cur.reshape(-1)

            if nidx_p < (pnts_c_intp.shape[0] - 1):
                pnt_c_next_ = pnts_c_intp[nidx_p + 1, :]
                pnt_c_next = pnt_c_next_.reshape(-1)
                vec_cur2next = pnt_c_next - pnt_c_cur
                rad_c = math.atan2(vec_cur2next[1], vec_cur2next[0])

            # data_t: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
            data_t = [0, pnt_c_cur[0], pnt_c_cur[1], rad_c, self.v_ego, self.ry + margin_ry, self.rx + margin_rx,
                      seg_in, lane_in, -1]
            is_collision_out = check_collision(data_t, data_vehicle_ov_sel)

            if is_collision_out == 0:
                cnt_valid = cnt_valid + 1
                idx_valid[cnt_valid - 1], rad_valid[cnt_valid - 1] = nidx_p, rad_c

        idx_valid, rad_valid = idx_valid[0:cnt_valid], rad_valid[0:cnt_valid]

        if cnt_valid > 0:
            idx_rand = np.random.randint(cnt_valid)
            idx_valid_sel = idx_valid[idx_rand]

            pnt_sel, rad_sel = pnts_c_intp[idx_valid_sel, :], rad_valid[idx_rand]

            # data_sel: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
            data_sel = [0, pnt_sel[0], pnt_sel[1], rad_sel, self.v_ego, self.ry, self.rx, seg_in, lane_in, -1]
            _, _, _, _, _, _, pnt_center, rad_center, _, _ = get_feature_part(self.track, data_sel, use_intp=0)

            # Update
            self.pnts_init = pnts_c_intp[idx_valid, :]

            self.x_ego_init, self.y_ego_init, self.theta_ego_init = pnt_center[0], pnt_center[1], rad_center
            self.x_ego, self.y_ego, self.theta_ego = pnt_center[0], pnt_center[1], rad_center

            if vel_rand:
                v_init = self.v_ref + (np.random.random() - 0.5) * self.v_ref
            else:
                v_init = self.v_ref

            self.v_ego_init, self.v_ego = v_init, v_init

            succeed_init = 1
        else:
            succeed_init = 0

        return succeed_init

    # Update init state
    def update_init_state(self, x, y, theta, v):
        # x, y, theta, v: (scalar) position-x, position-y, heading, linear-velocity
        self.x_ego_init, self.y_ego_init, self.theta_ego_init, self.v_ego_init = x, y, theta, v

    # Update state
    def update_state(self, x, y, theta, v):
        # x, y, theta, v: (scalar) position-x, position-y, heading, linear-velocity
        self.x_ego, self.y_ego, self.theta_ego, self.v_ego = x, y, theta, v

    # Update control
    def update_control(self, w, a):
        # w, a: (scalar) angular-velocity, acceleration
        self.w_ego, self.a_ego = w, a

    # Get next state
    def get_next_state(self, s_cur, w, a):
        # s_cur: (list) x, y, theta, v (dim = 4)
        # w, a: (scalar) angular-velocity, acceleration

        #       dot(x) = v * cos(theta)
        #       dot(y) = v * sin(theta)
        #       dot(theta) = v * kappa_1 * av
        #       dot(v) = kappa_2 * a
        x_new = s_cur[0] + math.cos(s_cur[2]) * s_cur[3] * self.dt
        y_new = s_cur[1] + math.sin(s_cur[2]) * s_cur[3] * self.dt
        theta_new = s_cur[2] + s_cur[3] * self.kappa[0] * w * self.dt
        lv_new = s_cur[3] + self.kappa[1] * a * self.dt

        theta_new = angle_handle(theta_new)
        theta_new = theta_new[0]

        s_new = np.array([x_new, y_new, theta_new, lv_new], dtype=np.float32)

        return s_new

    # Get trajectory (naive)
    def get_trajectory_naive(self, u, horizon):
        # u: (ndarray or list) angular-velocity, acceleration (dim = 2)
        # horizon: (scalar) trajectory horizon

        traj = np.zeros((horizon + 1, 4), dtype=np.float64)
        traj[0, :] = [self.x_ego, self.y_ego, self.theta_ego, self.v_ego]

        for nidx_h in range(1, horizon + 1):
            s_prev = traj[nidx_h - 1, :]
            s_new = self.get_next_state(s_prev, u[0], u[1])
            traj[nidx_h, :] = s_new

        traj = traj.astype(dtype=np.float32)
        return traj

    # Get trajectories (naive)
    def get_trajectory_array_naive(self, uset, horizon):
        # uset: (ndarray) set of controls (dim = N x 2)
        # horizon: (scalar) trajectory-horizon

        traj_array = []
        for nidx_d in range(0, uset.shape[0]):
            u_sel = uset[nidx_d, :]
            traj_out = self.get_trajectory_naive(u_sel, horizon)
            traj_array.append(traj_out)

        return traj_array

    # Get point ahead
    def get_pnt_ahead(self, pos, dist_forward, seg, lane):
        # pos: (ndarray) x, y (dim = 2)
        # dist_forward: (scalar) distance forward
        # seg, lane: (scalar) indexes of segment & lane

        if seg == -1 or lane == -1:
            seg_, lane_ = get_index_seg_and_lane(pos, self.track.pnts_poly_track)
            seg, lane = seg_[0], lane_[0]

        pnts_c_tmp = self.track.pnts_m_track[seg][lane]  # [0, :] start --> [end, :] end

        if seg < (self.track.num_seg - 1):
            child_tmp = self.track.idx_child[seg][lane]
            for nidx_tmp in range(0, len(child_tmp)):
                pnts_c_next_tmp = self.track.pnts_m_track[seg + 1][self.track.idx_child[seg][lane][nidx_tmp]]  # [0, :] start --> [end, :] end
                pnts_c_tmp = np.concatenate((pnts_c_tmp, pnts_c_next_tmp), axis=0)
            pnts_c = pnts_c_tmp
        else:
            pnts_c = pnts_c_tmp

        pos_r = np.reshape(pos, (1, 2))
        diff_tmp = np.tile(pos_r, (pnts_c.shape[0], 1)) - pnts_c[:, 0:2]
        dist_tmp = np.sqrt(np.sum(diff_tmp*diff_tmp, axis=1))
        idx_cur = np.argmin(dist_tmp, axis=0)

        dist_sum = 0.0
        idx_ahead = pnts_c.shape[0] - 1
        for nidx_d in range(idx_cur + 1, pnts_c.shape[0]):
            dist_c_tmp1 = (pnts_c[nidx_d, 0] - pnts_c[nidx_d - 1, 0]) * (pnts_c[nidx_d, 0] - pnts_c[nidx_d - 1, 0]) +\
                          (pnts_c[nidx_d, 1] - pnts_c[nidx_d - 1, 1]) * (pnts_c[nidx_d, 1] - pnts_c[nidx_d - 1, 1])
            dist_c_tmp2 = math.sqrt(dist_c_tmp1)
            dist_sum = dist_sum + dist_c_tmp2
            if dist_sum > dist_forward:
                idx_ahead = nidx_d
                break

        pnt_ahead = pnts_c[idx_ahead, :]
        pnt_ahead = pnt_ahead.reshape(-1)

        return pnt_ahead

    # Find control (naive)
    def find_control_naive(self, u_set, horizon, data_ov, time_in, dist_ahead):
        # u_set: (ndarray) control candidate set (dim = N x 2)
        # horizon: (scalar) trajectory horizon
        # data_ov: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = N x 10)
        # time_in: (scalar) time

        if len(u_set) > 0:
            if ~isinstance(u_set, np.ndarray):
                u_set = np.array(u_set)

        if len(data_ov) > 0:
            if ~isinstance(data_ov, np.ndarray):
                data_ov = np.array(data_ov)

        len_u_set = u_set.shape[0]
        h_horizon_c = math.floor(horizon * 2 / 5)  # horizon to compute

        traj_array = []

        check_outside_array = np.zeros((len_u_set,), dtype=np.int32)
        check_collision_array = np.zeros((len_u_set, ), dtype=np.int32)
        check_lv_array = np.zeros((len_u_set, ), dtype=np.int32)
        dist_array = np.zeros((len_u_set,), dtype=np.float32)
        cnt_check = 0

        x_cur = np.array([self.x_ego, self.y_ego, self.theta_ego], dtype=np.float32)
        pnt_ahead = self.get_pnt_ahead(x_cur[0:2], dist_ahead, -1, -1)

        # Save other vehicle data w.r.t. time
        data_vehicle_array = []
        if len(data_ov) > 0:
            for nidx_h in range(1, horizon + 1):
                # Select data (w.r.t. time)
                time_sel = time_in + nidx_h
                idx_sel_1_ = np.where(data_ov[:, 0] == time_sel)
                idx_sel_1 = idx_sel_1_[0]
                data_vehicle_sel = data_ov[idx_sel_1, :]

                data_vehicle_array.append(data_vehicle_sel)

        for nidx_u in range(0, len_u_set):
            u_sel = u_set[nidx_u, :]

            traj_tmp = self.get_trajectory_naive(u_sel[0:2], horizon)

            h_outside, h_collision, h_lv_outside = 0, 0, 0
            for nidx_h in range(1, h_horizon_c):
                is_outside, is_collision, is_lv_outside = 0, 0, 0

                x_tmp = traj_tmp[nidx_h, :]

                # Get indexes of seg & lane
                seg_tmp_, lane_tmp_ = get_index_seg_and_lane(x_tmp[0:2], self.track.pnts_poly_track)
                seg_tmp, lane_tmp = seg_tmp_[0], lane_tmp_[0]

                # Check inside-track =---------------------------------------------------------------------------------#
                if seg_tmp == -1 or lane_tmp == -1:
                    h_outside = nidx_h + 1
                    is_outside = 1

                # Check collision -------------------------------------------------------------------------------------#
                if len(data_vehicle_array) > 0:
                    data_vehicle_sel = data_vehicle_array[nidx_h - 1]

                    # data_t: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
                    data_t = [0, x_tmp[0], x_tmp[1], x_tmp[2], u_sel[0], (self.ry * 1.1), (self.rx * 1.2),
                              seg_tmp, lane_tmp, -1]
                    is_collision = check_collision(data_t, data_vehicle_sel)

                    if is_collision == 1:
                        h_collision = nidx_h + 1

                # Check linear-velocity -------------------------------------------------------------------------------#
                is_lv_outside = (x_tmp[3] < self.v_range[0]) or (x_tmp[3] > self.v_range[1])
                if is_lv_outside:
                    h_lv_outside = nidx_h + 1

                if is_outside == 1 or is_collision == 1 or is_lv_outside == 1:
                    break

            # Check reward --------------------------------------------------------------------------------------------#
            diff_ahead = pnt_ahead[0:2] - traj_tmp[-1, 0:2]
            dist_ahead = norm(diff_ahead)

            # Update
            traj_array.append(traj_tmp)
            cnt_check = cnt_check + 1
            check_outside_array[cnt_check - 1] = h_outside
            check_collision_array[cnt_check - 1] = h_collision
            check_lv_array[cnt_check - 1] = h_lv_outside

            dist_array[cnt_check - 1] = dist_ahead

        # Choose trajectory
        idx_invalid_1 = np.where(check_outside_array >= 1)
        idx_invalid_2 = np.where(check_collision_array >= 1)
        idx_invalid_3 = np.where(check_lv_array >= 1)

        idx_invalid_ = np.concatenate((idx_invalid_1[0], idx_invalid_2[0]), axis=0)
        idx_invalid_ = np.concatenate((idx_invalid_, idx_invalid_3[0]), axis=0)
        if len(idx_invalid_) > 0:
            idx_invalid = np.unique(idx_invalid_)
        else:
            idx_invalid = idx_invalid_

        cost_array = dist_array
        cost_array_c = np.copy(cost_array)
        cost_array_c[idx_invalid] = 10000

        idx_min = np.argmin(cost_array_c)
        traj_sel = traj_array[idx_min]

        return traj_sel, traj_array, cost_array, idx_invalid, pnt_ahead

    # Find control (w.r.t. reward)
    def find_control_wrt_r(self, u_set, horizon, data_ov, time_in):
        # u_set: (ndarray) control candidate set (dim = N x 2)
        # horizon: (scalar) trajectory horizon
        # data_ov: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = N x 10)
        # time_in: (scalar) time

        if len(u_set) > 0:
            if ~isinstance(u_set, np.ndarray):
                u_set = np.array(u_set)

        if len(data_ov) > 0:
            if ~isinstance(data_ov, np.ndarray):
                data_ov = np.array(data_ov)

        len_u_set = u_set.shape[0]

        traj_array = []
        r_array = np.zeros((len_u_set, ), dtype=np.float32)

        for nidx_u in range(0, len_u_set):
            u_sel = u_set[nidx_u, :]
            traj_tmp = self.get_trajectory_naive(u_sel[0:2], horizon)
            traj_array.append(traj_tmp)

            for nidx_h in range(0, horizon):
                nidx_t = time_in + nidx_h + 1

                # Get next state
                s_ev_next = traj_tmp[nidx_h + 1, :]

                seg_ev_next, lane_ev_next, data_ov_next, data_ev_next, f_next, id_near_next, data_ov_near_all_next, \
                data_ov_near_next, _ = self.get_info_t(nidx_t, s_ev_next, s_ev_next[3], data_ov, use_intp=1)

                # EPI-STEP4: GET REWARD -------------------------------------------------------------------------------#
                # Compute reward component
                r_outside, r_collision, r_dev_rad, r_dev_dist, r_cf_dist, r_speed, r_angular, r_goal, dist2goal = \
                    self.compute_reward_component(nidx_t, data_ev_next, s_ev_next[3], u_sel[0], data_ov_near_all_next,
                                                  f_next[0], f_next[4], f_next[8])
                # Get reward
                r_cur = self.compute_reward(r_outside, r_collision, r_dev_rad, r_dev_dist, r_cf_dist, r_speed,
                                            r_angular, r_goal)

                r_array[nidx_u] = r_array[nidx_u] + r_cur

        idx_max = np.argmax(r_array)
        traj_sel = traj_array[idx_max]

        return traj_sel, traj_array, r_array

    # [UTILS] ---------------------------------------------------------------------------------------------------------#
    # Get info w.r.t. time (utils)
    def get_info_t(self, t_cur, s_ev_t, v_ego, data_ov, use_intp=1):
        # Set indexes of seg & lane
        seg_ev_t_, lane_ev_t_ = get_index_seg_and_lane(s_ev_t[0:2], self.track.pnts_poly_track)
        seg_ev_t, lane_ev_t = seg_ev_t_[0], lane_ev_t_[0]

        # Select current vehicle data (w.r.t time)
        if len(data_ov) == 0:
            data_ov_t = []
        else:
            idx_sel = np.where(data_ov[:, 0] == t_cur)
            data_ov_t = data_ov[idx_sel[0], :]

        # Set data vehicle ego
        # structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
        data_ev_t = np.array([t_cur, s_ev_t[0], s_ev_t[1], s_ev_t[2], v_ego, self.ry, self.rx, seg_ev_t, lane_ev_t, -1],
                             dtype=np.float32)

        # Get feature
        f_t, id_near_t, _, _, _, dist_cf = get_feature(self.track, data_ev_t, data_ov_t, use_intp=use_intp)

        # Get near other vehicles
        data_ov_near_all_t = get_selected_vehicles(data_ov, id_near_t)

        if len(data_ov_near_all_t) > 0:
            idx_sel = np.where(data_ov_near_all_t[:, 0] == t_cur)
            data_ov_near_t = data_ov_near_all_t[idx_sel[0], :]
        else:
            data_ov_near_t = []

        return seg_ev_t, lane_ev_t, data_ov_t, data_ev_t, f_t, id_near_t, data_ov_near_all_t, data_ov_near_t, dist_cf

    # Compute reward component (utils)
    def compute_reward_component(self, t_cur, data_ev_next, v_ev, w_ev, data_ov_near_all_next, lane_dev_rad,
                                 lane_dev_dist_scaled, dist_cf):
        # t_cur: (scalar) current-time
        # data_ev_next: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
        # v_ev, w_ev: (scalar) linear-velocity, angular-velocity

        # 0: Check outside
        if data_ev_next[7] == -1 or data_ev_next[8] == -1:
            r_outside = 1
        else:
            r_outside = 0

        # 1: Check collision
        r_collision = check_collision_near(data_ev_next, data_ov_near_all_next, t_cur + 1)

        # 2 ~ 6
        r_dev_rad, r_dev_dist, r_cf_dist, r_speed, r_angular = \
            self.compute_reward_component_rest(v_ev, w_ev, lane_dev_rad, lane_dev_dist_scaled, dist_cf)

        # 7: Reach goal pnts
        dist2goal_, reach_goal_ = get_dist2goal(data_ev_next[1:3], data_ev_next[7], data_ev_next[8],
                                                self.track.indexes_goal, self.track.pnts_goal)
        dist2goal = min(dist2goal_)
        r_goal = max(reach_goal_)

        return r_outside, r_collision, r_dev_rad, r_dev_dist, r_cf_dist, r_speed, r_angular, r_goal, dist2goal

    # Compute reward component rest (utils)
    def compute_reward_component_rest(self, v_ev, w_ev, lane_dev_rad, lane_dev_dist_scaled, dist_cf):
        # v_ev, w_ev: (scalar) linear-velocity, angular-velocity

        r_w_lanedev = self.r_weight[2]

        # 2: Lane dev (rad)
        if abs(lane_dev_rad) > self.dev_rad_th * 2:
            r_dev_rad = -10 / r_w_lanedev
        elif abs(lane_dev_rad) > self.dev_rad_th:
            r_dev_rad = -0.33 / r_w_lanedev
        else:
            r_dev_rad = (self.dev_rad_th - abs(lane_dev_rad)) * (self.dev_rad_th - abs(lane_dev_rad))
            # max: (dev_rad_th)^2, min: 0

        # 3: Lane dev (dist)
        r_dev_dist = (0.5 - abs(lane_dev_dist_scaled)) * (0.5 - abs(lane_dev_dist_scaled))  # max: 0.25, min: 0

        # 4: Center front dist
        if dist_cf > self.dist_cf_th:  # max: (dist_cf_th)^2, min: 0
            r_cf_dist = self.dist_cf_th * self.dist_cf_th
        else:
            r_cf_dist = dist_cf * dist_cf

        # 5: Move forward
        if (v_ev < self.v_range[0]) or (v_ev > self.v_range[1]):
            r_speed = -100
        else:
            r_speed = self.v_ref * self.v_ref - (self.v_ref - v_ev) * (self.v_ref - v_ev)  # min: 0, max: (lev_ref)^2

        # 6: Turn left or right
        r_angular = self.w_max * self.w_max - (w_ev * w_ev)  # min: 0, max: (av_max)^2

        return r_dev_rad, r_dev_dist, r_cf_dist, r_speed, r_angular

    # Compute reward (utils)
    def compute_reward(self, r_outside, r_collision, r_dev_rad, r_dev_dist, r_cf_dist, r_speed, r_angular, r_goal):
        r_out = self.r_weight[0] * r_outside + self.r_weight[1] * r_collision + self.r_weight[2] * r_dev_rad + \
                self.r_weight[3] * r_dev_dist + self.r_weight[4] * r_cf_dist + self.r_weight[5] * r_speed + \
                self.r_weight[6] * r_angular + self.r_weight[7] * r_goal
        r_out = min(max(r_out, -1), + 1)

        return r_out

