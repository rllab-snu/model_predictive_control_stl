# MODEL PREDICTIVE CONTROL UNDER STL-CONSTRAINTS (MPCSTL)
#   - Reference:
#       [1] Kyunghoon Cho and Songhwai Oh,
#       "Learning-Based Model Predictive Control under Signal Temporal Logic Specifications,"
#       in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), May 2018.
#
#   - STL rules:
#       1: Lane-observation (down, right)
#       2: Lane-observation (up, left)
#       3: Collision (front)
#       4: Collision (others)
#       5: Speed-limit
#       6: Slow-down
#
#   - System Dynamic: 4-state unicycle dynamics
#       State: [x, y, theta, v]
#       Control: [a, w]
#           dot(x) = v * cos(theta)
#           dot(y) = v * sin(theta)
#           dot(theta) = v * kappa_1 * w
#           dot(v) = kappa_2 * a


from src.utils_sim import *
from src.utils_stl import *
from src.get_rgb import *
from gurobipy import *  # Use gurobi for optimization


class MPCSTL(object):
    def __init__(self, dt, rx, ry, kappa, h, h_relax, u_lb, u_hb, c_end, c_u, v_ref, v_range, dist_ahead, v_th,
                 until_t_s, until_t_a, until_t_b, until_v_th, until_d_th, until_lanewidth, do_plot_debug):
        self.dt = dt  # Time step
        self.rx, self.ry = rx, ry  # Ego-vehicle size
        self.kappa = kappa  # Dynamic parameters
        self.h = h  # MPC-horizon
        self.h_relax = h_relax  # Horizon of stl-constraint relaxation

        self.dim_x = 4  # Dimension of state
        self.dim_u = 2  # Dimension of control
        self.maxnum_oc = 6  # Maximum number of other-vehicles to consider

        # Initial state (ego-vehicle)
        self.xinit = np.zeros((self.dim_x,), dtype=np.float32)
        self.xinit_conv = np.zeros((self.dim_x,), dtype=np.float32)

        # Goal state
        self.xgoal, self.xgoal_conv = [], []

        # Lower & Upper bounds for control
        self.u_lb, self.u_ub = u_lb, u_hb

        self.c_end, self.c_u = c_end, c_u
        self.v_ref = v_ref  # Reference (linear) velocity
        self.v_range = v_range  # Range of (linear) velocity
        self.dist_ahead = dist_ahead  # Distance to the goal state
        self.v_th = v_th  # Velocity threshold

        # Until-logic parameters
        self.until_t_s, self.until_t_a, self.until_t_b = until_t_s, until_t_a, until_t_b
        self.until_v_th_ref, self.until_d_th_ref = until_v_th, until_d_th
        self.until_lanewidth = until_lanewidth
        self.until_v_th, self.until_d_th = until_v_th, until_d_th

        # OPTIMIZATION (GUROBI) ---------------------------------------------------------------------------------------#
        # Set optimization model
        self.opt_model = None  # Optimization model

        # Set optimization parameters
        self.list_h, self.list_dim_u, self.list_dim_x = range(0, self.h), range(0, self.dim_u), range(0, self.dim_x)

        self.params_u_lb = {(i, j): 0 for i in self.list_h for j in self.list_dim_u}  # Lower bound (control)
        for nidx_i in self.list_h:
            self.params_u_lb[nidx_i, 0], self.params_u_lb[nidx_i, 1] = u_lb[0], u_lb[1]

        self.params_u_ub = {(i, j): 0 for i in self.list_h for j in self.list_dim_u}  # Upper bound (control)
        for nidx_i in self.list_h:
            self.params_u_ub[nidx_i, 0], self.params_u_ub[nidx_i, 1] = u_hb[0], u_hb[1]

        self.params_x_lb = {(i, j): 0 for i in self.list_h for j in self.list_dim_x}  # Lower bound (state)
        for nidx_i in self.list_h:
            self.params_x_lb[nidx_i, 0], self.params_x_lb[nidx_i, 1] = -200, -200
            self.params_x_lb[nidx_i, 2] = -math.pi / 2.0
            self.params_x_lb[nidx_i, 3] = self.v_range[0]

        self.params_x_ub = {(i, j): 0 for i in self.list_h for j in self.list_dim_x}  # Upper bound (state)
        for nidx_i in self.list_h:
            self.params_x_ub[nidx_i, 0], self.params_x_ub[nidx_i, 1] = +200, +200
            self.params_x_ub[nidx_i, 2] = +math.pi / 2.0
            self.params_x_ub[nidx_i, 3] = self.v_range[1]

        self.cp_l_d, self.cp_l_u = [], []  # Lane-constraints (point)
        self.rad_l = 0.0  # Lane-constraints (angle)

        self.traj_ov_cf = np.zeros((1, 3))  # Collision-constraints (trajectory): id_cf
        self.size_ov_cf = np.zeros((1, 2))  # Collision-constraints (size: dx, dy): id_cf
        self.traj_ov = []  # Collision-constraints (trajectory): list [id_lf, id_lr, id_rf, id_rr, id_cr]
        self.size_ov = []  # Collision-constraints (size: dx, dy): list [id_lf, id_lr, id_rf, id_rr, id_cr]

        # Initial computed robustness slackness
        self.r_l_down_init, self.r_l_up_init, self.r_c_cf_init = 0.0, 0.0, 0.0
        self.r_c_rest_array_init = []

        self.M = float(5e5)  # Big value (mixed-integer programming)

        # Set variables
        self.u_vars, self.x_vars = [], []
        self.z_c_cf_vars, self.r_c_cf_vars = [], []
        self.z_c_ov_vars, self.r_c_ov_vars = [], []

        self.r_until, self.r_until_and1 = [], []
        self.z_until_and1, self.z_until_and2 = [], []
        self.r_until_and_hist, self.z_until_and_hist = [], []
        self.r_until_or_hist, self.z_until_or_hist = [], []
        self.r_until_alw, self.z_until_alw = [], []
        self.r_until_ev, self.z_until_ev = [], []

        # FOR DEBUG ---------------------------------------------------------------------------------------------------#
        self.do_plot_debug = do_plot_debug
        self.cp_l_d_rec, self.cp_l_u_rec = [], []  # Lane-constraints (point)
        self.traj_l_d_rec, self.traj_l_u_rec = [], []

        # History
        self.cnt_hist_ev = 0
        self.x_hist_ev = np.zeros((10000, 4), dtype=np.float32)
        self.xgoal_hist_ev = np.zeros((10000, 2), dtype=np.float32)
        self.cpd_hist_ev = np.zeros((10000, 2), dtype=np.float32)
        self.cpu_hist_ev = np.zeros((10000, 2), dtype=np.float32)

        self.x_conv_hist_ev = np.zeros((10000, 4), dtype=np.float32)
        self.xgoal_conv_hist_ev = np.zeros((10000, 2), dtype=np.float32)
        self.cpd_conv_hist_ev = np.zeros((10000, 2), dtype=np.float32)
        self.cpu_conv_hist_ev = np.zeros((10000, 2), dtype=np.float32)

    # Reset optimization model
    def reset_opt_model(self):
        self.opt_model = None
        self.opt_model = Model("MPCSTL")
        self.opt_model.setParam('OutputFlag', False)  # Whether to be verbose

        # self.opt_model.setParam("SubMIPNodes", 2e9)  # Nodes explored in sub-MIP heuristics (max: 2e9)
        # self.opt_model.setParam("Heuristics", 0.6)  # Time spent in feasibility heuristics
        # Larger values produce more and better feasible solutions,
        # at a cost of slower progress in the best bound.

        # self.opt_model.setParam('TimeLimit', 1000000)
        self.opt_model.setParam('FeasibilityTol', float(1e-9))
        self.opt_model.setParam('OptimalityTol', float(1e-9))
        # self.opt_model.setParam('IntFeasTol', 0.1)
        # self.opt_model.setParam('MIPGap', 0.001)
        # self.opt_model.setParam('Method', 2)
        # self.opt_model.setParam('ImproveStartTime', 600)
        return

    # Get goal point (ver1)
    def get_goal_state_ver1(self, sim_control, s_ev_cur):
        pnt_goal = sim_control.get_pnt_ahead(s_ev_cur[0:2], self.dist_ahead, -1, -1)
        theta_goal = get_lane_rad(pnt_goal[0:2], sim_control.track.pnts_poly_track,
                                  sim_control.track.pnts_lr_border_track)
        pose_goal = np.zeros((self.dim_x,), dtype=np.float32)
        # pose_goal[0:2], pose_goal[2], pose_goal[3] = pnt_goal, theta_goal, sim_control.lv_ref
        pose_goal[0:2], pose_goal[2], pose_goal[3] = pnt_goal, theta_goal, 0.0
        return pose_goal

    # Get goal point (ver2)
    def get_goal_state_ver2(self, pnt, angle):
        dx = math.cos(angle) * self.dist_ahead
        dy = math.sin(angle) * self.dist_ahead
        x_goal = pnt[0] + dx
        y_goal = pnt[1] + dy

        pose_goal = np.zeros((self.dim_x,), dtype=np.float32)
        pose_goal[0], pose_goal[1], pose_goal[2], pose_goal[3] = x_goal, y_goal, angle, 0.0
        return pose_goal

    # Rotate w.r.t. reference pose
    def convert_state(self, x_in, xgoal_in, cp2rotate, theta2rotate):
        # x_in: (ndarray) ego-vehicle system state (dim = 4)
        # xgoal_in: (ndarray) ego-vehicle system state (dim = 4)
        # cp2rotate: (ndarray) center point to convert (dim = 2)
        # theta2rotate: (scalar) angle (rad) to convert

        # Convert state (ego-vehicle)
        self.xinit = x_in
        xconv_in = np.zeros((self.dim_x,), dtype=np.float32)
        x_in_r = np.reshape(x_in[0:2], (1, 2))
        xconv_in_tmp1 = get_rotated_pnts_tr(x_in_r, -cp2rotate, -theta2rotate)
        xconv_in_tmp1 = xconv_in_tmp1.reshape(-1)
        xconv_in[0:2] = xconv_in_tmp1
        xconv_in[2] = angle_handle(x_in[2] - theta2rotate)
        xconv_in[3] = x_in[3]
        self.xinit_conv = xconv_in

        # Convert state (goal)
        if len(xgoal_in) > 0:
            self.xgoal = xgoal_in
            xgoal_conv_in = np.zeros((self.dim_x,), dtype=np.float32)
            xgoal_in_r = np.reshape(xgoal_in[0:2], (1, 2))
            xgoal_conv_in_tmp1 = get_rotated_pnts_tr(xgoal_in_r, -cp2rotate, -theta2rotate)
            xgoal_conv_in_tmp1 = xgoal_conv_in_tmp1.reshape(-1)
            xgoal_conv_in[0:2] = xgoal_conv_in_tmp1
            xgoal_conv_in[2] = angle_handle(xgoal_in[2] - theta2rotate)
            xgoal_conv_in[3] = xgoal_in[3]
            self.xgoal_conv = xgoal_conv_in
        else:
            xgoal_conv_in = np.zeros((self.dim_x,), dtype=np.float32)
            xgoal_conv_in[0] = self.dist_ahead
            self.xgoal_conv = xgoal_conv_in

            self.xgoal = np.zeros((self.dim_x,), dtype=np.float32)
            pnt_goal_tmp = get_rotated_pnts_rt(xgoal_conv_in[0:2], +cp2rotate, +theta2rotate)
            self.xgoal[0:2] = pnt_goal_tmp

    # Get lane-constraints
    def get_lane_constraints(self, pnt_down, pnt_up, lane_angle, margin_dist, cp2rotate, theta2rotate):
        # pnt_down, pnt_up: (ndarray) point (dim = 2)
        # lane_angle: (scalar) lane-heading (rad)
        # margin_dist: (scalar) margin dist (bigger -> loosing constraints)
        # cp2rotate: (ndarray) center point to convert (dim = 2)
        # theta2rotate: (scalar) angle (rad) to convert

        self.cp_l_d, self.cp_l_u, self.rad_l = [], [], 0.0

        pnt_down_r = np.reshape(pnt_down[0:2], (1, 2))
        pnt_up_r = np.reshape(pnt_up[0:2], (1, 2))
        pnt_down_conv_ = get_rotated_pnts_tr(pnt_down_r, -cp2rotate, -theta2rotate)
        pnt_up_conv_ = get_rotated_pnts_tr(pnt_up_r, -cp2rotate, -theta2rotate)
        pnt_down_conv, pnt_up_conv = pnt_down_conv_[0, :], pnt_up_conv_[0, :]
        lane_angle_r = angle_handle(lane_angle - theta2rotate)

        margin_dist = margin_dist - self.ry / 2.0
        self.cp_l_d = pnt_down_conv + np.array([+margin_dist*math.sin(lane_angle_r),
                                                -margin_dist*math.cos(lane_angle_r)], dtype=np.float32)
        self.cp_l_u = pnt_up_conv + np.array([-margin_dist * math.sin(lane_angle_r),
                                              +margin_dist * math.cos(lane_angle_r)], dtype=np.float32)
        self.rad_l = lane_angle_r

        self.cp_l_d_rec = get_rotated_pnts_rt(self.cp_l_d, +cp2rotate, +theta2rotate)
        self.cp_l_u_rec = get_rotated_pnts_rt(self.cp_l_u, +cp2rotate, +theta2rotate)

        xtmp = np.arange(start=-30, stop=+30, step=0.5)
        len_xtmp = xtmp.shape[0]
        y_lower = self.cp_l_d[1] * np.ones((len_xtmp,), dtype=np.float32)
        y_upper = self.cp_l_u[1] * np.ones((len_xtmp,), dtype=np.float32)
        traj_lower = np.zeros((len_xtmp, 2), dtype=np.float32)
        traj_upper = np.zeros((len_xtmp, 2), dtype=np.float32)
        traj_lower[:, 0] = xtmp
        traj_lower[:, 1] = y_lower
        traj_upper[:, 0] = xtmp
        traj_upper[:, 1] = y_upper

        self.traj_l_d_rec = get_rotated_pnts_rt(traj_lower, +cp2rotate, +theta2rotate)
        self.traj_l_u_rec = get_rotated_pnts_rt(traj_upper, +cp2rotate, +theta2rotate)

    # Get collision-constraints
    def get_collision_constraints(self, traj_ov_near_list, size_ov_near_list, id_near, cp2rotate, theta2rotate):
        # traj_ov_near_list: (list)-> (ndarray) x y theta (dim = N x 3)
        #                  : [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
        # size_ov_near_list: (list)-> (ndarray) dx dy (dim = N x 2)
        #                  : [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
        # id_near: (ndarray) selected indexes [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
        # cp2rotate: (ndarray) center point to convert (dim = 2)
        # theta2rotate: (scalar) angle (rad) to convert

        # Do reset
        self.traj_ov, self.size_ov = [], []

        len_near = len(traj_ov_near_list)

        for nidx_l in range(0, len_near):
            traj_ov_near_list_sel = traj_ov_near_list[nidx_l]
            size_ov_near_list_sel = size_ov_near_list[nidx_l]
            id_near_sel = id_near[nidx_l]

            if nidx_l == 4:  # id_cf
                if id_near_sel == -1:
                    self.traj_ov_cf = traj_ov_near_list_sel
                    self.size_ov_cf = np.zeros((self.h + 1, 2), dtype=np.float32)
                else:
                    traj_tmp = np.zeros((self.h + 1, 3), dtype=np.float32)
                    traj_tmp[:, 0:2] = get_rotated_pnts_tr(traj_ov_near_list_sel[:, 0:2], -cp2rotate, -theta2rotate)
                    traj_tmp[:, 2] = angle_handle(traj_ov_near_list_sel[:, 2] - theta2rotate)

                    diff_tmp = traj_tmp[:, 0:2] - np.reshape(self.xinit_conv[0:2], (1, 2))
                    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
                    idx_tmp_ = np.where(dist_tmp > 100.0)
                    idx_tmp_ = idx_tmp_[0]
                    if len(idx_tmp_) > 0:
                        traj_tmp[idx_tmp_, 0:2] = [self.xinit_conv[0] - 100, self.xinit_conv[1] - 100]

                    self.traj_ov_cf = traj_tmp

                    self.size_ov_cf = np.zeros((self.h + 1, 2), dtype=np.float32)
                    for nidx_h in range(0, self.h + 1):
                        traj_tmp_sel = traj_tmp[nidx_h, :]
                        size_ov_near_list_sel_new = np.array([size_ov_near_list_sel[0], size_ov_near_list_sel[1]],
                                                             dtype=np.float32)
                        if size_ov_near_list_sel_new[1] < 2.0 * self.ry:
                            size_ov_near_list_sel_new[1] = 2.0 * self.ry
                        size_tmp_sel = self.get_modified_size_linear(size_ov_near_list_sel_new, traj_tmp_sel[2])
                        self.size_ov_cf[nidx_h, :] = size_tmp_sel

            else:
                if id_near_sel == -1:
                    traj_tmp = traj_ov_near_list_sel
                    size_tmp = np.zeros((self.h + 1, 2), dtype=np.float32)
                else:
                    traj_tmp = np.zeros((self.h + 1, 3), dtype=np.float32)
                    traj_tmp[:, 0:2] = get_rotated_pnts_tr(traj_ov_near_list_sel[:, 0:2], -cp2rotate, -theta2rotate)
                    traj_tmp[:, 2] = angle_handle(traj_ov_near_list_sel[:, 2] - theta2rotate)

                    diff_tmp = traj_tmp[:, 0:2] - np.reshape(self.xinit_conv[0:2], (1, 2))
                    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
                    idx_tmp_ = np.where(dist_tmp > 100.0)
                    idx_tmp_ = idx_tmp_[0]
                    if len(idx_tmp_) > 0:
                        traj_tmp[idx_tmp_, 0:2] = [self.xinit_conv[0] - 100, self.xinit_conv[1] - 100]

                    size_tmp = np.zeros((self.h + 1, 2), dtype=np.float32)
                    for nidx_h in range(0, self.h + 1):
                        traj_tmp_sel = traj_tmp[nidx_h, :]
                        size_tmp_sel = self.get_modified_size_linear(size_ov_near_list_sel, traj_tmp_sel[2])
                        size_tmp[nidx_h, :] = size_tmp_sel

                self.traj_ov.append(traj_tmp)
                self.size_ov.append(size_tmp)

    # Get modified size for linearization
    def get_modified_size_linear(self, size_in, heading, w=0.4):
        # size_in: (ndarray) dx dy (dim = 2)
        # heading: (scalar) heading
        # w: (scalar) weight

        rx = self.rx  # Dummy
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

    # Update history
    def update_history(self):
        self.cnt_hist_ev = self.cnt_hist_ev + 1

        self.x_hist_ev[self.cnt_hist_ev - 1, :] = self.xinit
        self.xgoal_hist_ev[self.cnt_hist_ev - 1, :] = self.xgoal[0:2]
        self.cpd_hist_ev[self.cnt_hist_ev - 1, :] = self.cp_l_d_rec
        self.cpu_hist_ev[self.cnt_hist_ev - 1, :] = self.cp_l_u_rec

        self.x_conv_hist_ev[self.cnt_hist_ev - 1, :] = self.xinit_conv
        self.xgoal_conv_hist_ev[self.cnt_hist_ev - 1, :] = self.xgoal_conv[0:2]
        self.cpd_conv_hist_ev[self.cnt_hist_ev - 1, :] = self.cp_l_d
        self.cpu_conv_hist_ev[self.cnt_hist_ev - 1, :] = self.cp_l_u

    # Compute cost
    def compute_cost(self, traj_in, u_in):
        # traj_in: (ndarray) ego-vehicle trajectory (dim = H x 4)
        # u_in: (ndarray) ego-vehicle control (dim = H x 2)

        if ~isinstance(traj_in, np.ndarray):
            traj_in = np.array(traj_in)
        size_traj_in = traj_in.shape
        if len(size_traj_in) == 1:
            traj_in = np.reshape(traj_in, (1, -1))

        if ~isinstance(u_in, np.ndarray):
            u_in = np.array(u_in)
        size_u_in = u_in.shape
        if len(size_u_in) == 1:
            u_in = np.reshape(u_in, (1, -1))

        len_traj_in = traj_in.shape[0]
        len_u_in = u_in.shape[0]

        cost_xend = np.zeros((self.dim_x,), dtype=np.float32)
        for nidx_d in range(0, self.dim_x):
            cost_xend[nidx_d] = (traj_in[len_traj_in - 1, nidx_d] - float(self.xgoal_conv[nidx_d])) * \
                                (traj_in[len_traj_in - 1, nidx_d] - float(self.xgoal_conv[nidx_d])) * \
                                float(self.c_end[nidx_d])

        cost_u = np.zeros((self.dim_u,), dtype=np.float32)
        for nidx_h in range(0, len_u_in):
            for nidx_d in range(0, self.dim_u):
                cost_u[nidx_d] += (u_in[nidx_h, nidx_d]) * (u_in[(nidx_h, nidx_d)]) * float(self.c_u[nidx_d])

        return cost_xend, cost_u

    # Compute robustness (for initial state)
    def compute_robustness_slackness_init(self, rmin_l_d_in, rmin_l_u_in, rmin_c_cf_in, rmin_c_rest_in, rmin_speed_in):
        idx_h_oc = np.arange(0, 1)
        r_l_down, r_l_up, r_c_cf, r_c_rest, r_c_rest_array, r_speed = \
            compute_robustness_part(self.xinit_conv, [self.rx, self.ry], self.cp_l_d, self.cp_l_u, self.rad_l,
                                    self.traj_ov_cf, self.size_ov_cf, self.traj_ov, self.size_ov, idx_h_oc, self.v_th)

        self.r_l_down_init = r_l_down
        self.r_l_up_init = r_l_up
        self.r_c_cf_init = r_c_cf
        self.r_c_rest_array_init = r_c_rest_array

        idx_relax = np.arange(0, self.h_relax)

        # Rule 1-2: lane-observation (down, up)
        rmin_l_d = rmin_l_d_in * np.ones((self.h,), dtype=np.float32)
        rmin_l_u = rmin_l_u_in * np.ones((self.h,), dtype=np.float32)

        if self.h_relax > 0:
            rmin_l_d[idx_relax] = min(r_l_down - 0.2, rmin_l_d_in)
            rmin_l_u[idx_relax] = min(r_l_up - 0.2, rmin_l_u_in)

        # Rule 3: collision (front)
        rmin_c_cf = rmin_c_cf_in * np.ones((self.h,), dtype=np.float32)
        if self.h_relax > 0:
            rmin_c_cf[idx_relax] = min(r_c_cf - 0.2, rmin_c_cf_in)

        # Rule 4: collision (rest)
        num_ov_rest = len(self.traj_ov)
        rmin_c_rest = []
        for nidx_oc in range(0, num_ov_rest):
            rmin_c_rest_tmp = rmin_c_rest_in * np.ones((self.h,), dtype=np.float32)
            if self.h_relax > 0:
                rmin_c_rest_tmp[idx_relax] = min(r_c_rest_array[nidx_oc] - 0.2, rmin_c_rest_in)

            rmin_c_rest.append(rmin_c_rest_tmp)

        # Rule 5: speed-limit
        rmin_c_speed = rmin_speed_in * np.ones((self.h,), dtype=np.float32)
        if self.h_relax > 0:
            rmin_c_speed[idx_relax] = min(r_speed - 0.2, rmin_speed_in)

        return rmin_l_d, rmin_l_u, rmin_c_cf, rmin_c_rest, rmin_c_speed

    # MODEL PREDICTIVE CONTROL ----------------------------------------------------------------------------------------#
    def control_by_mpc(self, rmin_l_d, rmin_l_u, rmin_c_cf, rmin_c_rest, rmin_speed, rmin_until, id_cf, dist_cf,
                       cp2rotate, theta2rotate, lanewidth_cur):
        # cp2rotate: (ndarray) center point to convert (dim = 2)
        # theta2rotate: (scalar) angle (rad) to convert
        # lanewidth_cur: (scalar) current lanewidth

        if id_cf == -1:
            use_until = 0
        elif dist_cf < abs(self.until_d_th) * 2.5:
            use_until = 1
            self.until_v_th = self.until_v_th_ref / self.until_lanewidth * lanewidth_cur
            self.until_d_th = self.until_d_th_ref / self.until_lanewidth * lanewidth_cur
        else:
            use_until = 0

        is_error1, rmin_l_mod = 0, 0.0
        for nidx_trial in range(0, 3):
            print("TRIAL: {:d}".format(nidx_trial + 1))
            is_error1, x_out, u_out, is_error1_until = self.solve_by_mip(rmin_l_d + rmin_l_mod, rmin_l_u + rmin_l_mod,
                                                                         rmin_c_cf, rmin_c_rest, rmin_speed, rmin_until,
                                                                         use_until=use_until)

            if is_error1 == 0:
                break
            else:
                if is_error1_until == 1:
                    use_until = 0
                else:
                    rmin_l_mod = rmin_l_mod - 1 * 0.1 * lanewidth_cur

        # if is_error1 == 1:
            # if self.do_plot_debug:
            #     self.plot_debug_ver1()

            # print("SOLVE MPC BY SAMPLING")
            # is_error2, x_out, u_out = self.solve_by_sampling(rmin_l_d, rmin_l_u, rmin_c_cf, rmin_c_rest, rmin_speed,
            #                                                  rmin_until, use_until=0)

        if is_error1 == 1:
            print("FAILED TO SOLVE")
            xinit_conv_r = np.reshape(self.xinit_conv, (1, -1))
            x_out = np.tile(xinit_conv_r, [self.h + 1, 1])
            u_out = np.zeros((self.h, self.dim_u), dtype=np.float32)

        x_out_revert = np.zeros((self.h + 1, self.dim_x), dtype=np.float32)
        x_out_revert[:, 0:2] = get_rotated_pnts_rt(x_out[:, 0:2], +cp2rotate, +theta2rotate)
        x_out_revert[:, 2] = angle_handle(x_out[:, 2] + theta2rotate)
        x_out_revert[:, 3] = x_out[:, 3]

        return x_out, u_out, x_out_revert

    # SOLVE BY MIXED INTEGER PROGRAMMING ------------------------------------------------------------------------------#
    def solve_by_mip(self, rmin_l_d, rmin_l_u, rmin_c_cf, rmin_c_rest, rmin_speed, rmin_until, use_until=1):

        self.reset_opt_model()  # Reset model

        # GET ROBUSTNESS SLACKNESS ------------------------------------------------------------------------------------#
        rmin_l_d_c, rmin_l_u_c, rmin_c_cf_c, rmin_c_rest_c, rmin_speed_c = \
            self.compute_robustness_slackness_init(rmin_l_d, rmin_l_u, rmin_c_cf, rmin_c_rest, rmin_speed)

        # SET VARIABLES -----------------------------------------------------------------------------------------------#
        u_dict = {(i, j) for i in self.list_h for j in self.list_dim_u}
        self.u_vars = self.opt_model.addVars(u_dict, lb=self.params_u_lb, ub=self.params_u_ub, vtype=GRB.CONTINUOUS,
                                             name="u")
        # u_vars = self.opt_model.addVars(u_dict, vtype=GRB.CONTINUOUS, name="u")

        x_dict = {(i, j) for i in self.list_h for j in self.list_dim_x}
        self.x_vars = self.opt_model.addVars(x_dict, lb=self.params_x_lb, ub=self.params_x_ub, vtype=GRB.CONTINUOUS, name="x")

        z_c_dict = {(i, j) for i in self.list_h for j in range(0, 4)}
        r_c_dict = {i for i in self.list_h}
        self.z_c_cf_vars = self.opt_model.addVars(z_c_dict, vtype=GRB.BINARY, name="z_c_cf")
        self.r_c_cf_vars = self.opt_model.addVars(r_c_dict, vtype=GRB.CONTINUOUS, name="r_c_cf")

        self.z_c_ov_vars, self.r_c_ov_vars = [], []
        for nidx_oc in range(0, self.maxnum_oc - 1):
            txt_labelz, txt_labelr = "z_c_" + str(nidx_oc), "r_c_" + str(nidx_oc)
            z_c_ov_vars_ = self.opt_model.addVars(z_c_dict, vtype=GRB.BINARY, name=txt_labelz)
            r_c_ov_vars_ = self.opt_model.addVars(r_c_dict, vtype=GRB.CONTINUOUS, name=txt_labelr)
            self.z_c_ov_vars.append(z_c_ov_vars_)
            self.r_c_ov_vars.append(r_c_ov_vars_)

        if use_until == 1:
            self.r_until = self.opt_model.addVar(vtype=GRB.CONTINUOUS, name="r_until")
            self.r_until_and1 = self.opt_model.addVar(vtype=GRB.CONTINUOUS, name="r_until_and1")
            until_dict0_z = {i for i in range(0, 2)}
            self.z_until_and1 = self.opt_model.addVars(until_dict0_z, vtype=GRB.BINARY, name="z_until_and1")
            self.z_until_and2 = self.opt_model.addVars(until_dict0_z, vtype=GRB.BINARY, name="z_until_and2")

            until_dict1_z = {(i, j) for i in range(0, self.until_t_b - self.until_t_a) for j in range(0, 2)}
            until_dict1_r = {i for i in range(0, self.until_t_b - self.until_t_a)}
            self.z_until_and_hist = self.opt_model.addVars(until_dict1_z, vtype=GRB.BINARY, name="z_until_and_hist")
            self.r_until_and_hist = self.opt_model.addVars(until_dict1_r, vtype=GRB.CONTINUOUS, name="r_until_and_hist")
            until_dict2_z = {(i, j) for i in range(0, self.until_t_b + 1 - self.until_t_a) for j in range(0, 2)}
            until_dict2_r = {i for i in range(0, self.until_t_b + 1 - self.until_t_a)}
            self.z_until_or_hist = self.opt_model.addVars(until_dict2_z, vtype=GRB.BINARY, name="z_until_or_hist")
            self.r_until_or_hist = self.opt_model.addVars(until_dict2_r, vtype=GRB.CONTINUOUS, name="r_until_or_hist")

            alw_dic_z = {i for i in range(0, self.until_t_a - self.until_t_s)}
            self.z_until_alw = self.opt_model.addVars(alw_dic_z, vtype=GRB.BINARY, name="z_until_alw")
            self.r_until_alw = self.opt_model.addVar(vtype=GRB.CONTINUOUS, name="r_until_alw")

            ev_dic_z = {i for i in range(0, self.until_t_b + 1 - self.until_t_a)}
            self.z_until_ev = self.opt_model.addVars(ev_dic_z, vtype=GRB.BINARY, name="z_until_ev")
            self.r_until_ev = self.opt_model.addVar(vtype=GRB.CONTINUOUS, name="r_until_ev")

        self.opt_model.update()  # Update model

        # SET OBJECTIVE -----------------------------------------------------------------------------------------------#
        cost_opt = 0.0
        for nidx_d in range(0, self.dim_x):
            cost_opt += (self.x_vars[(self.h - 1, nidx_d)] - float(self.xgoal_conv[nidx_d])) * \
                        (self.x_vars[(self.h - 1, nidx_d)] - float(self.xgoal_conv[nidx_d])) * float(self.c_end[nidx_d])

        for nidx_h in range(0, self.h):
            for nidx_d in range(0, self.dim_u):
                cost_opt += (self.u_vars[(nidx_h, nidx_d)]) * (self.u_vars[(nidx_h, nidx_d)]) * float(self.c_u[nidx_d])

        # for nidx_h in range(0, self.h - 1):
        #     cost_opt += (x_vars[(nidx_h, 3)] - float(self.v_ref)) * (x_vars[(nidx_h, 3)] - float(self.v_ref)) * 0.1

        self.opt_model.setObjective(cost_opt, GRB.MINIMIZE)

        # SET CONSTRAINTS ---------------------------------------------------------------------------------------------#
        # Set constraints (dynamic)
        self.set_dynamic_constraints()

        # Set constraints (lane)
        self.set_lane_constraints(rmin_l_d_c, rmin_l_u_c)

        # Set constraints (collision)
        self.set_collision_constraints(self.x_vars, self.z_c_cf_vars, self.r_c_cf_vars, rmin_c_cf_c, self.traj_ov_cf,
                                       self.size_ov_cf, "c_cf")

        for nidx_oc in range(0, self.maxnum_oc - 1):  # [id_lf, id_lr, id_rf, id_rr, id_cr]
            txt_label = "c_ov" + str(nidx_oc)
            self.set_collision_constraints(self.x_vars, self.z_c_ov_vars[nidx_oc], self.r_c_ov_vars[nidx_oc],
                                           rmin_c_rest_c[nidx_oc], self.traj_ov[nidx_oc], self.size_ov[nidx_oc],
                                           txt_label)

        # Set constraints (speed)
        self.set_speed_constraints(rmin_speed_c)

        # Set constraints (until)
        if use_until == 1:
            self.set_until_constraint(rmin_until)

        # OPTIMIZE MODEL ----------------------------------------------------------------------------------------------#
        self.opt_model.optimize()
        is_error = self.check_opt_status()

        is_error_until = 0
        if is_error == 1:
            self.opt_model.computeIIS()  # Computes Irreducible Incosistent Subsytems (Not Vital)

            print('\nThe following constraint(s) cannot be satisfied:')
            for c in self.opt_model.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
                    if "until" in c.constrName:
                        is_error_until = 1

            # if self.do_plot_debug:
            #     self.plot_debug_ver1()
            # print("STOP")

            # self.opt_model.feasRelaxS(1, False, False, True)  # Relaxes the problem so that it becomes feasible
            # self.opt_model.feasRelaxS(1, False, True, False)  # Relaxes the problem so that it becomes feasible
            # self.opt_model.optimize()  # Give the optimize command once again!
            # is_error = self.check_opt_status()

        # Set output
        if is_error == 1:
            xinit_conv_r = np.reshape(self.xinit_conv, (1, -1))
            x_out = np.tile(xinit_conv_r, [self.h + 1, 1])
            u_out = np.zeros((self.h, self.dim_u), dtype=np.float32)
        else:
            x_out = np.zeros((self.h + 1, self.dim_x), dtype=np.float32)
            u_out = np.zeros((self.h, self.dim_u), dtype=np.float32)
            x_out[0, :] = self.xinit_conv
            for nidx_h in range(0, self.h):
                x_out[nidx_h + 1, :] = [self.x_vars[(nidx_h, 0)].X, self.x_vars[(nidx_h, 1)].X,
                                        self.x_vars[(nidx_h, 2)].X, self.x_vars[(nidx_h, 3)].X]
                u_out[nidx_h, :] = [self.u_vars[(nidx_h, 0)].X, self.u_vars[(nidx_h, 1)].X]

            if use_until == 1:
                print("r_until: {:.3f}".format(self.r_until.X))

        return is_error, x_out, u_out, is_error_until

    # Check optimization status
    def check_opt_status(self):
        is_error = 0
        if self.opt_model.status == GRB.Status.OPTIMAL:
            print('Optimal objective: %g' % self.opt_model.objVal)
        elif self.opt_model.status == GRB.Status.INF_OR_UNBD:
            print('Model is infeasible or unbounded')
            is_error = 1
        elif self.opt_model.status == GRB.Status.INFEASIBLE:
            print('Model is infeasible')
            is_error = 1
        elif self.opt_model.status == GRB.Status.UNBOUNDED:
            print('Model is unbounded')
            is_error = 1
        else:
            print('Optimization ended with status %d' % self.opt_model.status)
        return is_error

    # Set dynamic constraints
    def set_dynamic_constraints(self):
        dt = float(self.dt)
        kappa1, kappa2 = float(self.kappa[0]), float(self.kappa[1])
        theta_ref, v_ref = float(self.xinit_conv[2]), float(self.xinit_conv[3])
        cos_ref, sin_ref = float(math.cos(theta_ref)), float(math.sin(theta_ref))

        xinit_conv_x, xinit_conv_y, xinit_conv_theta, xinit_conv_v = \
            float(self.xinit_conv[0]), float(self.xinit_conv[1]), float(self.xinit_conv[2]), float(self.xinit_conv[3])

        a02, a03 = -1 * v_ref * sin_ref * dt, cos_ref * dt
        a12, a13 = v_ref * cos_ref * dt, sin_ref * dt

        b0 = v_ref * kappa1 * dt
        b1 = kappa2 * dt
        c0 = v_ref * sin_ref * dt * theta_ref
        c1 = -v_ref * cos_ref * dt * theta_ref

        rhs00 = xinit_conv_x + a02 * xinit_conv_theta + a03 * xinit_conv_v + c0
        rhs01 = xinit_conv_y + a12 * xinit_conv_theta + a13 * xinit_conv_v + c1

        for nidx_h in range(0, self.h):
            if nidx_h == 0:
                self.opt_model.addConstr(self.x_vars[(nidx_h, 0)] == rhs00, "dyn0_{:d}".format(nidx_h))
                self.opt_model.addConstr(self.x_vars[(nidx_h, 1)] == rhs01, "dyn1_{:d}".format(nidx_h))
                self.opt_model.addConstr(self.x_vars[(nidx_h, 2)] == xinit_conv_theta + b0 * self.u_vars[(nidx_h, 0)],
                                         "dyn2_{:d}".format(nidx_h))
                self.opt_model.addConstr(self.x_vars[(nidx_h, 3)] == xinit_conv_v + b1 * self.u_vars[(nidx_h, 1)],
                                         "dyn3_{:d}".format(nidx_h))
            else:
                txt_dyn_tmp1 = "dyn0_{:d}".format(nidx_h)
                self.opt_model.addConstr(self.x_vars[(nidx_h, 0)] == self.x_vars[(nidx_h - 1, 0)] +
                                         a02 * self.x_vars[(nidx_h - 1, 2)] + a03 * self.x_vars[(nidx_h - 1, 3)] +
                                         c0, txt_dyn_tmp1)
                txt_dyn_tmp2 = "dyn1_{:d}".format(nidx_h)
                self.opt_model.addConstr(self.x_vars[(nidx_h, 1)] == self.x_vars[(nidx_h - 1, 1)] +
                                         a12 * self.x_vars[(nidx_h - 1, 2)] + a13 * self.x_vars[(nidx_h - 1, 3)] + c1,
                                         txt_dyn_tmp2)
                txt_dyn_tmp3 = "dyn2_{:d}".format(nidx_h)
                self.opt_model.addConstr(self.x_vars[(nidx_h, 2)] == self.x_vars[(nidx_h - 1, 2)] +
                                         b0 * self.u_vars[(nidx_h, 0)], txt_dyn_tmp3)
                txt_dyn_tmp4 = "dyn3_{:d}".format(nidx_h)
                self.opt_model.addConstr(self.x_vars[(nidx_h, 3)] == self.x_vars[(nidx_h - 1, 3)] +
                                         b1 * self.u_vars[(nidx_h, 1)], txt_dyn_tmp4)

    # Set lane constraints
    def set_lane_constraints(self, rmin_down, rmin_up):
        # rmin_down, rmin_up: (ndarray) robustness slackness (dim = N x 1)

        a_d = float(-math.tan(self.rad_l))
        b_d = {i: float(rmin_down[i] - math.tan(self.rad_l) * self.cp_l_d[0] + self.cp_l_d[1]) for i in self.list_h}

        self.opt_model.addConstrs((a_d * self.x_vars[(i, 0)] + self.x_vars[(i, 1)] >= b_d[i] for i in range(0, self.h)),
                                  name='ld')

        a_u = float(math.tan(self.rad_l))
        # b_u = float(rmin_up + math.tan(self.rad_l) * self.cp_l_u[0] - self.cp_l_u[1])
        b_u = {i: float(rmin_up[i] + math.tan(self.rad_l) * self.cp_l_u[0] - self.cp_l_u[1]) for i in self.list_h}
        self.opt_model.addConstrs((a_u * self.x_vars[(i, 0)] - self.x_vars[(i, 1)] >= b_u[i] for i in range(0, self.h)),
                                  name='lu')

        # for nidx_h in range(0, self.h):
        #     y_down = math.tan(self.rad_l) * (x_vars[(nidx_h, 0)] - self.cp_l_d[0]) + self.cp_l_d[1]
        #     self.opt_model.addConstr(x_vars[(nidx_h, 1)] - y_down >= rmin_down, name="l_d{:d}".format(nidx_h))
        #
        #     y_up = math.tan(self.rad_l) * (x_vars[(nidx_h, 0)] - self.cp_l_u[0]) + self.cp_l_u[1]
        #     self.opt_model.addConstr(y_up - x_vars[(nidx_h, 1)] >= rmin_up, name="l_u{:d}".format(nidx_h))

    # Set collision constraints
    def set_collision_constraints(self, x_vars, z_vars, r_vars, rmin, traj_in, size_in, txt_label, param_x_1=1.0,
                                  param_x_2=0.125, param_x_3=1.25, param_y_1=0.1, param_y_2=0.125, param_y_3=0.1,
                                  param_y_4=0.2, lanewidth_cur=3.28):

        diff_traj_tmp = traj_in[range(1, self.h+1), 0:2] - traj_in[range(0, self.h), 0:2]
        dist_traj_tmp = np.sqrt(np.sum(diff_traj_tmp * diff_traj_tmp, axis=1))

        dist_traj = 0
        for nidx_h in range(0, self.h):
            x_oc_sel = traj_in[nidx_h + 1, :]

            # Modify size w.r.t. distance
            dist_traj = min(dist_traj + dist_traj_tmp[nidx_h] * lanewidth_cur / 3.28, 1000.0)
            mod_size_x = min(param_x_1 * (math.exp(param_x_2 * dist_traj) - 1.0), param_x_3)
            mod_size_y = param_y_1 + min(param_y_2 * (math.exp(param_y_3 * dist_traj) - 1.0), param_y_4)
            rx_oc = (size_in[nidx_h + 1, 0] + self.rx) / 2.0 + mod_size_x
            ry_oc = (size_in[nidx_h + 1, 1] + self.ry) / 2.0 + mod_size_y

            r_b_oc0, r_b_oc1 = float(-x_oc_sel[0] - rx_oc), float(x_oc_sel[0] - rx_oc)
            r_b_oc2, r_b_oc3 = float(-x_oc_sel[1] - ry_oc), float(x_oc_sel[1] - ry_oc)

            self.opt_model.addConstr(z_vars[(nidx_h, 0)] + z_vars[(nidx_h, 1)] + z_vars[(nidx_h, 2)] +
                                     z_vars[(nidx_h, 3)] == 1, name="{:s}_z{:d}".format(txt_label, nidx_h))

            self.opt_model.addConstr(r_vars[nidx_h] - x_vars[(nidx_h, 0)] >= r_b_oc0,
                                     name="{:s}_0{:d}".format(txt_label, nidx_h))
            self.opt_model.addConstr(r_vars[nidx_h] + x_vars[(nidx_h, 0)] >= r_b_oc1,
                                     name="{:s}_1{:d}".format(txt_label, nidx_h))
            self.opt_model.addConstr(r_vars[nidx_h] - x_vars[(nidx_h, 1)] >= r_b_oc2,
                                     name="{:s}_2{:d}".format(txt_label, nidx_h))
            self.opt_model.addConstr(r_vars[nidx_h] + x_vars[(nidx_h, 1)] >= r_b_oc3,
                                     name="{:s}_3{:d}".format(txt_label, nidx_h))

            self.opt_model.addConstr(r_vars[nidx_h] - x_vars[(nidx_h, 0)] - self.M * z_vars[(nidx_h, 0)] - r_b_oc0 +
                                     self.M >= 0.0, name="{:s}_4l{:d}".format(txt_label, nidx_h))
            self.opt_model.addConstr(-r_vars[nidx_h] + x_vars[(nidx_h, 0)] - self.M * z_vars[(nidx_h, 0)] + r_b_oc0 +
                                     self.M >= 0.0, name="{:s}_4r{:d}".format(txt_label, nidx_h))

            self.opt_model.addConstr(r_vars[nidx_h] + x_vars[(nidx_h, 0)] - self.M * z_vars[(nidx_h, 1)] - r_b_oc1 +
                                     self.M >= 0.0, name="{:s}_5l{:d}".format(txt_label, nidx_h))
            self.opt_model.addConstr(-r_vars[nidx_h] - x_vars[(nidx_h, 0)] - self.M * z_vars[(nidx_h, 1)] + r_b_oc1 +
                                     self.M >= 0.0, name="{:s}_5r{:d}".format(txt_label, nidx_h))

            self.opt_model.addConstr(r_vars[nidx_h] - x_vars[(nidx_h, 1)] - self.M * z_vars[(nidx_h, 2)] - r_b_oc2 +
                                     self.M >= 0.0, name="{:s}_6l{:d}".format(txt_label, nidx_h))
            self.opt_model.addConstr(-r_vars[nidx_h] + x_vars[(nidx_h, 1)] - self.M * z_vars[(nidx_h, 2)] + r_b_oc2 +
                                     self.M >= 0.0, name="{:s}_6r{:d}".format(txt_label, nidx_h))

            self.opt_model.addConstr(r_vars[nidx_h] + x_vars[(nidx_h, 1)] - self.M * z_vars[(nidx_h, 3)] - r_b_oc3 +
                                     self.M >= 0.0, name="{:s}_7l{:d}".format(txt_label, nidx_h))
            self.opt_model.addConstr(-r_vars[nidx_h] - x_vars[(nidx_h, 1)] - self.M * z_vars[(nidx_h, 3)] + r_b_oc3 +
                                     self.M >= 0.0, name="{:s}_7r{:d}".format(txt_label, nidx_h))

            self.opt_model.addConstr(r_vars[nidx_h] >= float(rmin[nidx_h]), name="{:s}_8{:d}".format(txt_label, nidx_h))

    # Set speed constraints
    def set_speed_constraints(self, rmin_speed):
        for nidx_h in range(0, self.h):
            self.opt_model.addConstr(float(self.v_th) - self.x_vars[(nidx_h, 3)] >= float(rmin_speed[nidx_h]),
                                     name="speed_0{:d}".format(nidx_h))

    # Set until-logic constraint
    def set_until_constraint(self, rmin_until):
        # Until
        for nidx_t in range(self.until_t_a, self.until_t_b + 1):
            self.set_until_constraint_ubuntil(nidx_t)

        # Always
        self.set_until_constraint_alw()

        # Eventually
        self.set_until_constraint_ev()

        # and1 (always + eventually)
        self.opt_model.addConstr(self.r_until_and1 <= self.r_until_alw, name="until_and1_r0")
        self.opt_model.addConstr(self.r_until_and1 <= self.r_until_ev, name="until_and1_r1")

        self.opt_model.addConstr(self.z_until_and1[0] + self.z_until_and1[1] == 1, name="until_and1_M")
        self.opt_model.addConstr(self.r_until_alw - (1 - self.z_until_and1[0]) * self.M <= self.r_until_and1,
                                 name="until_and1_M0")
        self.opt_model.addConstr(self.r_until_and1 <= self.r_until_alw + (1 - self.z_until_and1[0]) * self.M,
                                 name="until_and1_M1")
        self.opt_model.addConstr(self.r_until_ev - (1 - self.z_until_and1[1]) * self.M <= self.r_until_and1,
                                 name="until_and1_M2")
        self.opt_model.addConstr(self.r_until_and1 <= self.r_until_ev + (1 - self.z_until_and1[1]) * self.M,
                                 name="until_and1_M3")

        # and2 (+ until)
        self.opt_model.addConstr(self.r_until <= self.r_until_and1, name="until_and2_r0")
        self.opt_model.addConstr(self.r_until <= self.r_until_or_hist[0], name="until_and2_r1")

        self.opt_model.addConstr(self.z_until_and2[0] + self.z_until_and2[1] == 1, name="until_and2_M")
        self.opt_model.addConstr(self.r_until_and1 - (1 - self.z_until_and2[0]) * self.M <= self.r_until,
                                 name="until_and2_M0")
        self.opt_model.addConstr(self.r_until <= self.r_until_and1 + (1 - self.z_until_and2[0]) * self.M,
                                 name="until_and2_M1")
        self.opt_model.addConstr(self.r_until_or_hist[0] - (1 - self.z_until_and2[1]) * self.M <= self.r_until,
                                 name="until_and2_M2")
        self.opt_model.addConstr(self.r_until <= self.r_until_or_hist[0] + (1 - self.z_until_and2[1]) * self.M,
                                 name="until_and2_M3")

        self.opt_model.addConstr(self.r_until >= rmin_until, name="until_final")

    def set_until_constraint_ubuntil(self, idx_t):
        idx_t_m = idx_t - self.until_t_a

        r_phi1 = float(self.until_v_th) - self.x_vars[(idx_t, 3)]
        r_phi2 = self.x_vars[(idx_t, 0)] - float(self.traj_ov_cf[idx_t + 1, 0]) + \
                 (self.rx + float(self.size_ov_cf[idx_t + 1, 0])) / 2.0 + float(self.until_d_th)
        if idx_t == self.until_t_b:
            # (or)
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] >= r_phi1,
                                     name="until_hist_or_r0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] >= r_phi2,
                                     name="until_hist_or_r1{:d}".format(idx_t_m))

            self.opt_model.addConstr(self.z_until_or_hist[(idx_t_m, 0)] + self.z_until_or_hist[(idx_t_m, 1)] == 1,
                                     name="until_hist_or_M{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi1 - (1 - self.z_until_or_hist[(idx_t_m, 0)]) * self.M <=
                                     self.r_until_or_hist[idx_t_m], name="until_hist_or_M0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] <= r_phi1 + (1 - self.z_until_or_hist[(idx_t_m, 0)])
                                     * self.M, name="until_hist_or_M1{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi2 - (1 - self.z_until_or_hist[(idx_t_m, 1)]) * self.M
                                     <= self.r_until_or_hist[idx_t_m], name="until_hist_or_M2{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] <= r_phi2 + (1 - self.z_until_or_hist[(idx_t_m, 1)])
                                     * self.M, name="until_hist_or_M3{:d}".format(idx_t_m))
        else:
            # (and)
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] <= r_phi1,
                                     name="until_hist_and_r0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] <= self.r_until_or_hist[idx_t_m + 1],
                                     name="until_hist_and_r1{:d}".format(idx_t_m))

            self.opt_model.addConstr(self.z_until_and_hist[(idx_t_m, 0)] + self.z_until_and_hist[(idx_t_m, 1)] == 1,
                                     name="until_hist_and_M{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi1 - (1 - self.z_until_and_hist[(idx_t_m, 0)]) * self.M <=
                                     self.r_until_and_hist[idx_t_m], name="until_hist_and_M0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] <= r_phi1 + (1 - self.z_until_and_hist[(idx_t_m, 0)]) * self.M,
                                     name="until_hist_and_M1{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m + 1] - (1 - self.z_until_and_hist[(idx_t_m, 1)]) * self.M
                                     <= self.r_until_and_hist[idx_t_m], name="until_hist_and_M2{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] <= self.r_until_or_hist[idx_t_m + 1] +
                                     (1 - self.z_until_and_hist[(idx_t_m, 1)]) * self.M, name="until_hist_and_M3{:d}".format(idx_t_m))

            # (or)
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] >= r_phi2,
                                     name="until_hist_or_r0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] >= self.r_until_and_hist[idx_t_m],
                                     name="until_hist_or_r1{:d}".format(idx_t_m))

            self.opt_model.addConstr(self.z_until_or_hist[(idx_t_m, 0)] + self.z_until_or_hist[(idx_t_m, 1)] == 1,
                                     name="until_hist_or_M{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi2 - (1 - self.z_until_or_hist[(idx_t_m, 0)]) * self.M <=
                                     self.r_until_or_hist[idx_t_m], name="until_hist_or_M0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] <= r_phi2 + (1 - self.z_until_or_hist[(idx_t_m, 0)]) * self.M,
                                     name="until_hist_or_M1{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] - (1 - self.z_until_or_hist[(idx_t_m, 1)]) * self.M
                                     <= self.r_until_or_hist[idx_t_m], name="until_hist_or_M2{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] <= self.r_until_and_hist[idx_t_m] +
                                     (1 - self.z_until_or_hist[(idx_t_m, 1)]) * self.M,
                                     name="until_hist_or_M3{:d}".format(idx_t_m))

    def set_until_constraint_alw(self):
        self.opt_model.addConstr(quicksum(self.z_until_alw) == 1, name="until_alw_M")
        for nidx_t in range(self.until_t_s, self.until_t_a):
            idx_t_m = nidx_t - self.until_t_s
            r_phi1 = float(self.until_v_th) - self.x_vars[(nidx_t, 3)]

            self.opt_model.addConstr(self.r_until_alw <= r_phi1, name="until_alw_r{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi1 - (1 - self.z_until_alw[idx_t_m]) * self.M <= self.r_until_alw,
                                     name="until_alw_M0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_alw <= r_phi1 + (1 - self.z_until_alw[idx_t_m]) * self.M,
                                     name="until_alw_M1{:d}".format(idx_t_m))

    def set_until_constraint_ev(self):
        self.opt_model.addConstr(quicksum(self.z_until_ev) == 1, name="until_ev_M")
        for nidx_t in range(self.until_t_a, self.until_t_b + 1):
            idx_t_m = nidx_t - self.until_t_a
            r_phi2 = self.x_vars[(nidx_t, 0)] - float(self.traj_ov_cf[nidx_t + 1, 0]) + \
                     (self.rx + float(self.size_ov_cf[nidx_t + 1, 0])) / 2.0 + float(self.until_d_th)

            self.opt_model.addConstr(self.r_until_ev >= r_phi2, name="until_ev_r{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi2 - (1 - self.z_until_ev[idx_t_m]) * self.M <= self.r_until_ev,
                                     name="until_ev_M0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_ev <= r_phi2 + (1 - self.z_until_ev[idx_t_m]) * self.M,
                                     name="until_ev_M1{:d}".format(idx_t_m))

    # SOLVE BY SAMPLING -----------------------------------------------------------------------------------------------#
    def solve_by_sampling(self, rmin_l_d, rmin_l_u, rmin_c_cf, rmin_c_rest, rmin_speed, rmin_until, use_until=1):
        u_w_set = np.linspace(self.u_lb[0], self.u_ub[0], num=21)  # Set of angular-velocities
        u_a_set = np.linspace(self.u_lb[1], self.u_ub[1], num=21)  # Set of accelerations
        u_set = get_control_set_naive(u_w_set, u_a_set)  # Set of control

        len_u_set = u_set.shape[0]

        traj_sampled = []
        traj_sampled_valid = []
        u_sampled_valid = []
        traj_sampled_invalid = []
        for nidx_u in range(0, len_u_set):
            u_sel = u_set[nidx_u, :]
            traj_sampled_tmp = self.get_trajectory_naive(u_sel, self.h, is_dyn_linear=1)
            traj_sampled.append(traj_sampled_tmp)

            idx_h_ev = np.arange(1, self.h + 1)
            idx_h_ov = np.arange(1, self.h + 1)
            r_l_down, r_l_up, r_c_cf, r_c_rest, r_c_rest_array, r_speed, r_until = \
                compute_robustness(traj_sampled_tmp[idx_h_ev, 0:3], traj_sampled_tmp[idx_h_ev, 3], [self.rx, self.ry],
                                   self.cp_l_d, self.cp_l_u, self.rad_l, self.traj_ov_cf, self.size_ov_cf, self.traj_ov,
                                   self.size_ov, idx_h_ov, self.v_th, self.until_t_s, self.until_t_a, self.until_t_b,
                                   self.until_v_th, self.until_d_th)

            idx_sel_1_1 = np.where(traj_sampled_tmp[:, 3] < self.v_range[0])
            idx_sel_1_1 = idx_sel_1_1[0]
            idx_sel_1_2 = np.where(traj_sampled_tmp[:, 3] > self.v_range[1])
            idx_sel_1_2 = idx_sel_1_2[0]
            idx_invalid1 = np.concatenate([idx_sel_1_1, idx_sel_1_2])
            idx_invalid1 = np.reshape(idx_invalid1, -1)
            idx_invalid1 = np.unique(idx_invalid1)
            if len(idx_invalid1) > 0:
                is_invalid = 1
            else:
                is_invalid = (r_l_down < rmin_l_d) or (r_l_up < rmin_l_u) or (r_c_cf < rmin_c_cf) or \
                             (r_c_rest < rmin_c_rest) or (r_speed < rmin_speed)
                if use_until == 1:
                    is_invalid = is_invalid or (r_until < rmin_until)

            if is_invalid == 1:
                traj_sampled_invalid.append(traj_sampled_tmp)
            else:
                traj_sampled_valid.append(traj_sampled_tmp)
                u_sel_r = np.reshape(u_sel, (1, -1))
                u_in = np.tile(u_sel_r, (self.h, 1))
                u_sampled_valid.append(u_in)

        if len(traj_sampled_valid) > 0:
            is_error = 0
            cost_array = np.zeros((len(traj_sampled_valid), ), dtype=np.float32)
            for nidx_traj in range(0, len(traj_sampled_valid)):
                traj_sampled_valid_sel = traj_sampled_valid[nidx_traj]
                u_sampled_valid_sel = u_sampled_valid[nidx_traj]
                cost_xend, cost_u = self.compute_cost(traj_sampled_valid_sel, u_sampled_valid_sel)
                cost_array[nidx_traj] = np.sum(cost_xend) + np.sum(cost_u)

            idx_min = np.argmin(cost_array)
            x_out = traj_sampled_valid[idx_min]
            u_out = u_sampled_valid[idx_min]
        else:
            is_error = 1
            x_out, u_out = [], []
            print("Number of valid samples is zero!")

        return is_error, x_out, u_out

    def get_trajectory_naive(self, u, horizon, is_dyn_linear=1):
        # u: (ndarray or list) angular-velocity, acceleration (dim = 2)
        # horizon: (scalar) trajectory horizon
        # is_dyn_linear: (boolean) whether to use linear dynamic

        traj = np.zeros((horizon + 1, 4), dtype=np.float64)
        traj[0, :] = [self.xinit_conv[0], self.xinit_conv[1], self.xinit_conv[2], self.xinit_conv[3]]

        for nidx_h in range(1, horizon + 1):
            s_prev = traj[nidx_h - 1, :]
            s_new = self.get_next_state(s_prev, traj[0, :], u[0], u[1], is_dyn_linear=is_dyn_linear)
            traj[nidx_h, :] = s_new

        traj = traj.astype(dtype=np.float32)
        return traj

    # Get next state
    def get_next_state(self, s_cur, s_ref, w, a, is_dyn_linear=1):
        # s_cur: (list) x, y, theta, v (dim = 4)
        # s_ref: (list) reference - x, y, theta, v (dim = 4)
        # w, a: (scalar) angular-velocity, acceleration
        # is_dyn_linear: (boolean) whether to use linear dynamic

        if is_dyn_linear == 1:
            theta_ref, v_ref = s_ref[2], s_ref[3]
            sin_ref, cos_ref = math.sin(theta_ref), math.cos(theta_ref)
            x_new = s_cur[0] + (-v_ref * sin_ref * self.dt) * s_cur[2] + (cos_ref * self.dt) * s_cur[3] + \
                    (v_ref * sin_ref * self.dt * theta_ref)
            y_new = s_cur[1] + (v_ref * cos_ref * self.dt) * s_cur[2] + (sin_ref * self.dt) * s_cur[3] + \
                    (-v_ref * cos_ref * self.dt * theta_ref)
            theta_new = s_cur[2] + s_cur[3] * self.kappa[0] * w * self.dt
            lv_new = s_cur[3] + self.kappa[1] * a * self.dt
        else:
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

    # PLOT FOR DEBUG --------------------------------------------------------------------------------------------------#
    def plot_debug_ver1(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.traj_ov_cf[:, 0], self.traj_ov_cf[:, 1], 'r.')
        ax.plot(self.traj_ov_cf[0, 0], self.traj_ov_cf[0, 1], 'ro')

        # [id_lf, id_lr, id_rf, id_rr, id_cr]
        color_tmp = ['b', 'm', 'g', 'y', 'k']
        for nidx_ov in range(0, 5):
            ax.plot(self.traj_ov[nidx_ov][:, 0], self.traj_ov[nidx_ov][:, 1], '.', color=color_tmp[nidx_ov])
            ax.plot(self.traj_ov[nidx_ov][0, 0], self.traj_ov[nidx_ov][0, 1], 'o', color=color_tmp[nidx_ov])

        ax.plot(self.xinit_conv[0], self.xinit_conv[1], 'cs')
        ax.plot(self.xgoal_conv[0], self.xgoal_conv[1], 'bx')

        for nidx_dd in range(0, self.h + 1, 3):
            x_tmp, y_tmp = self.traj_ov_cf[nidx_dd, 0], self.traj_ov_cf[nidx_dd, 1]
            size_tmp = self.size_ov_cf[nidx_dd, :]

            box_tmp = get_box_pnts(x_tmp, y_tmp, 0.0, size_tmp[0], size_tmp[1])
            ax.plot(box_tmp[:, 0], box_tmp[0:, 1], '-')

        for nidx_dd in range(0, self.h + 1, 3):
            x_tmp, y_tmp = self.traj_ov[4][nidx_dd, 0], self.traj_ov[4][nidx_dd, 1]
            size_tmp = self.size_ov[4][nidx_dd, :]

            box_tmp = get_box_pnts(x_tmp, y_tmp, 0.0, size_tmp[0], size_tmp[1])
            ax.plot(box_tmp[:, 0], box_tmp[0:, 1], '-')

        ax.plot(self.cpd_conv_hist_ev[range(0, self.cnt_hist_ev), 0], self.cpd_conv_hist_ev[range(0, self.cnt_hist_ev), 1], '.', color=get_rgb("Silver"))
        ax.plot(self.cpu_conv_hist_ev[range(0, self.cnt_hist_ev), 0], self.cpu_conv_hist_ev[range(0, self.cnt_hist_ev), 1], '.', color=get_rgb("Gold"))

        xtmp = np.arange(start=-50, stop=+50, step=0.5)
        len_xtmp = xtmp.shape[0]
        y_lower = self.cp_l_d[1] * np.ones((len_xtmp,), dtype=np.float32)
        y_upper = self.cp_l_u[1] * np.ones((len_xtmp,), dtype=np.float32)
        ax.plot(xtmp, y_lower, '-', color=get_rgb("Silver"))
        ax.plot(xtmp, y_upper, '-', color=get_rgb("Gold"))

        ax.plot(self.xgoal_conv_hist_ev[range(0, self.cnt_hist_ev), 0], self.xgoal_conv_hist_ev[range(0, self.cnt_hist_ev), 1], 'b.')

        box_ev_tmp = get_box_pnts(self.xinit_conv[0], self.xinit_conv[1], 0.0, self.rx, self.ry)
        ax.plot(box_ev_tmp[:, 0], box_ev_tmp[0:, 1], 'b-')

        ax.axis("equal")
        plt.xlim(self.xinit_conv[0] - 50, self.xinit_conv[0] + 50)
        plt.ylim(self.xinit_conv[1] - 50, self.xinit_conv[1] + 50)
        plt.show(block=False)

        plt.pause(0.1)

    def plot_debug_ver2(self, traj_computed, savename):
        import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1)
        fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
        ax.plot(self.traj_ov_cf[:, 0], self.traj_ov_cf[:, 1], 'r.')
        ax.plot(self.traj_ov_cf[0, 0], self.traj_ov_cf[0, 1], 'ro')

        # [id_lf, id_lr, id_rf, id_rr, id_cr]
        color_tmp = ['b', 'm', 'g', 'y', 'k']
        for nidx_ov in range(0, 5):
            ax.plot(self.traj_ov[nidx_ov][:, 0], self.traj_ov[nidx_ov][:, 1], '.', color=color_tmp[nidx_ov])
            ax.plot(self.traj_ov[nidx_ov][0, 0], self.traj_ov[nidx_ov][0, 1], 'o', color=color_tmp[nidx_ov])

        ax.plot(self.xinit_conv[0], self.xinit_conv[1], 'cs')
        ax.plot(self.xgoal_conv[0], self.xgoal_conv[1], 'bx')

        for nidx_dd in range(0, self.h + 1, 3):
            x_tmp, y_tmp = self.traj_ov_cf[nidx_dd, 0], self.traj_ov_cf[nidx_dd, 1]
            size_tmp = self.size_ov_cf[nidx_dd, :]

            box_tmp = get_box_pnts(x_tmp, y_tmp, 0.0, size_tmp[0], size_tmp[1])
            ax.plot(box_tmp[:, 0], box_tmp[0:, 1], '-')

        for nidx_dd in range(0, self.h + 1, 3):
            x_tmp, y_tmp = self.traj_ov[4][nidx_dd, 0], self.traj_ov[4][nidx_dd, 1]
            size_tmp = self.size_ov[4][nidx_dd, :]

            box_tmp = get_box_pnts(x_tmp, y_tmp, 0.0, size_tmp[0], size_tmp[1])
            ax.plot(box_tmp[:, 0], box_tmp[0:, 1], '-')

        xtmp = np.arange(start=-30, stop=+30, step=0.5)
        len_xtmp = xtmp.shape[0]
        y_lower = self.cp_l_d[1] * np.ones((len_xtmp,), dtype=np.float32)
        y_upper = self.cp_l_u[1] * np.ones((len_xtmp,), dtype=np.float32)
        ax.plot(xtmp, y_lower, '-', color=get_rgb("Silver"))
        ax.plot(xtmp, y_upper, '-', color=get_rgb("Gold"))

        box_ev_tmp = get_box_pnts(self.xinit_conv[0], self.xinit_conv[1], 0.0, self.rx, self.ry)
        ax.plot(box_ev_tmp[:, 0], box_ev_tmp[0:, 1], 'b-')

        ax.plot(traj_computed[:, 0], traj_computed[:, 1], '.-', color=get_rgb("Royal Blue"))

        ax.axis("equal")
        plt.xlim(self.xinit_conv[0] - 10, self.xinit_conv[0] + 30)
        plt.ylim(self.xinit_conv[1] - 5, self.xinit_conv[1] + 5)
        plt.show(block=False)

        plt.savefig(savename)
        plt.pause(0.1)
        plt.close()



