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
#       state: [x, y, theta, v]
#       control: [w, a]
#           dot(x) = v * cos(theta)
#           dot(y) = v * sin(theta)
#           dot(theta) = v * kappa_1 * w
#           dot(v) = kappa_2 * a

from __future__ import print_function

from core.SimTrack import *
from core.SimScreen import *
from core.SimControl import *

from tracksim_mpcstl.MPCSTL import *


if __name__ == "__main__":

    VIEW_SCREEN = 1  # Whether to view screen or not
    MODE_SCREEN = 1  # 0: plot all track // 1: plot part of track
    SAVE_PIC = 0  # Whether to save pic or not
    PLOT_DEBUG = 1  # Plot for debugging

    np.random.seed(0)  # Set random seed

    # SET TRACK -------------------------------------------------------------------------------------------------------#
    trackname = "US101"
    # trackname = "I80"
    # trackname = "highD_55"
    sim_track = SimTrack(trackname)

    # LOAD VEHICLE DATA -----------------------------------------------------------------------------------------------#
    #       structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    if trackname == "US101":
        data_ov = np.load("../data_vehicle/dv_ngsim_us101_1.npy")
    elif trackname == "I80":
        data_ov = np.load("../data_vehicle/dv_ngsim_i80_1.npy")
    elif "highD" in trackname:
        text_split = trackname.split("_")
        track_num = text_split[-1]
        filename2read = "../data_vehicle/dv_highD_{:s}.npy".format(track_num)
        data_ov = np.load(filename2read)
    else:
        data_ov = np.load("../data_vehicle/dv_ngsim_us101_1.npy")

    id_unique_ = data_ov[:, -1]
    id_unique = np.unique(id_unique_, axis=0)
    id_unique = id_unique.astype(dtype=np.int32)

    t_min, t_max = int(np.amin(data_ov[:, 0])), int(np.amax(data_ov[:, 0]))
    print("t_min: " + str(t_min) + ", t_max: " + str(t_max))

    # SET CONTROL -----------------------------------------------------------------------------------------------------#
    # Initial indexes array for seg & lane
    if trackname == "US101":
        seg_init_array, lane_init_array = [0, 1], [0, 1, 2, 3, 4]
    elif trackname == "I80":
        seg_init_array, lane_init_array = [0], [0, 1, 2]
    elif "highD" in trackname:
        seg_init_array, lane_init_array = [0], np.arange(1, sim_track.num_lane[0], 2)
    else:
        seg_init_array, lane_init_array = [0, 1], [0, 1, 2, 3, 4]

    dt = 0.1  # Time-step
    rx_ev, ry_ev = 4.2, 1.8  # Size of ego-vehicle
    kappa = [0.2, 1]  # Dynamic parameters
    v_ref = 8.0  # Reference (linear) velocity
    v_range = [0, 30]  # Range of (linear) velocity
    sim_control = SimControl(sim_track, dt, rx_ev, ry_ev, kappa, v_ref, v_range)

    t_init = t_min + np.random.randint(int((t_max - t_min) * 4 / 5))  # Starting time
    idx_seg_rand, idx_lane_rand = np.random.randint(len(seg_init_array)), np.random.randint(len(lane_init_array))
    seg_init, lane_init = seg_init_array[idx_seg_rand], lane_init_array[idx_lane_rand]  # Initial indexes

    margin_rx, margin_ry = sim_control.rx, 0.2

    sim_control.set_initial_state(data_ov, t_init, seg_init, lane_init, margin_rx, margin_ry)

    min_vel, max_vel, mean_vel = min(data_ov[:, 4]), max(data_ov[:, 4]), np.mean(data_ov[:, 4])
    print("min-vel: " + str(min_vel) + ", max-vel: " + str(max_vel) + ", mean-vel: " + str(mean_vel))

    # SET MIP-STL -----------------------------------------------------------------------------------------------------#
    act_range = np.array([[-math.pi * 0.2, +math.pi * 0.2], [-100, +100]], dtype=np.float32)
    horizon_ev = 16  # Horizon of trajectory
    horizon_relax = 0  # Horizon of stl-constraint relaxation
    c_end = [10, 10, 0.01, 0.01]  # Cost parameters (end-point)
    c_u = [2e2, 0.1]  # Cost parameters (control)
    dist_ahead = horizon_ev * sim_control.v_ref * 0.15  # Distance to the goal state
    # dist_ahead = horizon_ev * sim_control.v_ref * 0.1  # Distance to the goal state
    v_th = 25  # Velocity threshold

    # Until-logic parameters
    until_t_s, until_t_a, until_t_b = int(horizon_ev / 4), int(horizon_ev / 2), (horizon_ev - 1)
    until_v_th, until_d_th = 12, 6  # 12, 5
    until_lanewidth = 3.6
    sim_mpcstl = MPCSTL(sim_control.dt, sim_control.rx, sim_control.ry, sim_control.kappa, horizon_ev, horizon_relax,
                        act_range[:, 0], act_range[:, 1], c_end, c_u, sim_control.v_ref, sim_control.v_range,
                        dist_ahead, v_th, until_t_s, until_t_a, until_t_b, until_v_th, until_d_th, until_lanewidth,
                        PLOT_DEBUG)

    # SET SCREEN ------------------------------------------------------------------------------------------------------#
    if VIEW_SCREEN == 1:
        if MODE_SCREEN == 0:
            screen_alpha, screen_size = sim_track.screen_alpha_wide, sim_track.screen_size_wide
        else:
            screen_alpha, screen_size = sim_track.screen_alpha_narrow, sim_track.screen_size_narrow

        sim_screen = SimScreen(screen_size, screen_alpha, MODE_SCREEN)
        sim_screen.set_pnts_track_init(sim_track.pnts_poly_track, sim_track.pnts_outer_border_track,
                                       sim_track.pnts_inner_border_track)

        if MODE_SCREEN == 0:
            sim_screen.set_pnts_range(sim_track.pnt_min, sim_track.pnt_max)
        else:
            sim_screen.set_pnts_range_sub(sim_track.pnt_min, sim_track.pnt_max)
            sim_screen.set_screen_height_sub(screen_size[0] * sim_track.screen_sub_height_ratio)
            sim_screen.set_screen_alpha_sub()

    # MAIN: RUN SIM ---------------------------------------------------------------------------------------------------#
    # Used to manage how fast the screen updates
    if VIEW_SCREEN == 1:
        clock = pygame.time.Clock()
        clock_rate = 30
    else:
        clock = []

    for nidx_t in range(t_init, t_max):
        # STEP1: GET CURRENT INFO -------------------------------------------------------------------------------------#
        # Select current vehicle data (w.r.t time)
        s_ev_cur = [sim_control.x_ego, sim_control.y_ego, sim_control.theta_ego, sim_control.v_ego]
        idx_sel_ = np.where(data_ov[:, 0] == nidx_t)
        idx_sel = idx_sel_[0]
        data_ov_cur = data_ov[idx_sel, :]

        # Set indexes of seg & lane
        seg_ev_, lane_ev_ = get_index_seg_and_lane([sim_control.x_ego, sim_control.y_ego], sim_track.pnts_poly_track)
        seg_ev, lane_ev = seg_ev_[0], lane_ev_[0]

        # Set data vehicle ego
        # structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
        data_ev = np.array([nidx_t, sim_control.x_ego, sim_control.y_ego, sim_control.theta_ego,
                            sim_control.v_ego, sim_control.ry, sim_control.rx, seg_ev, lane_ev, -1], dtype=np.float32)

        # Get feature
        f_cur, id_near_cur, pnts_debug_f_ev, pnts_debug_f_ov, rad_center, dist_cf = \
            get_feature(sim_track, data_ev, data_ov_cur, use_intp=1)
        lanewidth_cur = f_cur[3]
        pnt_center, pnt_minleft, pnt_minright = pnts_debug_f_ev[0, :], pnts_debug_f_ev[1, :], pnts_debug_f_ev[2, :]

        # Get near other vehicles
        data_ov_near, data_ov_near_list = get_selected_vehicles_near(data_ov, id_near_cur)
        traj_ov_near_list, size_ov_near_list = get_vehicle_trajectory_near(data_ov_near_list, id_near_cur, nidx_t,
                                                                           horizon_ev, handle_remain=2)
        _, traj_ov_near_plot, size_ov_near_plot = get_vehicle_trajectory(data_ov_near, nidx_t, horizon_ev,
                                                                         handle_remain=1)

        # (CHECK)
        is_collision = check_collision_near(data_ev, data_ov_near, nidx_t)
        if is_collision == 1:
            print("COLLISION!")

        # STEP2: FIND CONTROL -----------------------------------------------------------------------------------------#
        pose_goal = sim_mpcstl.get_goal_state_ver1(sim_control, s_ev_cur)  # Get goal state
        # pose_goal_2 = sim_mpcstl.get_goal_state_ver2(pnt_center, rad_center)  # Get goal state
        # pose_goal = []
        cp2rotate_mpc, theta2rotate_mpc = pnt_center, rad_center

        sim_mpcstl.convert_state(s_ev_cur, pose_goal, cp2rotate_mpc, theta2rotate_mpc)
        sim_mpcstl.get_lane_constraints(pnt_minright, pnt_minleft, rad_center, 0.0, cp2rotate_mpc, theta2rotate_mpc)

        sim_mpcstl.get_collision_constraints(traj_ov_near_list, size_ov_near_list, id_near_cur, cp2rotate_mpc,
                                             theta2rotate_mpc)

        sim_mpcstl.update_history()

        print("[t:{:d}] xinit_conv: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]".
              format(nidx_t, sim_mpcstl.xinit_conv[0], sim_mpcstl.xinit_conv[1], sim_mpcstl.xinit_conv[2],
                     sim_mpcstl.xinit_conv[3]))

        rmin_l = 0.0
        rmin_until = 0.0
        x_out, u_out, x_out_revert = sim_mpcstl.control_by_mpc(rmin_l, rmin_l, 0.0, 0.0, 0.0, rmin_until,
                                                               id_near_cur[4], dist_cf, cp2rotate_mpc, theta2rotate_mpc,
                                                               lanewidth_cur=lanewidth_cur)
        traj_sel_ev = x_out_revert

        idx_h_ev, idx_h_ov = np.arange(1, sim_mpcstl.h + 1), np.arange(1, sim_mpcstl.h + 1)
        r_l_down_out, r_l_up_out, r_c_cf_out, r_c_rest_out, r_c_rest_array_out, r_speed_out, r_until_out = \
            compute_robustness(x_out[idx_h_ev, 0:3], x_out[idx_h_ev, 3], [sim_mpcstl.rx, sim_mpcstl.ry],
                               sim_mpcstl.cp_l_d, sim_mpcstl.cp_l_u, sim_mpcstl.rad_l, sim_mpcstl.traj_ov_cf,
                               sim_mpcstl.size_ov_cf, sim_mpcstl.traj_ov, sim_mpcstl.size_ov, idx_h_ov, sim_mpcstl.v_th,
                               sim_mpcstl.until_t_s, sim_mpcstl.until_t_a, sim_mpcstl.until_t_b, sim_mpcstl.until_v_th,
                               sim_mpcstl.until_d_th)

        cost_xend, cost_u = sim_mpcstl.compute_cost(x_out, u_out)

        print("         xend_conv: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]".
              format(nidx_t, x_out[-1, 0], x_out[-1, 1], x_out[-1, 2], x_out[-1, 3]))
        print("[COST] cost_xend: [{:.3f}, {:.3f}, {:.3f}, {:.3f}], cost_u: [{:.3f}, {:.3f}]".
              format(cost_xend[0], cost_xend[1], cost_xend[2], cost_xend[3], cost_u[0], cost_u[1]))
        print("[ROBUSTNESS] {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".
              format(r_l_down_out, r_l_up_out, r_c_cf_out, r_c_rest_out, r_speed_out, r_until_out))

        # FIND NEXT-STATE
        s_ev_next = traj_sel_ev[1, :]
        # u_next = u_out[0, :]
        # s_ev_next = sim_control.get_next_state(s_ev_cur, u_next[0], u_next[0])

        # STEP3: DRAW -------------------------------------------------------------------------------------------------#
        if VIEW_SCREEN == 1:
            pygame.event.get()

            # Clear screen --------------------------------------------------------------------------------------------#
            # sim_screen.pygame_screen.fill(get_rgb("Gray"))
            sim_screen.draw_background("Light Gray")

            # Set middle point of window
            if MODE_SCREEN == 1:
                pnt_m_plot = np.array([sim_control.x_ego, sim_control.y_ego], dtype=np.float32)
                sim_screen.set_pnts_range_wrt_mean(pnt_m_plot)

            sim_screen.draw_track()
            sim_screen.draw_vehicle_fill(data_ov_cur, id_near_cur, get_rgb("Crimson"))

            sim_screen.draw_target_vehicle_fill(data_ev[1], data_ev[2], data_ev[3], data_ev[6], data_ev[5],
                                                get_rgb("Dodger Blue"))
            sim_screen.draw_target_vehicle_border(s_ev_next[0], s_ev_next[1], s_ev_next[2], data_ev[6], data_ev[5],
                                                  2, get_rgb("Blue"))

            sim_screen.draw_trajectory_array(traj_ov_near_plot, 2, get_rgb("Maroon"))
            sim_screen.draw_vehicle_border_trajectory_array(traj_ov_near_plot, size_ov_near_plot, 5, 1.5,
                                                            get_rgb("Maroon"))

            sim_screen.draw_trajectory(traj_sel_ev, 2.5, get_rgb("Blue"))
            sim_screen.draw_pnts(traj_sel_ev[:, 0:2], 1.2, get_rgb("Cyan"))
            sim_screen.draw_pnts(sim_mpcstl.x_hist_ev[range(0, sim_mpcstl.cnt_hist_ev), 0:2], 1.2,
                                 get_rgb("Dark Turquoise"))

            if sim_mpcstl.do_plot_debug:  # PLOT FOR DEBUGGING
                sim_screen.draw_pnt(cp2rotate_mpc, 5, get_rgb("Yellow Green"))
                sim_screen.draw_trajectory(sim_mpcstl.xgoal_hist_ev[range(0, sim_mpcstl.cnt_hist_ev), 0:2], 2.5,
                                           get_rgb("Green"))

                sim_screen.draw_trajectory(sim_mpcstl.cpd_hist_ev[range(0, sim_mpcstl.cnt_hist_ev), 0:2],
                                           2.5, get_rgb("Silver"))
                sim_screen.draw_trajectory(sim_mpcstl.cpu_hist_ev[range(0, sim_mpcstl.cnt_hist_ev), 0:2],
                                           2.5, get_rgb("Gold"))

                sim_screen.draw_pnts(sim_mpcstl.traj_l_d_rec[:, 0:2], 2, get_rgb("Silver"))
                sim_screen.draw_pnts(sim_mpcstl.traj_l_u_rec[:, 0:2], 2, get_rgb("Gold"))

                for nidx_seg in range(0, len(sim_track.pnts_m_track)):
                    pnts_m_track = sim_track.pnts_m_track[nidx_seg]
                    for nidx_lane in range(0, len(pnts_m_track)):
                        pnts_m_lane = pnts_m_track[nidx_lane]
                        # sim_screen.draw_trajectory(pnts_m_lane[:, 0:2], get_rgb("White"), 1.5)
                        sim_screen.draw_pnts(pnts_m_lane[:, 0:2], 1.5, get_rgb("Deep Pink"))

            if MODE_SCREEN == 1:  # DRAW-SUB
                sim_screen.draw_box_range(get_rgb("Black"), 1.5)
                sim_screen.draw_track_sub(sim_track.pnts_poly_track)
                sim_screen.draw_pnts_sub(data_ov_cur[:, 1:3], 1.0, get_rgb("Salmon"))
                sim_screen.draw_pnt_sub(s_ev_cur[0:2], 2.0, get_rgb("Cyan"))
                sim_screen.draw_pnts_sub(sim_track.pnts_goal[:, 0:2], 1.0, get_rgb("Yellow"))

        # STEP4: DISPLAY ----------------------------------------------------------------------------------------------#
        if VIEW_SCREEN == 1:
            # Get data surface
            data_surface = sim_screen.bgra_surf_to_rgba_string()

            # Create pygame surface
            pygame_surface = pygame.image.frombuffer(data_surface, (sim_screen.screen_size[0],
                                                                    sim_screen.screen_size[1]), 'RGBA')

            # Show pygame surface
            sim_screen.pygame_screen.blit(pygame_surface, (0, 0))
            pygame.display.flip()

            # Limit frame-rate (frames per second)
            clock.tick(clock_rate)

        # SAVE PIC
        if SAVE_PIC == 1:
            savename1 = "%s/t1_%d.png" % ("C:/Users/rllab-khcho0923/Desktop/test", nidx_t - t_init)
            sim_mpcstl.plot_debug_ver2(x_out, savename1)
            if VIEW_SCREEN == 1:
                savename2 = "%s/t2_%d.png" % ("C:/Users/rllab-khcho0923/Desktop/test", nidx_t - t_init)
                pygame.image.save(sim_screen.pygame_screen, savename2)

        # STEP6: UPDATE -----------------------------------------------------------------------------------------------#
        sim_control.update_state(s_ev_next[0], s_ev_next[1], s_ev_next[2], s_ev_next[3])

    print("DONE")
