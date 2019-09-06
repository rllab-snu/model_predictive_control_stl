# TRACK SIMULATION
#   - Naive control (ver1)
#   - pygame-based simulator
#   - use pycairo for graphics

from __future__ import print_function

from __future__ import print_function

import sys
sys.path.insert(0, "../")

from core.SimTrack import *
from core.SimControl import *
from core.SimScreen import *


if __name__ == "__main__":

    VIEW_SCREEN = 1  # Whether to view screen or not
    MODE_SCREEN = 1  # 0: plot all track // 1: plot part of track

    np.random.seed(1)  # Set random seed

    # SET TRACK -------------------------------------------------------------------------------------------------------#
    trackname = "US101"
    # trackname = "I80"
    # trackname = "highD_60"
    sim_track = SimTrack(trackname)

    # LOAD VEHICLE DATA -----------------------------------------------------------------------------------------------#
    #       structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    if trackname == "US101":
        data_ov = np.load("../data_vehicle/dv_ngsim_us101_2.npy")
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
        seg_init_array, lane_init_array = [0], [0, 1]
    else:
        seg_init_array, lane_init_array = [0, 1], [0, 1, 2, 3, 4]

    dt = 0.1  # Time-step
    rx_ev, ry_ev = 4.2, 1.8  # Size of ego-vehicle
    kappa = [0.2, 1]  # Dynamic parameters
    v_ref = 10.0  # Reference (linear) velocity
    v_range = [0, 30]  # Range of (linear) velocity
    sim_control = SimControl(sim_track, dt, rx_ev, ry_ev, kappa, v_ref, v_range)

    t_init = t_min + np.random.randint(int((t_max - t_min) * 4 / 5))  # Starting time
    idx_seg_rand, idx_lane_rand = np.random.randint(len(seg_init_array)), np.random.randint(len(lane_init_array))
    seg_init, lane_init = seg_init_array[idx_seg_rand], lane_init_array[idx_lane_rand]  # Initial indexes
    margin_rx, margin_ry = rx_ev, 0.2

    sim_control.set_initial_state(data_ov, t_init, seg_init, lane_init, margin_rx, margin_ry)

    min_vel, max_vel, mean_vel = min(data_ov[:, 4]), max(data_ov[:, 4]), np.mean(data_ov[:, 4])
    print("min-vel: " + str(min_vel) + ", max-vel: " + str(max_vel) + ", mean-vel: " + str(mean_vel))

    u_w_set = np.linspace(-math.pi * 0.15, +math.pi * 0.15, num=9)  # Set of angular-velocities
    u_a_set = np.linspace(-12.5, +12.5, num=7)  # Set of accelerations
    u_set = get_control_set_naive(u_w_set, u_a_set)  # Set of control
    horizon_ev = 15  # Horizon of trajectory
    dist_ahead = horizon_ev * sim_control.v_ref * 0.15  # Distance to the goal state

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
        data_ev_cur = np.array([nidx_t, sim_control.x_ego, sim_control.y_ego, sim_control.theta_ego,
                                sim_control.v_ego, sim_control.ry, sim_control.rx, seg_ev, lane_ev, -1], dtype=np.float32)

        # Get feature
        f_cur, id_near_cur, pnts_debug_f_ev, pnts_debug_f_ov, _, _ = get_feature(sim_track, data_ev_cur, data_ov_cur,
                                                                                 use_intp=1)

        # Get near other vehicles
        data_ov_near = get_selected_vehicles(data_ov, id_near_cur)

        id_traj_ov_near_plot, traj_ov_near_plot, size_traj_ov_near_plot = \
            get_vehicle_trajectory(data_ov_near, nidx_t, horizon_ev, handle_remain=1)

        # (CHECK)
        is_collision = check_collision_near(data_ev_cur, data_ov_near, nidx_t)
        if is_collision == 1:
            print("COLLISION!")

        # STEP2: FIND CONTROL -----------------------------------------------------------------------------------------#
        traj_sel_ev, traj_array_ev, cost_array_ev, idx_traj_invalid_ev, pnt_ahead_ev = \
            sim_control.find_control_naive(u_set, horizon_ev, data_ov_near, nidx_t, dist_ahead)

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

            sim_screen.draw_track()  # Draw track
            sim_screen.draw_vehicle_fill(data_ov_cur, id_near_cur, get_rgb("Crimson"))
            sim_screen.draw_vehicle_arrow(data_ov_cur, id_near_cur, 20, get_rgb("White"))
            sim_screen.draw_vehicle_border(data_ov_cur, id_near_cur, 3, get_rgb("Red"))

            sim_screen.draw_target_vehicle_fill(data_ev_cur[1], data_ev_cur[2], data_ev_cur[3], data_ev_cur[6],
                                                data_ev_cur[5], get_rgb("Dodger Blue"))
            sim_screen.draw_target_vehicle_arrow(data_ev_cur[1], data_ev_cur[2], data_ev_cur[3], data_ev_cur[6],
                                                 data_ev_cur[5], data_ev_cur[4], sim_control.v_range[1], get_rgb("White"))
            sim_screen.draw_vehicle_border(data_ev_cur, [-1], 3, get_rgb("Blue"))

            sim_screen.draw_pnt(pnt_ahead_ev[0:2], 3, get_rgb("Red"))

            sim_screen.draw_trajectory_array(traj_ov_near_plot, 2, get_rgb("Maroon"))
            sim_screen.draw_vehicle_border_trajectory_array(traj_ov_near_plot, size_traj_ov_near_plot, 5, 1.5,
                                                            get_rgb("Maroon"))

            sim_screen.draw_trajectory_array_w_cost(traj_array_ev, cost_array_ev, idx_traj_invalid_ev, 0, 1)
            sim_screen.draw_trajectory(traj_sel_ev, 2.5, get_rgb("Blue"))

            sim_screen.draw_pnts(pnts_debug_f_ev[:, 0:2], 2.5, get_rgb("Yellow"))
            sim_screen.draw_pnts_cmap(pnts_debug_f_ov[:, 0:2], 3)

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

        # STEP6: UPDATE -----------------------------------------------------------------------------------------------#
        s_next = traj_sel_ev[1, :]
        sim_control.update_state(s_next[0], s_next[1], s_next[2], s_next[3])

    print("Done")
