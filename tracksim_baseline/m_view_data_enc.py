# TRACK SIMULATOR (VIEW VEHICLE DATA)
#   - pygame-based simulator
#   - use pycairo for graphics

from __future__ import print_function

import sys
sys.path.insert(0, "../")

from core.SimTrack import *
from core.SimScreen import *


if __name__ == "__main__":

    MODE_SCREEN = 1  # 0: plot all track // 1: plot part of track

    horizon = 18  # Horizon length

    # SET TRACK -------------------------------------------------------------------------------------------------------#
    trackname = "US101"
    # trackname = "I80"
    # trackname = "highD_25"
    sim_track = SimTrack(trackname)

    # LOAD VEHICLE DATA -----------------------------------------------------------------------------------------------#
    #       structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    is_track_simple = 0
    if trackname == "US101":
        data_v = np.load("../data_vehicle/dv_ngsim_us101_1_s.npy")
    elif trackname == "I80":
        data_v = np.load("../data_vehicle/dv_ngsim_i80_1_s.npy")
    elif "highD" in trackname:
        text_split = trackname.split("_")
        track_num = text_split[-1]
        filename2read = "../data_vehicle/dv_highD_{:s}.npy".format(track_num)
        data_v = np.load(filename2read)
        is_track_simple = 1
    else:
        data_v = np.load("../data_vehicle/dv_ngsim_us101_1.npy")

    # Target vehicle id
    id_unique = np.unique(data_v[:, -1])
    id_tv = id_unique[40]
    # id_tv = 1863

    # SET SIM SCREEN --------------------------------------------------------------------------------------------------#
    if MODE_SCREEN == 0:
        screen_alpha, screen_size = sim_track.screen_alpha_wide, sim_track.screen_size_wide
    else:
        screen_alpha, screen_size = sim_track.screen_alpha_narrow, sim_track.screen_size_narrow

    if id_tv >= 0:
        idx_tmp = np.where(data_v[:, -1] == id_tv)
        idx_tmp = idx_tmp[0]
        data_vehicle_tmp = data_v[idx_tmp, :]

        t_min, t_max = int(np.amin(data_vehicle_tmp[:, 0])), int(np.amax(data_vehicle_tmp[:, 0]))
    else:
        t_min, t_max = int(np.amin(data_v[:, 0])), int(np.amax(data_v[:, 0]))
    print("t_min: " + str(t_min) + ", t_max: " + str(t_max))

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
    clock = pygame.time.Clock()
    clock_rate = 30

    if id_tv >= 0:
        traj_tv = np.zeros((t_max - t_min + 1, 4), dtype=np.float32)

    for t_cur in range(t_min, t_max + 1):
        # STEP1: GET CURRENT INFO -------------------------------------------------------------------------------------#
        # Select data (w.r.t time)
        idx_sel_ = np.where(data_v[:, 0] == t_cur)
        idx_sel = idx_sel_[0]
        data_ov_cur = data_v[idx_sel, :]

        id_min = min(data_ov_cur[:, -1])
        id_max = max(data_ov_cur[:, -1])
        print("[" + str(t_cur) + ", (" + str(t_min) + ", " + str(t_max) + ")] vehicle: (%d / %d)" % (id_min, id_max))

        if id_tv >= 0:
            idx_tv = np.where(data_ov_cur[:, 9] == id_tv)
            idx_tv = idx_tv[0][0]  # indexes of target vehicle
            data_tv_cur = data_ov_cur[idx_tv, :]
            # data_tv_cur[7], data_tv_cur[8] = -1, -1
            data_tv_cur = data_tv_cur.reshape(-1)
            s_tv = data_tv_cur[1:5]
            traj_tv[t_cur - t_min, :] = s_tv

            # Get feature
            idx_ov_rest = np.setdiff1d(np.arange(0, data_ov_cur.shape[0]), idx_tv)
            data_ov_cur_rest = data_ov_cur[idx_ov_rest, :]
            f_cur, id_near_cur, pnts_debug_f_ev, pnts_debug_f_ov, _, _ = get_feature(sim_track, data_tv_cur,
                                                                                     data_ov_cur_rest, use_intp=0)

            traj_x_out, size_x_out = get_vehicle_trajectory_per_id(data_v, t_cur, id_tv, horizon, do_reverse=1,
                                                                   handle_remain=1)
            traj_x_enc = encode_trajectory(traj_x_out, 0, sim_track.pnts_poly_track, sim_track.pnts_lr_border_track,
                                           is_track_simple=is_track_simple)
            traj_x_dec = decode_trajectory(s_tv[0:3], traj_x_enc, horizon, 0, sim_track.pnts_poly_track,
                                           sim_track.pnts_lr_border_track, is_track_simple=is_track_simple)

            traj_x_out_sp = traj_x_out[np.arange(0, horizon + 2, 2), :]
            traj_x_enc_sp = encode_trajectory(traj_x_out_sp, 0, sim_track.pnts_poly_track,
                                              sim_track.pnts_lr_border_track, is_track_simple=is_track_simple)
            traj_x_dec_sp = decode_trajectory(s_tv[0:3], traj_x_enc_sp, int(horizon / 2), 0, sim_track.pnts_poly_track,
                                              sim_track.pnts_lr_border_track, is_track_simple=is_track_simple)

            print(np.max(traj_x_enc_sp))

            traj_y_out, size_y_out = get_vehicle_trajectory_per_id(data_v, t_cur, id_tv, horizon, do_reverse=0,
                                                                   handle_remain=1)
            traj_y_enc = encode_trajectory(traj_y_out, 1, sim_track.pnts_poly_track, sim_track.pnts_lr_border_track,
                                           is_track_simple=is_track_simple)

            traj_y_dec_ = decode_trajectory(s_tv[0:3], traj_y_enc, horizon, 1, sim_track.pnts_poly_track,
                                            sim_track.pnts_lr_border_track, is_track_simple=is_track_simple)
            # traj_decoded = traj_decoded_[0:-1:2, :]
            traj_y_dec = traj_y_dec_

            traj_naive = np.copy(traj_y_out)
            diff_tmp = traj_y_out[1, :] - traj_y_out[0, :]
            v_init = np.sqrt(diff_tmp[0] * diff_tmp[0] + diff_tmp[1] * diff_tmp[1]) / 0.1
            theta_init = traj_y_out[0, 2]
            traj_naive[:, 2] = theta_init
            for nidx_d in range(1, traj_naive.shape[0]):
                traj_naive[nidx_d, 0:2] = [traj_naive[nidx_d - 1, 0] + 0.1 * v_init * np.cos(theta_init),
                                           traj_naive[nidx_d - 1, 1] + 0.1 * v_init * np.sin(theta_init)]
        else:
            idx_tv = -1
            data_tv_cur, s_tv, id_near_cur, pnts_debug_f_ev, pnts_debug_f_ov = [], [], [], [], []
            traj_y_out, traj_y_dec = [], []

        if MODE_SCREEN == 1:
            sim_screen.set_pnts_range_wrt_mean(s_tv[0:2])  # Set middle point of window

        # Check collision
        is_collision = 0
        if MODE_SCREEN == 1:
            if len(data_tv_cur) > 0:
                idx_oc = np.setdiff1d(np.arange(0, data_ov_cur.shape[0]), idx_tv)
                data_vehicle_collision = data_ov_cur[idx_oc, :]
                is_collision = check_collision(data_tv_cur, data_vehicle_collision)

        # STEP2: DRAW -------------------------------------------------------------------------------------------------#
        # Clear screen
        sim_screen.draw_background("White Smoke")

        sim_screen.draw_track()  # Draw track
        if id_tv >= 0:
            sim_screen.draw_vehicle_fill(data_ov_cur, id_near_cur, get_rgb("Crimson"))
            sim_screen.draw_vehicle_arrow(data_ov_cur, id_near_cur, 20, get_rgb("White"))
            sim_screen.draw_vehicle_border(data_ov_cur, id_near_cur, 3, get_rgb("Red"))

            sim_screen.draw_target_vehicle_fill(data_tv_cur[1], data_tv_cur[2], data_tv_cur[3], data_tv_cur[6],
                                                data_tv_cur[5], get_rgb("Dodger Blue"))
            sim_screen.draw_target_vehicle_arrow(data_tv_cur[1], data_tv_cur[2], data_tv_cur[3], data_tv_cur[6],
                                                 data_tv_cur[5], data_tv_cur[4], 20, get_rgb("White"))
            sim_screen.draw_vehicle_border(data_tv_cur, [id_tv], 3, get_rgb("Blue"))

            sim_screen.draw_vehicle_origin(data_tv_cur, get_rgb("Dodger Blue"))

            sim_screen.draw_pnts(traj_y_out[:, 0:2], 2.5, get_rgb("Red"))
            sim_screen.draw_pnts(traj_y_dec[:, 0:2], 1.2, get_rgb("Black"))
            sim_screen.draw_trajectory(traj_y_out[:, 0:2], 1, get_rgb("Red"))

            sim_screen.draw_pnts(traj_x_out[:, 0:2], 2.5, get_rgb("Blue"))
            sim_screen.draw_pnts(traj_x_dec_sp[:, 0:2], 1.2, get_rgb("Cyan"))
            sim_screen.draw_trajectory(traj_x_out[:, 0:2], 1, get_rgb("Blue"))
            sim_screen.draw_trajectory(traj_x_dec_sp[:, 0:2], 0.5, get_rgb("Cyan"))

            if MODE_SCREEN == 1:
                sim_screen.draw_pnts(pnts_debug_f_ev[:, 0:2], 2.5, get_rgb("Yellow"))
                sim_screen.draw_pnts_cmap(pnts_debug_f_ov[:, 0:2], 3)
        else:
            sim_screen.draw_vehicle_origin(data_ov_cur, get_rgb("Crimson"))

        # STEP3: DISPLAY ----------------------------------------------------------------------------------------------#
        pygame.event.get()
        pygame.display.set_mode((sim_screen.screen_size[0], sim_screen.screen_size[1]))

        # Get data surface
        data_surface = sim_screen.bgra_surf_to_rgba_string()

        # Create pygame surface
        pygame_surface = pygame.image.frombuffer(data_surface, (sim_screen.screen_size[0],
                                                                sim_screen.screen_size[1]), 'RGBA')

        # Show pygame surface
        sim_screen.pygame_screen.blit(pygame_surface, (0, 0))
        pygame.display.flip()

        if is_collision == 1:
            print("Collision!")

        # Limit frame-rate (frames per second)
        clock.tick(clock_rate)

    print("Done")
