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

    # SET TRACK -------------------------------------------------------------------------------------------------------#
    # trackname = "US101"
    # trackname = "I80"
    trackname = "highD_25"
    sim_track = SimTrack(trackname)

    # LOAD VEHICLE DATA -----------------------------------------------------------------------------------------------#
    #       structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    if trackname == "US101":
        data_v = np.load("../data_vehicle/dv_ngsim_us101_2_s.npy")
    elif trackname == "I80":
        data_v = np.load("../data_vehicle/dv_ngsim_i80_2_s.npy")
    elif "highD" in trackname:
        text_split = trackname.split("_")
        track_num = text_split[-1]
        filename2read = "../data_vehicle/dv_highD_{:s}.npy".format(track_num)
        data_v = np.load(filename2read)
    else:
        data_v = np.load("../data_vehicle/dv_ngsim_us101_1.npy")

    # Target vehicle id
    id_unique = np.unique(data_v[:, -1])
    # id_tv = id_unique[300]
    id_tv = -1
    if id_tv == -1:
        MODE_SCREEN = 0

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
        v_mean, v_min, v_max = np.mean(data_vehicle_tmp[:, 4]), np.amin(data_vehicle_tmp[:, 4]), \
                               np.amax(data_vehicle_tmp[:, 4])
        len_mean, len_min, len_max = np.mean(data_vehicle_tmp[:, 5]), np.amin(data_vehicle_tmp[:, 5]), \
                                     np.amax(data_vehicle_tmp[:, 5])
        width_mean, width_min, width_max = np.mean(data_vehicle_tmp[:, 6]), np.amin(data_vehicle_tmp[:, 6]), \
                                           np.amax(data_vehicle_tmp[:, 6])
        id_mean, id_min, id_max = np.mean(data_vehicle_tmp[:, -1]), np.amin(data_vehicle_tmp[:, -1]), \
                                  np.amax(data_vehicle_tmp[:, -1])
    else:
        t_min, t_max = int(np.amin(data_v[:, 0])), int(np.amax(data_v[:, 0]))
        v_mean, v_min, v_max = np.mean(data_v[:, 4]), np.amin(data_v[:, 4]), np.amax(data_v[:, 4])
        len_mean, len_min, len_max = np.mean(data_v[:, 5]), np.amin(data_v[:, 5]), np.amax(data_v[:, 5])
        width_mean, width_min, width_max = np.mean(data_v[:, 6]), np.amin(data_v[:, 6]), np.amax(data_v[:, 6])
        id_mean, id_min, id_max = np.mean(data_v[:, -1]), np.amin(data_v[:, -1]), np.amax(data_v[:, -1])

    print("t_min: " + str(t_min) + ", t_max: " + str(t_max))
    print("v_mean: " + str(v_mean) + ", v_min: " + str(v_min) + ", v_max: " + str(v_max))
    print("len_mean: " + str(len_mean) + ", len_min: " + str(len_min) + ", len_max: " + str(len_max))
    print("width_mean: " + str(width_mean) + ", width_min: " + str(width_min) + ", width_max: " + str(width_max))
    print("id_mean: " + str(id_mean) + ", id_min: " + str(id_min) + ", id_max: " + str(id_max))

    sim_screen = SimScreen(screen_size, screen_alpha, MODE_SCREEN)  # Set simulation screen
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

    for t_cur in range(t_min + 15000, t_max):
        # STEP1: GET CURRENT INFO -------------------------------------------------------------------------------------#
        # Select data (w.r.t time)
        idx_sel_ = np.where(data_v[:, 0] == t_cur)
        idx_sel = idx_sel_[0]
        data_ov_cur = data_v[idx_sel, :]

        id_min = min(data_ov_cur[:, -1])
        id_max = max(data_ov_cur[:, -1])
        print("[" + str(t_cur) + ", (" + str(t_min) + ", " + str(t_max) + ")] vehicle: (%d / %d)" % (id_min, id_max))

        xmin_ov_cur, xmax_ov_cur = np.min(data_ov_cur[:, 1]), np.max(data_ov_cur[:, 1])
        ymin_ov_cur, ymax_ov_cur = np.min(data_ov_cur[:, 2]), np.max(data_ov_cur[:, 2])
        print("min: [{:f}, {:f}] // max: [{:f}, {:f}]".format(xmin_ov_cur, ymin_ov_cur, xmax_ov_cur, ymax_ov_cur))

        if id_tv >= 0:
            idx_tv = np.where(data_ov_cur[:, 9] == id_tv)
            idx_tv = idx_tv[0][0]  # indexes of target vehicle
            data_tv_cur = data_ov_cur[idx_tv, :]

            # data_tv_cur[7], data_tv_cur[8] = -1, -1
            data_tv_cur = data_tv_cur.reshape(-1)
            s_tv = data_tv_cur[1:5]
            # Get feature
            idx_ov_rest = np.setdiff1d(np.arange(0, data_ov_cur.shape[0]), idx_tv)
            data_ov_cur_rest = data_ov_cur[idx_ov_rest, :]
            f_cur, id_near_cur, pnts_debug_f_ev, pnts_debug_f_ov, _, _ = get_feature(sim_track, data_tv_cur,
                                                                                     data_ov_cur_rest, use_intp=0)
        else:
            idx_tv = -1
            data_tv_cur, s_tv, id_near_cur, pnts_debug_f_ev, pnts_debug_f_ov = [], [], [], [], []

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
        sim_screen.draw_background("Light Gray")

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

            sim_screen.draw_pnt(sim_track.pnt_min[0:2], 3, get_rgb("Red"))
            sim_screen.draw_pnt(sim_track.pnt_max[0:2], 3, get_rgb("Blue"))

            if MODE_SCREEN == 1:
                sim_screen.draw_pnts(pnts_debug_f_ev[:, 0:2], 2.5, get_rgb("Yellow"))
                sim_screen.draw_pnts_cmap(pnts_debug_f_ov[:, 0:2], 3)
        else:
            sim_screen.draw_vehicle_origin(data_ov_cur, get_rgb("Crimson"))
            sim_screen.draw_vehicle_arrow(data_ov_cur, data_ov_cur[:, -1], 20, get_rgb("White"))

        if MODE_SCREEN == 1:
            # Draw sub
            sim_screen.draw_box_range(get_rgb("Black"), 1.5)
            sim_screen.draw_track_sub(sim_track.pnts_poly_track)
            sim_screen.draw_pnts_sub(data_ov_cur[:, 1:3], 1.0, get_rgb("Salmon"))
            sim_screen.draw_pnts_sub(sim_track.pnts_goal[:, 0:2], 1.0, get_rgb("Yellow"))

            if id_tv >= 0:
                sim_screen.draw_pnt_sub(s_tv[0:2], 2.0, get_rgb("Cyan"))

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
