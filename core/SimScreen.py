# SET SCREEN FOR SIMULATOR (PYGAME)
#   - Pygame-based simulator
#   - Use pycairo for graphics

from __future__ import print_function

import sys
sys.path.insert(0, "../../")

import pygame
import pygame.gfxdraw

import cairo
# import cairocffi as cairo  # for virtual env

from PIL import Image

import matplotlib
import matplotlib.cm

from src.utils_sim import *
from src.get_rgb import *


class SimScreen(object):
    def __init__(self, screen_size, screen_alpha, screen_mode):
        # Set screen settings
        self.screen_size = screen_size
        self.screen_alpha = screen_alpha
        self.screen_mode = screen_mode

        # Set display
        pygame.init()  # Initialize the game engine
        pygame.display.set_caption("demo track sim")
        self.pygame_screen = pygame.display.set_mode((self.screen_size[0], self.screen_size[1]), 0, 32)

        # Create raw surface data
        # data_surface_raw = np.empty(self.screen_size[0] * self.screen_size[1] * 4, dtype=np.int8)

        # Set surface (cairo)
        # stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, self.screen_size[0])
        # self.pycairo_surface = cairo.ImageSurface.create_for_data(data_surface_raw, cairo.FORMAT_ARGB32,
        #                                                           self.screen_size[0], self.screen_size[1], stride)

        self.pycairo_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.screen_size[0], self.screen_size[1])

        # Set context (cairo)
        self.ctx = cairo.Context(self.pycairo_surface)

        # Pnts of range (screen)
        self.pnt_min = []
        self.pnt_max = []

        # Pnts (track)
        self.pnts_poly_track, self.pnts_outer_border_track, self.pnts_inner_border_track = [], [], []

        # Pixel (track)
        self.pixel_poly_track, self.pixel_outer_border_track, self.pixel_inner_border_track = [], [], []

        # Pixel (vehicle)
        self.pixel_vehicle_others, self.pixel_vehicle_ego = [], []

        # Sub-screen
        self.screen_size_sub = (screen_size[0]/3, screen_size[1]/3)
        self.screen_alpha_sub = []
        self.pnt_min_sub, self.pnt_max_sub = [], []
        self.quadrant_number_sub = 2  # 1: upper-left, 2: upper-right, 3: lower-right, 4: lower-left

    # SET PARAMETERS --------------------------------------------------------------------------------------------------#
    def set_screen_size(self, screen_size):
        # screen_size: (tuple) screen size (dim = 2)
        self.screen_size = screen_size

    def set_screen_alpha(self, screen_alpha):
        # screen_alpha: (scalar) screen ratio
        self.screen_alpha = screen_alpha

    def set_pnts_range(self, pnt_min, pnt_max):
        # pnt_min, pnt_max: (list or ndarray) point (dim = 2)
        self.pnt_min = pnt_min
        self.pnt_max = pnt_max

    def set_pnts_range_wrt_mean(self, pnt_mean):
        # pnt_mean: (list or ndarray) point (dim = 2)
        screen_size_tmp = np.array(self.screen_size)
        pnt_range = screen_size_tmp / self.screen_alpha
        self.pnt_min = pnt_mean - pnt_range / 2
        self.pnt_max = pnt_mean + pnt_range / 2

    def set_pnts_track_init(self, pnts_poly_track, pnts_outer_border_track, pnts_inner_border_track):
        self.pnts_poly_track = pnts_poly_track
        self.pnts_outer_border_track = pnts_outer_border_track
        self.pnts_inner_border_track = pnts_inner_border_track

    # UTILS -----------------------------------------------------------------------------------------------------------#
    def convert2pixel(self, pnts):
        # pnts: (ndarray) points (dim = N x 2)

        if ~isinstance(pnts, np.ndarray):
            pnts = np.array(pnts)

        num_pnts = pnts.shape[0]
        pnts_conv = np.zeros((num_pnts, 2), dtype=np.float64)

        pnts_conv[:, 0] = pnts[:, 0] - np.repeat(self.pnt_min[0], num_pnts, axis=0)
        pnts_conv[:, 1] = np.repeat(self.pnt_max[1], num_pnts, axis=0) - pnts[:, 1]

        pnts_conv = pnts_conv * self.screen_alpha
        # pnts_conv = np.round(pnts_conv, 0)

        return pnts_conv

    def update_pixel_track(self):
        self.pixel_poly_track = self.update_pixel_track_sub(self.pnts_poly_track)
        self.pixel_outer_border_track = self.update_pixel_track_sub(self.pnts_outer_border_track)
        self.pixel_inner_border_track = self.update_pixel_track_sub(self.pnts_inner_border_track)

    def update_pixel_track_sub(self, pnts_track):
        pixel_track = []
        num_lane_seg = 0  # number of lane-segment
        for nidx_seg in range(0, len(pnts_track)):
            seg_sel = pnts_track[nidx_seg]

            pixel_seg = []
            for nidx_lane in range(0, len(seg_sel)):
                num_lane_seg = num_lane_seg + 1
                pnts_tmp = seg_sel[nidx_lane]
                pnts_conv_tmp = self.convert2pixel(pnts_tmp)
                pixel_seg.append(pnts_conv_tmp)

            pixel_track.append(pixel_seg)
        return pixel_track

    def snapCoords(self, x, y):
        (xd, yd) = self.ctx.user_to_device(x, y)
        return (round(xd) + 0.5, round(yd) + 0.5)

    # DRAW (BASIC) ----------------------------------------------------------------------------------------------------#
    # Get image from pycairo-surface
    def bgra_surf_to_rgba_string(self):
        # Convert memoryview object to byte-array
        data_tmp = self.pycairo_surface.get_data()
        data_tmp = data_tmp.tobytes()

        # Use PIL to get img
        img = Image.frombuffer('RGBA', (self.pycairo_surface.get_width(), self.pycairo_surface.get_height()), data_tmp,
                               'raw', 'BGRA', 0, 1)

        return img.tobytes('raw', 'RGBA', 0, 1)

    # Draw box with specific color
    def draw_background(self, color_name):
        # hcolor: (tuple) color

        hcolor = get_rgb(color_name)

        self.ctx.move_to(0, 0)
        self.ctx.line_to(self.screen_size[0], 0)
        self.ctx.line_to(self.screen_size[0], self.screen_size[1])
        self.ctx.line_to(0, self.screen_size[1])
        self.ctx.line_to(0, 0)
        self.ctx.set_line_width(1)
        self.ctx.set_source_rgb(hcolor[0], hcolor[1], hcolor[2])
        self.ctx.fill()

    # Draw point
    def draw_pnt(self, pnt, radius, hcolor):
        # pnt: (ndarray) point (dim = 1 x 2)
        # radius: (scalar) radius
        # hcolor: (tuple) color

        if ~isinstance(pnt, np.ndarray):
            pnt = np.array(pnt)
        pnt = pnt.reshape(-1)

        pnt = pnt[0:2]
        pnt = np.reshape(pnt, (1, 2))

        # Convert to pixel space
        pnt_sel_conv = self.convert2pixel(pnt)
        self.ctx.arc(pnt_sel_conv[0, 0], pnt_sel_conv[0, 1], radius, 0, 2*math.pi)
        self.ctx.set_source_rgb(hcolor[0], hcolor[1], hcolor[2])
        # self.ctx.stroke()
        self.ctx.fill()

    # Draw point (tr)
    def draw_pnt_tr(self, pnt, radius, hcolor, tr):
        # pnt: (ndarray) point (dim = 1 x 2)
        # radius: (scalar) radius
        # hcolor: (tuple) color
        # tr: (scalar) transparency 0 ~ 1

        if ~isinstance(pnt, np.ndarray):
            pnt = np.array(pnt)
        pnt = pnt.reshape(-1)

        pnt = pnt[0:2]
        pnt = np.reshape(pnt, (1, 2))

        # Convert to pixel space
        pnt_sel_conv = self.convert2pixel(pnt)
        self.ctx.arc(pnt_sel_conv[0, 0], pnt_sel_conv[0, 1], radius, 0, 2 * math.pi)
        self.ctx.set_source_rgba(hcolor[0], hcolor[1], hcolor[2], tr)
        # self.ctx.stroke()
        self.ctx.fill()

    # Draw points
    def draw_pnts(self, pnts, radius, hcolor):
        # pnts: (ndarray) points (dim = N x 2)
        # radius: (scalar) radius
        # hcolor: (tuple) color

        if ~isinstance(pnts, np.ndarray):
            pnts = np.array(pnts)

        if len(pnts.shape) == 1:
            pnts = np.reshape(pnts, (1, -1))

        num_in = pnts.shape[0]

        for nidx_d in range(0, num_in):
            pnt_sel = pnts[nidx_d, :]
            pnt_sel = np.reshape(pnt_sel, (1, 2))
            self.draw_pnt(pnt_sel, radius, hcolor)

    # Draw points (tr)
    def draw_pnts_tr(self, pnts, radius, hcolor, tr):
        # pnts: (ndarray) points (dim = N x 2)
        # radius: (scalar) radius
        # hcolor: (tuple) color
        # tr: (scalar) transparency 0 ~ 1

        if ~isinstance(pnts, np.ndarray):
            pnts = np.array(pnts)

        if len(pnts.shape) == 1:
            pnts = np.reshape(pnts, (1, -1))

        num_in = pnts.shape[0]

        for nidx_d in range(0, num_in):
            pnt_sel = pnts[nidx_d, :]
            pnt_sel = np.reshape(pnt_sel, (1, 2))
            self.draw_pnt_tr(pnt_sel, radius, hcolor, tr)

    # Draw points (colormap)
    def draw_pnts_cmap(self, pnts, radius):
        # pnts: (ndarray) points (dim = N x 2)
        # radius: (scalar) radius

        num_in = pnts.shape[0]
        cmap = matplotlib.cm.get_cmap("rainbow")

        for nidx_d in range(0, num_in):
            pnt_sel = pnts[nidx_d, 0:2]
            pnt_sel = np.reshape(pnt_sel, (1, 2))
            cmap_sel = cmap(nidx_d/num_in)
            self.draw_pnt(pnt_sel, radius, cmap_sel[0:3])

    # Draw points-array
    def draw_pnts_array(self, pnts_array, hcolor, radius):
        # pnts_array: list of points (ndarray, dim = N x 2)
        # hcolor: (tuple) rgb color
        # radius: (scalar) radius

        for nidx_d in range(0, len(pnts_array)):
            pnts_sel = pnts_array[nidx_d]
            self.draw_pnts(pnts_sel[:, 0:2], radius, hcolor)

    # DRAW (MAIN) -----------------------------------------------------------------------------------------------------#
    # Draw track
    def draw_track(self):
        # pnts_poly_track: (list) points of track

        if self.screen_mode == 0:
            linewidth_outer, linewidth_inner = 1, 1
        elif self.screen_mode == 1:
            linewidth_outer, linewidth_inner = 2, 2
        else:
            linewidth_outer, linewidth_inner = 1, 1

        self.update_pixel_track()

        # Plot track (polygon)
        for nidx_seg in range(0, len(self.pixel_poly_track)):
            pixel_poly_seg = self.pixel_poly_track[nidx_seg]

            # Plot lane-segment
            for nidx_lane in range(0, len(pixel_poly_seg)):
                idx_lane = len(pixel_poly_seg) - nidx_lane - 1
                # Pnts on lane-segment
                pixel_poly_lane = pixel_poly_seg[idx_lane]

                for nidx_pnt in range(0, pixel_poly_lane.shape[0]):
                    pixel_tmp = pixel_poly_lane[nidx_pnt, :]
                    pixel_tmp = self.snapCoords(pixel_tmp[0], pixel_tmp[1])
                    if nidx_pnt == 0:
                        self.ctx.move_to(pixel_tmp[0], pixel_tmp[1])
                    else:
                        self.ctx.line_to(pixel_tmp[0], pixel_tmp[1])

                pnt_0_tmp = pixel_poly_lane[0, :]
                pnt_0_tmp = self.snapCoords(pnt_0_tmp[0], pnt_0_tmp[1])
                self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

                # Set (fill) color
                cmap_lane = get_rgb("Dim Gray")

                # Plot (cairo)
                self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                self.ctx.set_source_rgb(cmap_lane[0], cmap_lane[1], cmap_lane[2])
                self.ctx.fill_preserve()
                # self.ctx.set_source_rgb(0, 0, 0)
                # self.ctx.set_line_width(linewidth_outer)
            self.ctx.stroke()

        # Plot track (outer)
        for nidx_seg in range(0, len(self.pixel_outer_border_track)):
            pixel_outer_seg = self.pixel_outer_border_track[nidx_seg]
            for nidx_lane in range(0, len(pixel_outer_seg)):
                pixel_outer_lane = pixel_outer_seg[nidx_lane]

                for nidx_pnt in range(0, pixel_outer_lane.shape[0]):
                    pixel_tmp = pixel_outer_lane[nidx_pnt, :]
                    pixel_tmp = self.snapCoords(pixel_tmp[0], pixel_tmp[1])
                    if nidx_pnt == 0:
                        self.ctx.move_to(pixel_tmp[0], pixel_tmp[1])
                    else:
                        self.ctx.line_to(pixel_tmp[0], pixel_tmp[1])

                # Set (fill) color
                cmap_lane = get_rgb("Black")

                # Plot (cairo)
                self.ctx.set_line_width(linewidth_outer)
                self.ctx.set_source_rgb(cmap_lane[0], cmap_lane[1], cmap_lane[2])
                self.ctx.stroke()

        # Plot track (inner)
        hcolor_inner = get_rgb("Silver")
        for nidx_seg in range(0, len(self.pixel_inner_border_track)):
            pixel_inner_seg = self.pixel_inner_border_track[nidx_seg]
            for nidx_lane in range(0, len(pixel_inner_seg)):
                pixel_inner_lane = pixel_inner_seg[nidx_lane]

                for nidx_pnt in range(0, pixel_inner_lane.shape[0], 2):
                    pixel_cur_tmp = pixel_inner_lane[nidx_pnt, :]
                    pixel_cur_tmp = self.snapCoords(pixel_cur_tmp[0], pixel_cur_tmp[1])

                    if (nidx_pnt + 1) <= (pixel_inner_lane.shape[0] - 1):
                        pixel_next_tmp = pixel_inner_lane[nidx_pnt + 1, :]
                        pixel_next_tmp = self.snapCoords(pixel_next_tmp[0], pixel_next_tmp[1])

                        self.ctx.move_to(pixel_cur_tmp[0], pixel_cur_tmp[1])
                        self.ctx.line_to(pixel_next_tmp[0], pixel_next_tmp[1])

                        self.ctx.set_line_width(linewidth_inner)
                        self.ctx.set_source_rgb(hcolor_inner[0], hcolor_inner[1], hcolor_inner[2])
                        self.ctx.stroke()
                    else:
                        self.ctx.arc(pixel_cur_tmp[0], pixel_cur_tmp[1], 1, 0, 2 * math.pi)
                        self.ctx.set_source_rgb(hcolor_inner[0], hcolor_inner[1], hcolor_inner[2])
                        self.ctx.fill()

    # Draw trajectory
    def draw_trajectory(self, traj, hlinewidth, hcolor):
        # traj: (ndarray) trajectory (dim = N x 2)
        # hlinewidth: (scalar) linewidth
        # hcolor: (tuple) rgb color

        # Convert to pixel space
        traj_conv_tmp = self.convert2pixel(traj)

        # Plot (cairo)
        for nidx_pnt in range(0, traj_conv_tmp.shape[0]):
            pnt_tmp = traj_conv_tmp[nidx_pnt, :]
            if nidx_pnt == 0:
                self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
            else:
                self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

        self.ctx.set_line_width(hlinewidth)
        self.ctx.set_source_rgb(hcolor[0], hcolor[1], hcolor[2])
        self.ctx.stroke()

    # Draw trajectory (transparent)
    def draw_trajectory_tr(self, traj, hlinewidth, hcolor, tr):
        # traj: (ndarray) trajectory (dim = N x 2)
        # hlinewidth: (scalar) linewidth
        # hcolor: (tuple) rgb color
        # tr: (scalar) transparency 0 ~ 1

        # Convert to pixel space
        traj_conv_tmp = self.convert2pixel(traj)

        # Plot (cairo)
        for nidx_pnt in range(0, traj_conv_tmp.shape[0]):
            pnt_tmp = traj_conv_tmp[nidx_pnt, :]
            if nidx_pnt == 0:
                self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
            else:
                self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

        self.ctx.set_line_width(hlinewidth)
        self.ctx.set_source_rgba(hcolor[0], hcolor[1], hcolor[2], tr)
        self.ctx.stroke()

    # Draw trajectory (array)
    def draw_trajectory_array(self, traj_array, hlinewidth, hcolor):
        # traj_array: list of trajectory (ndarray, dim = N x 2)
        # hlinewidth: (scalar) linewidth
        # hcolor: (tuple) rgb color

        for nidx_traj in range(0, len(traj_array)):
            traj_sel = traj_array[nidx_traj]
            self.draw_trajectory(traj_sel, hlinewidth, hcolor)

    # Draw trajectory (array, transparent)
    def draw_trajectory_array_tr(self, traj_array, hlinewidth, hcolor, tr):
        # traj_array: list of trajectory (ndarray, dim = N x 2)
        # hlinewidth: (scalar) linewidth
        # hcolor: (tuple) rgb color
        # tr: (scalar) transparency 0 ~ 1

        for nidx_traj in range(0, len(traj_array)):
            traj_sel = traj_array[nidx_traj]
            self.draw_trajectory_tr(traj_sel, hlinewidth, hcolor, tr)

    # Draw trajectory (array) with cost
    def draw_trajectory_array_w_cost(self, traj_array, cost, idx_invalid, cost_max, hlinewidth):
        # traj_array: list of trajectory (ndarray, dim = N x 2)
        # cost: (ndarray) cost of trajectory (dim = N)
        # idx_invalid: (ndarray or list) invalid trajectory array
        # cost_max: (scalar) cost_max (if 0, cost is scaled automatically)
        # hlinewidth: (scalar) linewidth

        if ~isinstance(cost, np.ndarray):
            cost = np.array(cost)
        cost = cost.reshape(-1)

        num_traj = len(traj_array)

        if len(idx_invalid) > 0:
            idx_valid = np.setdiff1d(np.arange(0, num_traj), idx_invalid)
        else:
            idx_valid = np.arange(0, num_traj)

        if len(idx_valid) > 0:
            cost_valid = cost[idx_valid]
            if cost_max == 0:
                max_cost_valid, min_cost_valid = max(cost_valid), min(cost_valid)
                if abs(max_cost_valid - min_cost_valid) < float(1e-4):
                    max_cost_valid, min_cost_valid = 1.0, 0.0
            else:
                max_cost_valid, min_cost_valid = cost_max, 0.0
        else:
            max_cost_valid, min_cost_valid = 1.0, 0.0

        # cmap = matplotlib.cm.get_cmap("cool")
        cmap = matplotlib.cm.get_cmap("autumn")
        for nidx_traj in range(0, len(idx_invalid)):  # Plot invalid trajectories
            idx_sel = idx_invalid[nidx_traj]
            traj_sel = traj_array[idx_sel]
            self.draw_trajectory_tr(traj_sel[:, 0:2], hlinewidth, get_rgb("Dark Slate Blue"), 0.95)
            # self.draw_pnts_tr(traj_sel[:, 0:2], 3, get_rgb("Medium Slate Blue"), 0.5)

        for nidx_traj in range(0, len(idx_valid)):  # Plot valid trajectories
            idx_sel = idx_valid[nidx_traj]
            traj_sel = traj_array[idx_sel]

            idx_tmp = (cost[idx_sel] - min_cost_valid) / (max_cost_valid - min_cost_valid)
            idx_tmp = min(max(idx_tmp, 0.0), 1.0)
            cmap_1 = cmap(idx_tmp)
            # cmap_1 = get_rgb("Orange")
            self.draw_trajectory_tr(traj_sel[:, 0:2], hlinewidth, cmap_1[0:3], 0.95)

    # Draw vehicle border trajectory
    def draw_vehicle_border_trajectory(self, traj, rx, ry, stepsize, hlinewidth, hcolor):
        # traj: (ndarray) trajectory (dim = N x 3)
        # rx, ry: (scalar) vehicle size
        # stepsize: (scalar) step-size
        # hlinewidth: (scalar) linewidth
        # hcolor: (tuple) rgb color

        for nidx_t in range(stepsize, traj.shape[0], stepsize):
            pnt_sel = traj[nidx_t, :]

            if ~np.isnan(pnt_sel[0]):
                self.draw_target_vehicle_border(pnt_sel[0], pnt_sel[1], pnt_sel[2], rx, ry, hlinewidth, hcolor)

    # Draw vehicle border trajectory (array)
    def draw_vehicle_border_trajectory_array(self, traj_array, size_array, stepsize, hlinewidth, hcolor):
        # traj_array: list of trajectory (ndarray, dim = N x 3)
        # size_array: (ndarray) size array rx, ry (dim = N x 2)
        # stepsize: (scalar) step-size
        # hlinewidth: (scalar) linewidth
        # hcolor: (tuple) rgb color

        for nidx_traj in range(0, len(traj_array)):
            traj_sel = traj_array[nidx_traj]
            size_sel = size_array[nidx_traj, :]
            rx_sel = size_sel[0]
            ry_sel = size_sel[1]
            self.draw_vehicle_border_trajectory(traj_sel, rx_sel, ry_sel, stepsize, hlinewidth, hcolor)

    # Draw vehicle (other-vehicles)
    def draw_vehicle_origin(self, data_vehicle, hcolor):
        # data_vehicle: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = N x 10, width > length)
        # hcolor: (tuple) hcolor

        if ~isinstance(data_vehicle, np.ndarray):
            data_vehicle = np.array(data_vehicle)
        shape_data_vehicle = data_vehicle.shape
        if len(shape_data_vehicle) == 1:
            data_vehicle = np.reshape(data_vehicle, (1, -1))

        num_vehicle = data_vehicle.shape[0]
        for nidx_n in range(0, num_vehicle):
            # Get vehicle-data
            #       structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
            data_vehicle_tmp = data_vehicle[nidx_n, :]

            # Get polygon-points
            pnts_vehicle_tmp = get_pnts_carshape(data_vehicle_tmp[1], data_vehicle_tmp[2], data_vehicle_tmp[3],
                                                 data_vehicle_tmp[6], data_vehicle_tmp[5])

            pnts_vehicle_pixel = self.convert2pixel(pnts_vehicle_tmp)

            # Plot vehicle
            for nidx_pnt in range(0, pnts_vehicle_pixel.shape[0]):
                pnt_tmp = pnts_vehicle_pixel[nidx_pnt, :]
                if nidx_pnt == 0:
                    self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
                else:
                    self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

            pnt_0_tmp = pnts_vehicle_pixel[0, :]
            self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

            # Plot (cairo)
            self.ctx.set_line_width(1)
            self.ctx.set_source_rgb(hcolor[0], hcolor[1], hcolor[2])
            self.ctx.fill_preserve()
            self.ctx.set_source_rgb(0, 0, 0)
            self.ctx.set_line_width(1)
            self.ctx.stroke()

    # Draw vehicle (other-vehicles, tr)
    def draw_vehicle_fill_tr(self, data_vehicle, id_sel, hcolor, tr):
        # data_vehicle: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = N x 10, width > length)
        # id_sel: (ndarray) id of vehicles
        # hcolor: (tuple) rgb color
        # tr: (scalar) transparency 0 ~ 1

        if ~isinstance(data_vehicle, np.ndarray):
            data_vehicle = np.array(data_vehicle)
        shape_data_vehicle = data_vehicle.shape
        if len(shape_data_vehicle) == 1:
            data_vehicle = np.reshape(data_vehicle, (1, -1))

        num_vehicle = data_vehicle.shape[0]
        for nidx_n in range(0, num_vehicle):
            # Get vehicle-data
            #       structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
            data_vehicle_tmp = data_vehicle[nidx_n, :]

            # Get polygon-points
            pnts_vehicle_tmp = get_pnts_carshape(data_vehicle_tmp[1], data_vehicle_tmp[2], data_vehicle_tmp[3],
                                                 data_vehicle_tmp[6], data_vehicle_tmp[5])

            # Set (fill) color
            if np.isin(data_vehicle_tmp[-1], id_sel):
                cmap_vehicle = hcolor
            else:
                cmap_vehicle = get_rgb("Light Gray")

            pnts_vehicle_pixel = self.convert2pixel(pnts_vehicle_tmp)

            # Plot vehicle
            for nidx_pnt in range(0, pnts_vehicle_pixel.shape[0]):
                pnt_tmp = pnts_vehicle_pixel[nidx_pnt, :]
                if nidx_pnt == 0:
                    self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
                else:
                    self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

            pnt_0_tmp = pnts_vehicle_pixel[0, :]
            self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

            # Plot (cairo)
            self.ctx.set_line_width(1)
            self.ctx.set_source_rgba(cmap_vehicle[0], cmap_vehicle[1], cmap_vehicle[2], tr)
            self.ctx.fill_preserve()
            self.ctx.set_source_rgb(0, 0, 0)
            self.ctx.set_line_width(1)
            self.ctx.stroke()

    # Draw vehicle (other-vehicles)
    def draw_vehicle_fill(self, data_vehicle, id_sel, hcolor):
        # data_vehicle: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = N x 10, width > length)
        # id_sel: (ndarray) id of vehicles
        # hcolor: (tuple) rgb color

        if ~isinstance(data_vehicle, np.ndarray):
            data_vehicle = np.array(data_vehicle)
        shape_data_vehicle = data_vehicle.shape
        if len(shape_data_vehicle) == 1:
            data_vehicle = np.reshape(data_vehicle, (1, -1))

        num_vehicle = data_vehicle.shape[0]
        for nidx_n in range(0, num_vehicle):
            # Get vehicle-data
            #       structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
            data_vehicle_tmp = data_vehicle[nidx_n, :]

            # Get polygon-points
            pnts_vehicle_tmp = get_pnts_carshape(data_vehicle_tmp[1], data_vehicle_tmp[2], data_vehicle_tmp[3],
                                                 data_vehicle_tmp[6], data_vehicle_tmp[5])

            # Set (fill) color
            if np.isin(data_vehicle_tmp[-1], id_sel):
                cmap_vehicle = hcolor
            else:
                cmap_vehicle = get_rgb("Light Gray")

            pnts_vehicle_pixel = self.convert2pixel(pnts_vehicle_tmp)

            # Plot vehicle
            for nidx_pnt in range(0, pnts_vehicle_pixel.shape[0]):
                pnt_tmp = pnts_vehicle_pixel[nidx_pnt, :]
                if nidx_pnt == 0:
                    self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
                else:
                    self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

            pnt_0_tmp = pnts_vehicle_pixel[0, :]
            self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

            # Plot (cairo)
            self.ctx.set_line_width(1)
            self.ctx.set_source_rgb(cmap_vehicle[0], cmap_vehicle[1], cmap_vehicle[2])
            self.ctx.fill_preserve()
            self.ctx.set_source_rgb(0, 0, 0)
            self.ctx.set_line_width(1)
            self.ctx.stroke()

    # Draw vehicle fill (target)
    def draw_target_vehicle_fill(self, x, y, theta, rx, ry, hcolor):
        # x, y, theta, rx, ry: (scalar)
        # hcolor: (tuple) rgb color (fill)

        # Get polygon-points
        pnts_vehicle_tmp = get_pnts_carshape(x, y, theta, rx, ry)

        pnts_vehicle_pixel = self.convert2pixel(pnts_vehicle_tmp)

        # Plot vehicle
        for nidx_pnt in range(0, pnts_vehicle_pixel.shape[0]):
            pnt_tmp = pnts_vehicle_pixel[nidx_pnt, :]
            if nidx_pnt == 0:
                self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
            else:
                self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

        pnt_0_tmp = pnts_vehicle_pixel[0, :]
        self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

        # Plot (cairo)
        self.ctx.set_line_width(1)
        self.ctx.set_source_rgb(hcolor[0], hcolor[1], hcolor[2])
        self.ctx.fill_preserve()
        self.ctx.set_source_rgb(0, 0, 0)
        self.ctx.set_line_width(1)
        self.ctx.stroke()

    # Draw vehicle fill (target, tr)
    def draw_target_vehicle_fill_tr(self, x, y, theta, rx, ry, hcolor, tr):
        # x, y, theta, rx, ry: (scalar)
        # hcolor: (tuple) rgb color (fill)
        # tr: (scalar) transparency 0 ~ 1

        # Get polygon-points
        pnts_vehicle_tmp = get_pnts_carshape(x, y, theta, rx, ry)

        pnts_vehicle_pixel = self.convert2pixel(pnts_vehicle_tmp)

        # Plot vehicle
        for nidx_pnt in range(0, pnts_vehicle_pixel.shape[0]):
            pnt_tmp = pnts_vehicle_pixel[nidx_pnt, :]
            if nidx_pnt == 0:
                self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
            else:
                self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

        pnt_0_tmp = pnts_vehicle_pixel[0, :]
        self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

        # Plot (cairo)
        self.ctx.set_line_width(1)
        self.ctx.set_source_rgba(hcolor[0], hcolor[1], hcolor[2], tr)
        self.ctx.fill_preserve()
        self.ctx.set_source_rgb(0, 0, 0)
        self.ctx.set_line_width(1)
        self.ctx.stroke()

    # Draw vehicle arrow (other-vehicles)
    def draw_vehicle_arrow(self, data_vehicle, id_sel, lv_max, hcolor):
        # data_vehicle: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = N x 10, width > length)
        # id_sel: (ndarray) id of vehicles
        # lv_max: (scalar) maximum speed
        # hcolor: (tuple) rgb color

        if ~isinstance(data_vehicle, np.ndarray):
            data_vehicle = np.array(data_vehicle)
        shape_data_vehicle = data_vehicle.shape
        if len(shape_data_vehicle) == 1:
            data_vehicle = np.reshape(data_vehicle, (1, -1))

        idx_found = np.where(id_sel > -1)
        idx_found = idx_found[0]

        for nidx_d in range(0, idx_found.shape[0]):
            id_near_sel = id_sel[idx_found[nidx_d]]

            idx_tmp = np.where(data_vehicle[:, -1] == id_near_sel)
            idx_tmp = idx_tmp[0]

            if len(idx_tmp) > 0:
                data_vehicle_sel = data_vehicle[idx_tmp, :]
                data_vehicle_sel = data_vehicle_sel.reshape(-1)
                self.draw_target_vehicle_arrow(data_vehicle_sel[1], data_vehicle_sel[2], data_vehicle_sel[3],
                                               data_vehicle_sel[6], data_vehicle_sel[5], data_vehicle_sel[4],
                                               lv_max, hcolor)

    # Draw vehicle arrow (target)
    def draw_target_vehicle_arrow(self, x, y, theta, rx, ry, lv, lv_ref, hcolor):
        # x, y, theta, rx, ry, lv, lv_ref: (scalar)
        # hcolor: (tuple) rgb color (arrow)

        ratio_lv = 0.15 + (abs(lv) / lv_ref) * 0.85
        ratio_lv = min(ratio_lv, 1.0)

        ax, ay = ratio_lv * rx * 0.4, ry * 0.15
        bx, by = ratio_lv * rx * 0.15, ry * 0.15

        # Get polygon-points
        pnts_vehicle_tmp = get_pnts_arrow(x, y, theta, ax, ay, bx, by)
        pnts_vehicle_pixel = self.convert2pixel(pnts_vehicle_tmp)

        # Plot vehicle
        for nidx_pnt in range(0, pnts_vehicle_pixel.shape[0]):
            pnt_tmp = pnts_vehicle_pixel[nidx_pnt, :]
            if nidx_pnt == 0:
                self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
            else:
                self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

        pnt_0_tmp = pnts_vehicle_pixel[0, :]
        self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

        # Plot (cairo)
        self.ctx.set_line_width(1)
        self.ctx.set_source_rgb(hcolor[0], hcolor[1], hcolor[2])
        self.ctx.fill()

    # Draw vehicle border (target)
    def draw_target_vehicle_border(self, x, y, theta, rx, ry, hlinewidth, hcolor):
        # x, y, theta, rx, ry: (scalar)
        # hlinewidth: (scalar) linewidth
        # hcolor: (tuple) rgb color

        # Get polygon-points
        pnts_vehicle_tmp = get_pnts_carshape(x, y, theta, rx, ry)

        pnts_vehicle_pixel = self.convert2pixel(pnts_vehicle_tmp)

        # Plot vehicle
        for nidx_pnt in range(0, pnts_vehicle_pixel.shape[0]):
            pnt_tmp = pnts_vehicle_pixel[nidx_pnt, :]
            if nidx_pnt == 0:
                self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
            else:
                self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

        pnt_0_tmp = pnts_vehicle_pixel[0, :]
        self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

        # Plot (cairo)
        self.ctx.set_source_rgb(hcolor[0], hcolor[1], hcolor[2])
        self.ctx.set_line_width(hlinewidth)
        self.ctx.stroke()

    # Draw vehicle border
    def draw_vehicle_border(self, data_vehicle, id_sel, hlinewidth, hcolor):
        # data_vehicle: (ndarray) t x y theta v length width tag_segment tag_lane id (dim = N x 10, width > length)
        # id_sel: (ndarray) id of vehicles (dim = N)
        # hcolor: (tuple) rgb color
        # hlinewidth: (scalar) linewidth

        if ~isinstance(data_vehicle, np.ndarray):
            data_vehicle = np.array(data_vehicle)
        shape_data_v = data_vehicle.shape
        if len(shape_data_v) == 1:
            data_vehicle = np.reshape(data_vehicle, (1, -1))

        len_id = len(id_sel)

        for nidx_d in range(0, len_id):
            id_tmp = id_sel[nidx_d]
            if id_tmp == -1:
                continue

            idx_found = np.where(data_vehicle[:, -1] == id_tmp)
            idx_found = idx_found[0]

            data_vehicle_sel = data_vehicle[idx_found[0], :]
            self.draw_target_vehicle_border(data_vehicle_sel[1], data_vehicle_sel[2], data_vehicle_sel[3],
                                            data_vehicle_sel[6], data_vehicle_sel[5], hlinewidth, hcolor)

    # -----------------------------------------------------------------------------------------------------------------#
    # SUB-SCREEN (UPPER-RIGHT QUADRANT) -------------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#

    # SET PARAMETERS (SUB) --------------------------------------------------------------------------------------------#
    def set_screen_size_sub(self, screen_size):
        # screen_size: (tuple) screen size (dim = 2)
        self.screen_size_sub = screen_size

    def set_screen_height_sub(self, height):
        # screen_size: (scalar) screen height (dim = 1)
        ratio_hegiht2width = (self.pnt_max_sub[0] - self.pnt_min_sub[0]) / (self.pnt_max_sub[1] - self.pnt_min_sub[1])
        self.screen_size_sub = [ratio_hegiht2width * height, height]

    def set_pnts_range_sub(self, pnt_min, pnt_max):
        # pnt_min, pnt_max: (list or ndarray) point (dim = 2)
        self.pnt_min_sub = pnt_min
        self.pnt_max_sub = pnt_max

    def set_screen_alpha_sub(self):
        if len(self.pnt_min_sub) > 0 and len(self.pnt_max_sub) > 0:
            alpha1 = self.screen_size_sub[0] / (self.pnt_max_sub[0] - self.pnt_min_sub[0])
            alpha2 = self.screen_size_sub[1] / (self.pnt_max_sub[1] - self.pnt_min_sub[1])
        else:
            alpha1, alpha2 = 1.0, 1.0
        self.screen_alpha_sub = min(alpha1, alpha2)

    def set_quadrant_number_sub(self, quadrant_number):
        # quadrant_number: (scalar) 1, 2, 3, 4
        self.quadrant_number_sub = quadrant_number

    # UTILS (SUB) -----------------------------------------------------------------------------------------------------#
    def convert2pixel_sub(self, pnts):
        # pnts: (ndarray) points (dim = N x 2)

        if ~isinstance(pnts, np.ndarray):
            pnts = np.array(pnts)

        num_pnts = pnts.shape[0]
        pnts_conv = np.zeros((num_pnts, 2), dtype=np.float32)

        pnts_conv[:, 0] = pnts[:, 0] - np.repeat(self.pnt_min_sub[0], num_pnts, axis=0)
        pnts_conv[:, 1] = np.repeat(self.pnt_max_sub[1], num_pnts, axis=0) - pnts[:, 1]

        pnts_conv = pnts_conv * self.screen_alpha_sub

        ratio_hegiht2width = (self.pnt_max_sub[0] - self.pnt_min_sub[0]) / (self.pnt_max_sub[1] - self.pnt_min_sub[1])
        width_sub_tmp = self.screen_size_sub[0] - (ratio_hegiht2width * self.screen_size_sub[1])

        if self.quadrant_number_sub == 1:
            pnt_move = [1, 1]
        elif self.quadrant_number_sub == 2:
            pnt_move = [self.screen_size[0] - self.screen_size_sub[0] + width_sub_tmp - 1, 1]
        elif self.quadrant_number_sub == 3:
            pnt_move = [1, self.screen_size[1] - self.screen_size_sub[1] - 1]
        elif self.quadrant_number_sub == 4:
            pnt_move = [self.screen_size[0] - self.screen_size_sub[0] + width_sub_tmp - 1,
                        self.screen_size[1] - self.screen_size_sub[1] - 1]
        else:
            pnt_move = [0, 0]

        pnts_conv[:, 0] = pnts_conv[:, 0] + pnt_move[0]
        pnts_conv[:, 1] = pnts_conv[:, 1] + pnt_move[1]

        return pnts_conv

    # DRAW (SUB) -----------------------------------------------------------------------------------------------------#
    # Draw point (sub)
    def draw_pnt_sub(self, pnt, radius, hcolor):
        # pnt: (ndarray) point (dim = 1 x 2)
        # radius: (scalar) radius
        # hcolor: (tuple) rgb color

        if ~isinstance(pnt, np.ndarray):
            pnt = np.array(pnt)
        pnt = pnt.reshape(-1)

        pnt = pnt[0:2]
        pnt = np.reshape(pnt, (1, 2))

        # Convert to pixel space
        pnt_sel_conv = self.convert2pixel_sub(pnt)
        self.ctx.arc(pnt_sel_conv[0, 0], pnt_sel_conv[0, 1], radius, 0, 2 * math.pi)
        self.ctx.set_source_rgb(hcolor[0], hcolor[1], hcolor[2])
        # self.ctx.stroke()
        self.ctx.fill()

    # Draw points (sub)
    def draw_pnts_sub(self, pnts, radius, hcolor):
        # pnts: (ndarray) points (dim = N x 2)
        # radius: (scalar) radius
        # hcolor: (tuple) rgb color

        num_in = pnts.shape[0]

        for nidx_d in range(0, num_in):
            pnt_sel = pnts[nidx_d, :]
            pnt_sel = np.reshape(pnt_sel, (1, 2))
            self.draw_pnt_sub(pnt_sel, radius, hcolor)

    # Draw trajectory (sub)
    def draw_trajectory_sub(self, traj, hlinewidth, hcolor):
        # traj: (ndarray) trajectory (dim = N x 2)
        # hlinewidth: (scalar) linewidth
        # hcolor: (tuple) rgb color

        # Convert to pixel space
        traj_conv_tmp = self.convert2pixel_sub(traj)

        # Plot (cairo)
        for nidx_pnt in range(0, traj_conv_tmp.shape[0]):
            pnt_tmp = traj_conv_tmp[nidx_pnt, :]
            if nidx_pnt == 0:
                self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
            else:
                self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

        self.ctx.set_line_width(hlinewidth)
        self.ctx.set_source_rgb(hcolor[0], hcolor[1], hcolor[2])
        self.ctx.stroke()

    # Draw box range (sub)
    def draw_box_range(self, hcolor, hlinewidth):
        # hcolor: (tuple) rgb color
        # hlinewidth: (scalar) linewidth
        traj = np.zeros((5, 2), dtype=np.float32)
        traj[0, :] = [self.pnt_min_sub[0], self.pnt_min_sub[1]]
        traj[1, :] = [self.pnt_max_sub[0], self.pnt_min_sub[1]]
        traj[2, :] = [self.pnt_max_sub[0], self.pnt_max_sub[1]]
        traj[3, :] = [self.pnt_min_sub[0], self.pnt_max_sub[1]]
        traj[4, :] = [self.pnt_min_sub[0], self.pnt_min_sub[1]]
        self.draw_trajectory_sub(traj, hlinewidth, hcolor)

    # Draw track (sub)
    def draw_track_sub(self, pnts_poly_track):
        # pnts_poly_track: (list) points of track

        # Convert to pixel space
        pnts_pixel_track = []
        num_lane_seg = 0  # number of lane-segment
        for nidx_seg in range(0, len(pnts_poly_track)):
            seg_sel = pnts_poly_track[nidx_seg]

            pnts_pixel_seg = []
            for nidx_lane in range(0, len(seg_sel)):
                num_lane_seg = num_lane_seg + 1
                pnts_tmp = seg_sel[nidx_lane]
                pnts_conv_tmp = self.convert2pixel_sub(pnts_tmp)
                pnts_pixel_seg.append(pnts_conv_tmp)

            pnts_pixel_track.append(pnts_pixel_seg)

        # Plot track
        for nidx_seg in range(0, len(pnts_pixel_track)):
            pnts_pixel_seg = pnts_pixel_track[nidx_seg]

            # Plot lane-segment
            for nidx_lane in range(0, len(pnts_pixel_seg)):
                # Pnts on lane-segment
                pnts_pixel_lane = pnts_pixel_seg[nidx_lane]

                for nidx_pnt in range(0, pnts_pixel_lane.shape[0]):
                    pnt_tmp = pnts_pixel_lane[nidx_pnt, :]
                    if nidx_pnt == 0:
                        self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
                    else:
                        self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

                pnt_0_tmp = pnts_pixel_lane[0, :]
                self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

                # Set (fill) color
                cmap_lane = get_rgb("Dim Gray")

                # Plot (cairo)
                self.ctx.set_line_width(0.3)
                self.ctx.set_source_rgb(cmap_lane[0], cmap_lane[1], cmap_lane[2])
                self.ctx.fill_preserve()
                self.ctx.set_source_rgb(0, 0, 0)
                self.ctx.set_line_width(0.3)
                self.ctx.stroke()
