# UTILITY-FUNCTIONS (BASIC)
# Function list
#       make_numpy_array(x, keep_1dim=False)
#       get_rotated_pnts_rt(pnts_in, cp, theta)
#       get_rotated_pnts_tr(pnts_in, cp, theta)
#       get_box_pnts(x, y, theta, length, width)
#       get_box_pnts_precise(x, y, theta, length, width, nx=10, ny=10)
#       get_m_pnts(x, y, theta, length, nx=10)
#       get_intp_pnts(pnts_in, num_intp)
#       get_intp_pnts_wrt_dist(pnts_in, num_intp, dist_intv)
#       get_pnts_carshape(x, y, theta, length, width)
#       get_pnts_arrow(x, y, theta, ax, ay, bx, by)
#       inpolygon(xq, yq, xv, yv)
#       get_dist_point2line(pnt_i, pnt_a, pnt_b)
#       get_closest_pnt(pnt_i, pnts_line)
#       get_closest_pnt_intp(pnt_i, pnts_line, num_intp)
#       get_l2_loss(x1, x2)
#       angle_handle(q)
#       norm(vec)
#       interpolate_w_ratio(p1, p2, r_a, r_b)
#       interpolate_trajectory_cubic_spline(traj_in, alpha=2)
#       set_vector_in_range(vec_in, vec_min, vec_max)


import numpy as np
import math
from scipy import interpolate

from matplotlib import path


# Make numpy-array
def make_numpy_array(x, keep_1dim=False):
    # x: (list) input data
    if ~isinstance(x, np.ndarray):
        x = np.array(x)
    shape_x = x.shape
    if keep_1dim:
        x = x.reshape(-1)
    else:
        if len(shape_x) == 1:
            x = np.reshape(x, (1, -1))

    return x


# Rotate points (Rotation --> Transition)
def get_rotated_pnts_rt(pnts_in, cp, theta):
    # pnts_in: (ndarray) (dim: N x 2)
    # cp: (ndarray) (dim: 2)
    # theta: scalar (rad)

    pnts_in = make_numpy_array(pnts_in, keep_1dim=False)
    cp = make_numpy_array(cp, keep_1dim=True)

    num_pnts = pnts_in.shape[0]

    R = np.array([[+math.cos(theta), +math.sin(theta)], [-math.sin(theta), +math.cos(theta)]], dtype=np.float32)
    cp_tmp = np.reshape(cp, (1, 2))
    cp_tmp = np.tile(cp_tmp, (num_pnts, 1))

    pnts_out = np.matmul(pnts_in, R) + cp_tmp

    return pnts_out


# Rotate points (Transition --> Rotation)
def get_rotated_pnts_tr(pnts_in, cp, theta):
    # pnts_in: (ndarray) (dim: N x 2)
    # cp: (ndarray) (dim: 2)
    # theta: scalar (rad)

    pnts_in = make_numpy_array(pnts_in, keep_1dim=False)
    cp = make_numpy_array(cp, keep_1dim=True)

    num_pnts = pnts_in.shape[0]

    R = np.array([[+math.cos(theta), +math.sin(theta)], [-math.sin(theta), +math.cos(theta)]], dtype=np.float32)
    cp_tmp = np.reshape(cp, (1, 2))
    cp_tmp = np.tile(cp_tmp, (num_pnts, 1))

    pnts_out = np.matmul(pnts_in + cp_tmp, R)

    return pnts_out


# Return points of box-shape
def get_box_pnts(x, y, theta, length, width):
    # x, y, theta, length, width: (scalar) position-x, position-y, heading, length(dx), width(dy)

    rx, ry = length / 2.0, width / 2.0
    p0, p1, p2, p3 = [-rx, -ry], [+rx, -ry], [+rx, +ry], [-rx, +ry]

    pnts_box_ = np.array([[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]]], dtype=np.float32)
    pnts_box = get_rotated_pnts_rt(pnts_box_, [x, y], theta)

    return pnts_box


# Return points of box-shape (precise)
def get_box_pnts_precise(x, y, theta, length, width, nx=10, ny=10):
    # x, y, theta, length, width: (scalar) position-x, position-y, heading, length(dx), width(dy)
    # nx, ny: (scalar) number of interpolation

    rx, ry = length / 2.0, width / 2.0

    l0_x = np.linspace(-rx, +rx, num=nx, dtype=np.float32)
    l0_y = -ry*np.ones((nx,), dtype=np.float32)
    l1_x = +rx*np.ones((ny,), dtype=np.float32)
    l1_y = np.linspace(-ry, +ry, num=ny, dtype=np.float32)
    l2_x = np.linspace(+rx, -rx, num=nx, dtype=np.float32)
    l2_y = +ry * np.ones((nx,), dtype=np.float32)
    l3_x = -rx * np.ones((ny,), dtype=np.float32)
    l3_y = np.linspace(-ry, +ry, num=ny, dtype=np.float32)

    pnts_x = np.concatenate((l0_x, l1_x, l2_x, l3_x), axis=0)
    pnts_y = np.concatenate((l0_y, l1_y, l2_y, l3_y), axis=0)
    pnts_x = pnts_x.reshape((-1, 1))
    pnts_y = pnts_y.reshape((-1, 1))

    pnts_box_ = np.concatenate((pnts_x, pnts_y), axis=1)
    pnts_box = get_rotated_pnts_rt(pnts_box_, [x, y], theta)

    return pnts_box


# Return middle points
def get_m_pnts(x, y, theta, length, nx=10):
    # x, y, theta, length: (scalar) position-x, position-y, heading, length(dx)
    # nx: (scalar) number of interpolation

    rx = length / 2.0

    pnts_x = np.linspace(-rx, +rx, num=nx, dtype=np.float32)
    pnts_y = np.zeros((nx, ), dtype=np.float32)
    pnts_x = pnts_x.reshape((-1, 1))
    pnts_y = pnts_y.reshape((-1, 1))
    pnts_m_ = np.concatenate((pnts_x, pnts_y), axis=1)
    pnts_m = get_rotated_pnts_rt(pnts_m_, [x, y], theta)

    return pnts_m


# Get interpolate points
def get_intp_pnts(pnts_in, num_intp):
    # pnts_in: (ndarray) points (dim = N x 2)
    # num_intp: (scalar) interpolation number

    pnts_in = make_numpy_array(pnts_in, keep_1dim=False)

    pnt_x_tmp = pnts_in[:, 0].reshape(-1)
    pnt_y_tmp = pnts_in[:, 1].reshape(-1)

    do_flip = 0
    if (pnt_x_tmp[1] - pnt_x_tmp[0]) < 0:
        pnt_x_tmp = np.flip(pnt_x_tmp, axis=0)
        pnt_y_tmp = np.flip(pnt_y_tmp, axis=0)
        do_flip = 1

    x_range = np.linspace(min(pnt_x_tmp), max(pnt_x_tmp), num=num_intp)
    y_intp = np.interp(x_range, pnt_x_tmp, pnt_y_tmp)

    pnts_intp = np.zeros((num_intp, 2), dtype=np.float32)
    pnts_intp[:, 0] = x_range
    pnts_intp[:, 1] = y_intp

    if do_flip == 1:
        pnts_intp = np.flip(pnts_intp, axis=0)

    return pnts_intp


# Get interpolate points (w.r.t. dist)
def get_intp_pnts_wrt_dist(pnts_in, num_intp, dist_intv):
    # pnts_in: (ndarray) points (dim = N x 2)
    # num_intp: (scalar) interpolation number
    # dist_intv: (scalar) distance interval

    pnts_in = make_numpy_array(pnts_in, keep_1dim=False)

    pnts_intp = get_intp_pnts(pnts_in, num_intp)
    len_pnts_intp = pnts_intp.shape[0]
    diff_tmp = pnts_intp[np.arange(0, len_pnts_intp - 1), 0:2] - pnts_intp[np.arange(1, len_pnts_intp), 0:2]
    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))

    dist_sum, cnt_intv = 0, 0
    idx_intv = np.zeros((20000, ), dtype=np.int32)
    for nidx_d in range(0, dist_tmp.shape[0]):
        dist_sum = dist_sum + dist_tmp[nidx_d]
        if dist_sum >= (cnt_intv + 1) * dist_intv:
            cnt_intv = cnt_intv + 1
            idx_intv[cnt_intv] = nidx_d

    idx_intv = idx_intv[0:cnt_intv + 1]

    pnts_intv = pnts_intp[idx_intv, :]

    return pnts_intv


# Return points of car-shape
def get_pnts_carshape(x, y, theta, length, width):
    # x, y, theta, length, width: (scalar) position-x, position-y, heading, length(dx), width(dy)

    rx, ry = length / 2.0, width / 2.0
    lx = rx * 1.5
    th = math.atan2(ry, math.sqrt(lx * lx - ry * ry))

    p0 = np.array([rx - lx, 0.0], dtype=np.float64)
    p1 = np.array([-rx, +ry], dtype=np.float64)
    p2 = np.array([-rx, -ry], dtype=np.float64)
    # p3 = np.array([lx*math.cos(th) - lx + rx, -ry], dtype=np.float64)
    # p4 = np.array([lx*math.cos(th) - lx + rx, +ry], dtype=np.float64)

    # pnts curve connecting p3 to p4
    num_pnts_curve = 9
    th_curve = np.linspace(-th, +th, num_pnts_curve)
    pnts_curve = np.zeros((num_pnts_curve, 2), dtype=np.float64)
    pnts_curve[:, 0] = lx*np.cos(th_curve) + p0[0]
    pnts_curve[:, 1] = lx*np.sin(th_curve)

    # pnts (raw)
    pnts_vehicle = np.zeros((num_pnts_curve + 2, 2), dtype=np.float64)
    pnts_vehicle[0, :] = p1
    pnts_vehicle[1, :] = p2
    pnts_vehicle[2:, :] = pnts_curve

    # pnts (rotated)
    pnt_cp = np.array([x, y], dtype=np.float64)
    pnts_vehicle_r = get_rotated_pnts_rt(pnts_vehicle, pnt_cp, theta)

    return pnts_vehicle_r


# Return points of arrow
def get_pnts_arrow(x, y, theta, ax, ay, bx, by):
    # x, y, theta: (scalar) position-x, position-y, heading
    # ax, ay, bx, by: (scalar) arrow parameters

    # bx < ax
    pnts_arrow = np.zeros((7, 2), dtype=np.float32)
    pnts_arrow[0, :] = [-ax, +ay]
    pnts_arrow[1, :] = [ax - bx, +ay]
    pnts_arrow[2, :] = [ax - bx, ay + by]
    pnts_arrow[3, :] = [ax, 0]
    pnts_arrow[4, :] = [ax - bx, -(ay + by)]
    pnts_arrow[5, :] = [ax - bx, -ay]
    pnts_arrow[6, :] = [-ax, -ay]

    # pnts (rotated)
    pnt_cp = np.array([x, y], dtype=np.float32)
    pnts_vehicle_r = get_rotated_pnts_rt(pnts_arrow, pnt_cp, theta)

    return pnts_vehicle_r


# Check whether point is in the polygon
def inpolygon(xq, yq, xv, yv):
    # xq, yq: (ndarray) points query (dim = Nq)
    # xv, yv: (ndarray) points vertex (dim = Nv)

    xq = make_numpy_array(xq, keep_1dim=True)
    yq = make_numpy_array(yq, keep_1dim=True)
    xv = make_numpy_array(xv, keep_1dim=True)
    yv = make_numpy_array(yv, keep_1dim=True)

    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(-1)


# Get distance from point to line
def get_dist_point2line(pnt_i, pnt_a, pnt_b):
    # pnt_i, pnt_a, pnt_b: (ndarray) point (dim = 2)

    pnt_i = make_numpy_array(pnt_i, keep_1dim=True)
    pnt_a = make_numpy_array(pnt_a, keep_1dim=True)
    pnt_b = make_numpy_array(pnt_b, keep_1dim=True)

    diff_ab = pnt_a[0:2] - pnt_b[0:2]
    dist_ab = np.sqrt(diff_ab[0]*diff_ab[0] + diff_ab[1]*diff_ab[1])
    if dist_ab == 0:
        # PNT_A == PNT_B
        diff_ia = pnt_i[0:2] - pnt_a[0:2]
        mindist = norm(diff_ia[0:2])
        minpnt = pnt_a
    else:
        # OTHERWISE
        vec_a2b = pnt_b[0:2] - pnt_a[0:2]
        vec_b2a = pnt_a[0:2] - pnt_b[0:2]
        vec_a2i = pnt_i[0:2] - pnt_a[0:2]
        vec_b2i = pnt_i[0:2] - pnt_b[0:2]

        dot_tmp1 = vec_a2i[0]*vec_a2b[0] + vec_a2i[1]*vec_a2b[1]
        dot_tmp2 = vec_b2i[0]*vec_b2a[0] + vec_b2i[1]*vec_b2a[1]
        if dot_tmp1 < 0:
            minpnt = pnt_a
        elif dot_tmp2 < 0:
            minpnt = pnt_b
        else:
            len_a2b = norm(vec_a2b[0:2])
            minpnt = pnt_a + dot_tmp1 * vec_a2b / len_a2b / len_a2b

        diff_tmp = minpnt[0:2] - pnt_i[0:2]
        mindist = np.sqrt(diff_tmp[0]*diff_tmp[0] + diff_tmp[1]*diff_tmp[1])

    return mindist, minpnt


# Get the closest point
def get_closest_pnt(pnt_i, pnts_line):
    # pnt_i: (ndarray) point (x, y)
    # pnts_line: (ndarray) points (dim = N x 2)

    pnt_i = make_numpy_array(pnt_i, keep_1dim=True)
    pnts_line = make_numpy_array(pnts_line, keep_1dim=False)

    m = pnts_line.shape[0]
    mindist = 1e8

    pnt = []
    for nidx_i in range(0, m - 1):
        pnt_a = pnts_line[nidx_i, :]
        pnt_b = pnts_line[nidx_i + 1, :]

        [dist, cpnt] = get_dist_point2line(pnt_i, pnt_a, pnt_b)
        if dist < mindist:
            mindist = dist
            pnt = cpnt

    return pnt, mindist


# Get the closest point using interpolate
def get_closest_pnt_intp(pnt_i, pnts_line, num_intp=100):
    # pnt_i: (ndarray) point (x, y, dim=2)
    # pnts_line: (ndarray) points (dim = N x 2)
    # num_intp: (scalar) number of interpolation

    pnt_i = make_numpy_array(pnt_i, keep_1dim=True)
    pnts_line = make_numpy_array(pnts_line, keep_1dim=False)

    len_pnts_line = pnts_line.shape[0]
    pnt_i_r = np.reshape(pnt_i[0:2], (1, 2))
    diff_tmp = np.tile(pnt_i_r, (len_pnts_line, 1)) - pnts_line[:, 0:2]
    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
    idx_min = np.argmin(dist_tmp, axis=0)
    pnt_cur = pnts_line[idx_min, 0:2]
    pnt_cur = np.reshape(pnt_cur, (1, -1))

    if idx_min == 0:
        pnt_next = pnts_line[idx_min + 1, 0:2]
        pnt_next = np.reshape(pnt_next, (1, -1))
        pnts_tmp = np.concatenate((pnt_cur, pnt_next), axis=0)
        pnts_tmp_intp = get_intp_pnts(pnts_tmp, num_intp)
    elif idx_min == (len_pnts_line - 1):
        pnt_prev = pnts_line[idx_min - 1, 0:2]
        pnt_prev = np.reshape(pnt_prev, (1, -1))
        pnts_tmp = np.concatenate((pnt_prev, pnt_cur), axis=0)
        pnts_tmp_intp = get_intp_pnts(pnts_tmp, num_intp)
    else:
        pnt_prev = pnts_line[idx_min - 1, 0:2]
        pnt_prev = np.reshape(pnt_prev, (1, -1))
        pnts_tmp1 = np.concatenate((pnt_prev, pnt_cur), axis=0)
        pnts_tmp_intp1 = get_intp_pnts(pnts_tmp1, num_intp)
        pnt_next = pnts_line[idx_min + 1, 0:2]
        pnt_next = np.reshape(pnt_next, (1, -1))
        pnts_tmp2 = np.concatenate((pnt_cur, pnt_next), axis=0)
        pnts_tmp_intp2 = get_intp_pnts(pnts_tmp2, num_intp)
        pnts_tmp_intp = np.concatenate((pnts_tmp_intp1, pnts_tmp_intp2), axis=0)

    diff_tmp_new = np.tile(pnt_i_r, (pnts_tmp_intp.shape[0], 1)) - pnts_tmp_intp[:, 0:2]
    dist_tmp_new = np.sqrt(np.sum(diff_tmp_new * diff_tmp_new, axis=1))
    idx_cur_new = np.argmin(dist_tmp_new, axis=0)

    pnt_out = pnts_tmp_intp[idx_cur_new, :]
    mindist_out = dist_tmp_new[idx_cur_new]

    return pnt_out, mindist_out


# Get l2-loss
def get_l2_loss(x1, x2):
    # x1, x2: (ndarray, dim = N x ?)

    x1 = make_numpy_array(x1, keep_1dim=False)
    x2 = make_numpy_array(x2, keep_1dim=False)

    diff_tmp = x1 - x2
    l2_array = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
    l2_out = np.sum(l2_array)
    return l2_array, l2_out


# Adjust angle
def angle_handle(q):
    # q: (ndarray) angle

    q = make_numpy_array(q, keep_1dim=True)
    q_new = np.copy(q)

    idx_found_upper = np.where(q > math.pi)
    idx_found_upper = idx_found_upper[0]

    idx_found_lower = np.where(q < -math.pi)
    idx_found_lower = idx_found_lower[0]

    if len(idx_found_upper) > 0:
        q_new[idx_found_upper] = q[idx_found_upper] - 2 * math.pi
    if len(idx_found_lower) > 0:
        q_new[idx_found_lower] = q[idx_found_lower] + 2 * math.pi

    return q_new


# Get l2-norm
def norm(vec):
    # vec: (ndarray) vector (dim = 2)

    vec = make_numpy_array(vec, keep_1dim=True)
    len_vec = np.sqrt(vec[0]*vec[0] + vec[1]*vec[1])

    return len_vec


# Interpolate two points w.r.t ratio
def interpolate_w_ratio(p1, p2, r_a, r_b):
    # p1, p2: (ndarray) point (dim = 2)
    # r_a, r_b: scalar

    p1 = make_numpy_array(p1, keep_1dim=True)
    p2 = make_numpy_array(p2, keep_1dim=True)

    x_tmp = (r_b * p1[0] + r_a * p2[0]) / (r_a + r_b)
    y_tmp = (r_b * p1[1] + r_a * p2[1]) / (r_a + r_b)
    p3 = np.array([x_tmp, y_tmp], dtype=np.float32)
    return p3


# Interpolate trajectory (cubic-spline)
def interpolate_trajectory_cubic_spline(traj_in, alpha=2):

    traj_in = make_numpy_array(traj_in, keep_1dim=False)

    len_traj = traj_in.shape[0]
    dim_traj = traj_in.shape[1]
    len_traj_out = len_traj * alpha

    t_in = np.arange(0, len_traj)
    t_out = np.linspace(start=0, stop=len_traj, num=len_traj_out)
    traj_out = np.zeros((len_traj_out, dim_traj), dtype=np.float32)

    for nidx_d in range(0, dim_traj):
        y = traj_in[:, nidx_d]
        tck = interpolate.splrep(t_in, y, s=0)
        ynew = interpolate.splev(t_out, tck, der=0)
        traj_out[:, nidx_d] = ynew

    return traj_out


# Set vector in range
def set_vector_in_range(vec_in, vec_min, vec_max):
    # vec_in: (ndarray) vector-in (dim = N)
    # vec_min, vec_max: (ndarray) min, max (dim = N)

    vec_in = make_numpy_array(vec_in, keep_1dim=True)
    vec_min = make_numpy_array(vec_min, keep_1dim=True)
    vec_max = make_numpy_array(vec_max, keep_1dim=True)

    vec_new = np.maximum(vec_in, vec_min)
    vec_new = np.minimum(vec_new, vec_max)

    return vec_new
