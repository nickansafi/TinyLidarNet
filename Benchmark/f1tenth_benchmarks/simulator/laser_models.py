# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""
Prototype of Utility functions and classes for simulating 2D LIDAR scans
Author: Hongrui Zheng
"""

import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt as edt
from PIL import Image
import os
import yaml


def get_dt(bitmap, resolution):
    """
    Distance transformation, returns the distance matrix from the input bitmap.
    Uses scipy.ndimage, cannot be JITted.

        Args:
            bitmap (numpy.ndarray, (n, m)): input binary bitmap of the environment, where 0 is obstacles, and 255 (or anything > 0) is freespace
            resolution (float): resolution of the input bitmap (m/cell)

        Returns:
            dt (numpy.ndarray, (n, m)): output distance matrix, where each cell has the corresponding distance (in meters) to the closest obstacle
    """
    dt = resolution * edt(bitmap)
    return dt

@njit(cache=True)
def xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution):
    """
    Translate (x, y) coordinate into (r, c) in the matrix

        Args:
            x (float): coordinate in x (m)
            y (float): coordinate in y (m)
            orig_x (float): x coordinate of the map origin (m)
            orig_y (float): y coordinate of the map origin (m)
        
        Returns:
            r (int): row number in the transform matrix of the given point
            c (int): column number in the transform matrix of the given point
    """
    # translation
    x_trans = x - orig_x
    y_trans = y - orig_y

    # rotation
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c

    # clip the state to be a cell
    if x_rot < 0 or x_rot >= width * resolution or y_rot < 0 or y_rot >= height * resolution:
        c = -1
        r = -1
    else:
        c = int(x_rot/resolution)
        r = int(y_rot/resolution)

    return r, c

@njit(cache=True)
def distance_transform(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt):
    """
    Look up corresponding distance in the distance matrix

        Args:
            x (float): x coordinate of the lookup point
            y (float): y coordinate of the lookup point
            orig_x (float): x coordinate of the map origin (m)
            orig_y (float): y coordinate of the map origin (m)

        Returns:
            distance (float): corresponding shortest distance to obstacle in meters
    """
    r, c = xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution)
    distance = dt[r, c]
    return distance

@njit(cache=True)
def trace_ray(x, y, theta_index, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range):
    """
    Find the length of a specific ray at a specific scan angle theta
    Purely math calculation and loops, should be JITted.

        Args:
            x (float): current x coordinate of the ego (scan) frame
            y (float): current y coordinate of the ego (scan) frame
            theta_index(int): current index of the scan beam in the scan range
            sines (numpy.ndarray (n, )): pre-calculated sines of the angle array
            cosines (numpy.ndarray (n, )): pre-calculated cosines ...

        Returns:
            total_distance (float): the distance to first obstacle on the current scan beam
    """
    
    # int casting, and index precal trigs
    theta_index_ = int(theta_index)
    s = sines[theta_index_]
    c = cosines[theta_index_]

    # distance to nearest initialization
    dist_to_nearest = distance_transform(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt)
    total_dist = dist_to_nearest

    # ray tracing iterations
    while dist_to_nearest > eps and total_dist <= max_range:
        # move in the direction of the ray by dist_to_nearest
        x += dist_to_nearest * c
        y += dist_to_nearest * s

        # update dist_to_nearest for current point on ray
        # also keeps track of total ray length
        dist_to_nearest = distance_transform(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt)
        total_dist += dist_to_nearest

    if total_dist > max_range:
        total_dist = max_range
    
    return total_dist

@njit(cache=True)
def get_scan(pose, theta_dis, fov, num_beams, theta_index_increment, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range):
    """
    Perform the scan for each discretized angle of each beam of the laser, loop heavy, should be JITted

        Args:
            pose (numpy.ndarray(3, )): current pose of the scan frame in the map
            theta_dis (int): number of steps to discretize the angles between 0 and 2pi for look up
            fov (float): field of view of the laser scan
            num_beams (int): number of beams in the scan
            theta_index_increment (float): increment between angle indices after discretization

        Returns:
            scan (numpy.ndarray(n, )): resulting laser scan at the pose, n=num_beams
    """
    # empty scan array init
    scan = np.empty((num_beams,))

    # make theta discrete by mapping the range [-pi, pi] onto [0, theta_dis]
    theta_index = theta_dis * (pose[2] - fov/2.)/(2. * np.pi)

    # make sure it's wrapped properly
    theta_index = np.fmod(theta_index, theta_dis)
    while (theta_index < 0):
        theta_index += theta_dis

    # sweep through each beam
    for i in range(0, num_beams):
        # trace the current beam
        scan[i] = trace_ray(pose[0], pose[1], theta_index, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range)

        # increment the beam index
        theta_index += theta_index_increment

        # make sure it stays in the range [0, theta_dis)
        while theta_index >= theta_dis:
            theta_index -= theta_dis

    return scan

@njit(cache=True, error_model='numpy')
def check_ttc_jit(scan, vel, scan_angles, cosines, side_distances, ttc_thresh):
    """
    Checks the iTTC of each beam in a scan for collision with environment

    Args:
        scan (np.ndarray(num_beams, )): current scan to check
        vel (float): current velocity
        scan_angles (np.ndarray(num_beams, )): precomped angles of each beam
        cosines (np.ndarray(num_beams, )): precomped cosines of the scan angles
        side_distances (np.ndarray(num_beams, )): precomped distances at each beam from the laser to the sides of the car
        ttc_thresh (float): threshold for iTTC for collision

    Returns:
        in_collision (bool): whether vehicle is in collision with environment
        collision_angle (float): at which angle the collision happened
    """
    in_collision = False
    if vel != 0.0:
        num_beams = scan.shape[0]
        for i in range(num_beams):
            proj_vel = vel*cosines[i]
            ttc = (scan[i] - side_distances[i])/proj_vel
            if (ttc < ttc_thresh) and (ttc >= 0.0):
                in_collision = True
                break
    else:
        in_collision = False

    return in_collision

@njit(cache=True)
def cross(v1, v2):
    """
    Cross product of two 2-vectors

    Args:
        v1, v2 (np.ndarray(2, )): input vectors

    Returns:
        crossproduct (float): cross product
    """
    return v1[0]*v2[1]-v1[1]*v2[0]

@njit(cache=True)
def are_collinear(pt_a, pt_b, pt_c):
    """
    Checks if three points are collinear in 2D

    Args:
        pt_a, pt_b, pt_c (np.ndarray(2, )): points to check in 2D

    Returns:
        col (bool): whether three points are collinear
    """
    tol = 1e-8
    ba = pt_b - pt_a
    ca = pt_a - pt_c
    col = np.fabs(cross(ba, ca)) < tol
    return col

@njit(cache=True)
def get_range(pose, beam_theta, va, vb):
    """
    Get the distance at a beam angle to the vector formed by two of the four vertices of a vehicle

    Args:
        pose (np.ndarray(3, )): pose of the scanning vehicle
        beam_theta (float): angle of the current beam (world frame)
        va, vb (np.ndarray(2, )): the two vertices forming an edge

    Returns:
        distance (float): smallest distance at beam theta from scanning pose to edge
    """
    o = pose[0:2]
    v1 = o - va
    v2 = vb - va
    v3 = np.array([np.cos(beam_theta + np.pi/2.), np.sin(beam_theta + np.pi/2.)])

    denom = v2.dot(v3)
    distance = np.inf

    if np.fabs(denom) > 0.0:
        d1 = cross(v2, v1) / denom
        d2 = v1.dot(v3) / denom
        if d1 >= 0.0 and d2 >= 0.0 and d2 <= 1.0:
            distance = d1
    elif are_collinear(o, va, vb):
        da = np.linalg.norm(va - o)
        db = np.linalg.norm(vb - o)
        distance = min(da, db)

    return distance

@njit(cache=True)
def get_blocked_view_indices(pose, vertices, scan_angles):
    """
    Get the indices of the start and end of blocked fov in scans by another vehicle

    Args:
        pose (np.ndarray(3, )): pose of the scanning vehicle
        vertices (np.ndarray(4, 2)): four vertices of a vehicle pose
        scan_angles (np.ndarray(num_beams, )): corresponding beam angles
    """
    # find four vectors formed by pose and 4 vertices:
    vecs = vertices - pose[:2]
    vec_sq = np.square(vecs)
    norms = np.sqrt(vec_sq[:, 0] + vec_sq[:, 1])
    unit_vecs = vecs / norms.reshape(norms.shape[0], 1)

    # find angles between all four and pose vector
    ego_x_vec = np.array([[np.cos(pose[2])], [np.sin(pose[2])]])
    
    angles_with_x = np.empty((4, ))
    for i in range(4):
        angle = np.arctan2(ego_x_vec[1], ego_x_vec[0]) - np.arctan2(unit_vecs[i, 1], unit_vecs[i, 0])
        if angle > np.pi:
            angle = angle - 2*np.pi
        elif angle < -np.pi:
            angle = angle + 2*np.pi
        angles_with_x[i] = -angle[0]

    ind1 = int(np.argmin(np.abs(scan_angles - angles_with_x[0])))
    ind2 = int(np.argmin(np.abs(scan_angles - angles_with_x[1])))
    ind3 = int(np.argmin(np.abs(scan_angles - angles_with_x[2])))
    ind4 = int(np.argmin(np.abs(scan_angles - angles_with_x[3])))
    inds = [ind1, ind2, ind3, ind4]
    return min(inds), max(inds)


@njit(cache=True)
def ray_cast(pose, scan, scan_angles, vertices):
    """
    Modify a scan by ray casting onto another agent's four vertices

    Args:
        pose (np.ndarray(3, )): pose of the vehicle performing scan
        scan (np.ndarray(num_beams, )): original scan to modify
        scan_angles (np.ndarray(num_beams, )): corresponding beam angles
        vertices (np.ndarray(4, 2)): four vertices of a vehicle pose
    
    Returns:
        new_scan (np.ndarray(num_beams, )): modified scan
    """
    # pad vertices so loops around
    looped_vertices = np.empty((5, 2))
    looped_vertices[0:4, :] = vertices
    looped_vertices[4, :] = vertices[0, :]

    min_ind, max_ind = get_blocked_view_indices(pose, vertices, scan_angles)
    # looping over beams
    for i in range(min_ind, max_ind + 1):
        # looping over vertices
        for j in range(4):
            # check if original scan is longer than ray casted distance
            scan_range = get_range(pose, pose[2]+scan_angles[i], looped_vertices[j,:], looped_vertices[j+1,:])
            if scan_range < scan[i]:
                scan[i] = scan_range
    return scan

class ScanSimulator2D(object):
    """
    2D LIDAR scan simulator class

    Init params:
        num_beams (int): number of beams in the scan
        fov (float): field of view of the laser scan
        eps (float, default=0.0001): ray tracing iteration termination condition
        theta_dis (int, default=2000): number of steps to discretize the angles between 0 and 2pi for look up
        max_range (float, default=30.0): maximum range of the laser
    """

    def __init__(self, num_beams, fov, map_name, random_seed, eps=0.0001, theta_dis=2000, max_range=30.0):
        # initialization 
        self.num_beams = num_beams
        self.fov = fov
        self.eps = eps
        self.theta_dis = theta_dis
        self.max_range = max_range
        self.angle_increment = self.fov / (self.num_beams - 1)
        self.theta_index_increment = theta_dis * self.angle_increment / (2. * np.pi)
        self.orig_c = None
        self.orig_s = None
        self.orig_x = None
        self.orig_y = None
        self.map_height = None
        self.map_width = None
        self.map_resolution = None
        self.dt = None
        self.scan_rng = np.random.default_rng(seed=random_seed)
        
        # precomputing corresponding cosines and sines of the angle array
        theta_arr = np.linspace(0.0, 2*np.pi, num=theta_dis)
        self.sines = np.sin(theta_arr)
        self.cosines = np.cos(theta_arr)
        self.set_map(map_name)
    
    def set_map(self, map_name):
        map_path = os.getcwd()[0:-len(os.getcwd().lower().split("tinylidarnet")[-1])]+"/Benchmark/maps/" + map_name + ".yaml"
        
        with open(map_path, 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']
                map_img_path = os.getcwd()[0:-len(os.getcwd().lower().split("tinylidarnet")[-1])]+"/Benchmark/maps/" + map_metadata['image']
            except yaml.YAMLError as ex:
                print(ex)

        # load map image
        self.map_img = np.array(Image.open(map_img_path).convert('L').transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)

        # grayscale -> binary
        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]

        # calculate map parameters
        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]
        self.orig_s = np.sin(self.origin[2])
        self.orig_c = np.cos(self.origin[2])

        # get the distance transform
        self.dt = get_dt(self.map_img, self.map_resolution)

        return True

    def scan(self, pose, std_dev=0.01):
        """
        Perform simulated 2D scan by pose on the given map

            Args:
                pose (numpy.ndarray (3, )): pose of the scan frame (x, y, theta)
                rng (numpy.random.Generator): random number generator to use for whitenoise in scan, or None
                std_dev (float, default=0.01): standard deviation of the generated whitenoise in the scan

            Returns:
                scan (numpy.ndarray (n, )): data array of the laserscan, n=num_beams

            Raises:
                ValueError: when scan is called before a map is set
        """
        
        if self.map_height is None:
            raise ValueError('Map is not set for scan simulator.')
        
        scan = get_scan(pose, self.theta_dis, self.fov, self.num_beams, self.theta_index_increment, self.sines, self.cosines, self.eps, self.orig_x, self.orig_y, self.orig_c, self.orig_s, self.map_height, self.map_width, self.map_resolution, self.dt, self.max_range)

        noise = self.scan_rng.normal(0., std_dev, size=self.num_beams)
        scan += noise
            
        return scan

    def get_increment(self):
        return self.angle_increment

    def check_location(self, pose):
        if check_bounds(pose[0], pose[1], self.orig_x, self.orig_y, self.orig_c, self.orig_s, self.map_height, self.map_width, self.map_resolution):
            return True
        d = distance_transform(pose[0], pose[1], self.orig_x, self.orig_y, self.orig_c, self.orig_s, self.map_height, self.map_width, self.map_resolution, self.dt)
        if d < 0.001: #1mm
            return True
        return False


@njit(cache=True)
def check_bounds(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution):

    # translation
    x_trans = x - orig_x
    y_trans = y - orig_y

    # rotation
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c

    # clip the state to be a cell
    if x_rot < 0 or x_rot >= width * resolution or y_rot < 0 or y_rot >= height * resolution:
        return True
    return False

