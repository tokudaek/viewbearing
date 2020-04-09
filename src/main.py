#!/usr/bin/env python3
"""Find the view based from a location and a view bearing wrt the true north
"""

import argparse
import logging
import time
import os
from os.path import join as pjoin
from logging import info
import inspect

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_osm_file(cityname):
    """Get OSM file from city name

    Args:
    cityname(str): name of the region of interest

    Returns:
    path to the osm file
    """
    info(inspect.stack()[0][3] + '()')

    return ''

##########################################################
def parse_osm_network(osmpath):
    """Parse streets from OSM file and convert to an igraph obj.
    A circular street segment may consist of multiple nodes.

    Args:
    osmpath(str): osm file path

    Returns:
    igraph.Graph: network
    """

    info(inspect.stack()[0][3] + '()')
    return None

##########################################################
def get_nearest_road(point, network):
    """Get id of the nearest street in the network

    Args:
    point(np.array): array of shape (2,)
    network(igraph.Graph): network in osm format

    Returns:
    int: from list of network
    """
    info(inspect.stack()[0][3] + '()')
    return -1

##########################################################
def get_road_angle(roadid, network):
    """Get road orientation

    Args:
    roadid(int): road id
    network(igraph.Graph): network obj

    Returns:
    float: road orientation, alpha in [0, 180]. Note that instead of alpha, it may be 180-alpha
    """

    info(inspect.stack()[0][3] + '()')

    return 0

##########################################################
def plot_vectors(viewangle, roadangle, ax):
    """Plot the two vectors

    Args:
    viewangle(float): view orientation in radians
    roadangle(float): road orientation in radians
    ax(mpl.Axis): axis to plot
    """
    info(inspect.stack()[0][3] + '()')

    pass

##########################################################
def plot_top_view(point, viewangle, roadangle, network, ax):
    """Plot the view cone along with the network

    Args:
    point(np.array): array of shape (2,)
    viewangle(float): view orientation in radians
    roadangle(float): road orientation in radians
    network(igraph.Graph): network obj
    ax(mpl.Axis): axis to plot

    """
    info(inspect.stack()[0][3] + '()')

    pass

##########################################################
def plot_street_view(point, viewangle, roadangle, network, ax):
    """Plot the view cone along with the network

    Args:
    point(np.array): array of shape (2,)
    viewangle(float): view orientation in radians
    roadangle(float): road orientation in radians
    network(igraph.Graph): network obj
    ax(mpl.Axis): axis to plot

    """
    info(inspect.stack()[0][3] + '()')

    pass

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    # GSV params: heading, pitch (up-down), fov

    cityname = 'sao paulo'
    viewangle = np.pi / 4

    # given alpha=heading
    osmfile = get_osm_file(cityname)
    network = parse_osm_network(osmfile)

    points = [[0,0]]
    points = np.array(points)

    for point in points:
        fig, ax = plt.subplots()
        roadid = get_nearest_road(point, network)
        roadangle = get_road_angle(roadid, network)

        plot_vectors(viewangle, roadangle, ax)
        plot_top_view(point, viewangle, roadangle, network, ax)
        plot_street_view(point, viewangle, roadangle, network, ax)
        plt.close()

##########################################################
if __name__ == "__main__":
    main()

