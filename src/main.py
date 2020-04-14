#!/usr/bin/env python3
"""Find the view based from a location and a view bearing wrt the true north
"""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import math
import numpy as np
from itertools import product
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from multiprocessing import Pool
from datetime import datetime
import igraph
import random
from scipy.spatial.distance import cdist

palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']


#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

##########################################################
def hex2rgb(hexcolours, alpha=None):
    rgbcolours = np.zeros((len(hexcolours), 3), dtype=int)
    for i, h in enumerate(hexcolours):
        rgbcolours[i, :] = np.array([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])

    if alpha != None:
        aux = np.zeros((len(hexcolours), 4), dtype=float)
        aux[:, :3] = rgbcolours / 255.0
        aux[:, -1] = .7 # alpha
        rgbcolours = aux

    return rgbcolours

##########################################################
def versor2angle(versor, unit='radian'):
    """Convert versor to angle

    Args:
    versor(np.ndarray): 2d versor
    unit(str): degree or radian

    Returns:
    float: angle
    """
    info(inspect.stack()[0][3] + '()')
    zeroversor = np.array([1, 0])
    anglerad = np.arccos(np.dot(versor, zeroversor))
    if unit == 'radian': return anglerad
    else: return math.degrees(anglerad)

#############################################################
def get_4connected_neighbours_2d(i, j, n, thoroidal=False):
    """Get 4-connected neighbours. It does not check if there are repeated entries (2x2 or 1x1)

    Args:
    i(int): row of the matrix
    j(int): column of the matrix
    n(int): side of the square matrix

    Returns:
    ndarray 4x2: 4 neighbours indices
    """
    info(inspect.stack()[0][3] + '()')
    inds = []
    if j > 0: # left
        inds.append([i, j-1])
    elif thoroidal:
        inds.append([i, n-1])

    if j < n-1: # right
        inds.append([i, j+1])
    elif thoroidal:
        inds.append([i, 0])

    if i > 0: # top
        inds.append([i-1, j])
    elif thoroidal:
        inds.append([n-1, j])

    if i < n-1: # bottom
        inds.append([i+1, j])
    elif thoroidal:
        inds.append([0, j])

    return np.array(inds)

#############################################################
def generate_lattice(n, thoroidal=False, edgearg=-1):
    """Generate 2d lattice of side n

    Args:
    n(int): side of the lattice
    thoroidal(bool): thoroidal lattice
    s(float): edge size

    Returns:
    ndarray nx2, ndarray nxn: positions and adjacency matrix (triangular)
    """
    info(inspect.stack()[0][3] + '()')
    n2 = n*n
    pos = np.ndarray((n2, 2), dtype=float)
    adj = np.zeros((n2, n2), dtype=int)
    s = 1 / (n-1) if edgearg < 0 else edgearg

    k = 0
    for j in range(n):
        for i in range(n): # Set positions
            pos[k] = [i*s, j*s]
            k += 1

    for i in range(n): # Set connectivity
        for j in range(n):
            neighs2d = get_4connected_neighbours_2d(i, j, n, thoroidal)
            neighids = np.ravel_multi_index((neighs2d[:, 0], neighs2d[:, 1]), (n, n))
            curidx = np.ravel_multi_index((i, j), (n, n))

            for neigh in neighids:
                adj[curidx, neigh] = 1
    return pos, adj

#############################################################
def create_lattice(n, ax=None):
    info(inspect.stack()[0][3] + '()')
    m = int(np.sqrt(n))
    pos, adj = generate_lattice(m, False)
    g = igraph.Graph.Adjacency(adj.tolist(), mode=igraph.ADJ_UNDIRECTED)
    g.vs['x'] = pos[:, 0]
    g.vs['y'] = pos[:, 1]
    return g

#############################################################
def create_dummy_graph(n, top, ax=None):
    info(inspect.stack()[0][3] + '()')

    if top == 'lattice':
        g = create_lattice(n, ax)
    elif top == 'grg':
        radius = 10/n
        g = igraph.Graph.GRG(n, radius=radius)

    if ax: plot_graph(g, ax)
    return g

#############################################################
def plot_graph(g, ax=None):
    info(inspect.stack()[0][3] + '()')
    if not ax: return
    xs = g.vs['x']
    ys = g.vs['y']
    # ax.scatter(xs, ys, s=10)

    segments = np.zeros((g.ecount(), 2, 2), dtype=float)
    for i, e in enumerate(g.es()):
        segments[i, 0, :] = [xs[e.source], ys[e.source]]
        segments[i, 1, :] = [xs[e.target], ys[e.target]]
    col = matplotlib.collections.LineCollection(segments, zorder=0)
    ax.add_collection(col)
    ax.grid()
    ax.set_axisbelow(True)

#############################################################
def build_spatial_index(network):
    info(inspect.stack()[0][3] + '()')
    pass

#############################################################
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
def is_inside_segment(point, versor, p0, p1, eps=0.001):
    """Assuming p=p0+versor*t"""
    # if  np.abs(point[0] - 0.92512639) < 0.01:
        # breakpoint()

    i = 1 if versor[0] == 0 else 0 # in case it is a vertical line

    trange = sorted([0, (p1[i] - p0[i]) / versor[i]]) # p = p0 + p1*t
    t = (point[i] - p0[i]) / versor[i]
    if (t < trange[0] - eps) or (t > trange[1] + eps): return False
    else: return True

##########################################################
def get_nearest_road(point, network, ax=None):
    """Get id of the nearest street in the network. If there are two streets with same distance, get the first I find

    Args:
    point(np.array): array of shape (2,)
    network(igraph.Graph): network in osm format
    ax(matplotlib.Axis): axis to plot

    Returns:
    int: from list of network
    """
    info(inspect.stack()[0][3] + '()')
    dists = np.zeros(network.ecount(), dtype=float)
    refpoints = np.zeros((network.ecount(), 2), dtype=float)
    arevertices = np.zeros((network.ecount()), dtype=int)

    for i, e in enumerate(network.es):
        p0 = np.array([network.vs[e.source]['x'], network.vs[e.source]['y']])
        p1 = np.array([network.vs[e.target]['x'], network.vs[e.target]['y']])

        versor = (p1 - p0) / np.linalg.norm(p1 - p0) # versor

        projpoint = get_point_projection(point, p0, versor)
        inside = is_inside_segment(projpoint, versor, p0, p1, eps=0.001)
        if inside:
            candidates = [projpoint]
        else:
            candidates = [p0, p1]
            arevertices[i] = True
        localdists = cdist([point], candidates)[0]
        nearestid = np.argmin(localdists)
        dists[i] = localdists[nearestid]
        refpoints[i] = candidates[nearestid]

    edgeid = np.argmin(dists)
    refpoint = refpoints[edgeid]
    isvertex = bool(arevertices[edgeid])
    angle = refpoint - point
    angle /= np.linalg.norm(angle)
    dist = dists[edgeid]

    if ax:
        eps = 0.05
        # c = [np.random.rand(3,)]
        ax.scatter(point[0], point[1], c=palettehex[2], s=10)
        # ax.scatter(refpoint[0], refpoint[1], c=c, s=8)
        # mid = (point + refpoint) / 2
        # anglestr = '{:.1f}, {:.1f}'.format(angle[0], angle[1])
        # ax.text(mid[0]-eps, mid[1], anglestr, fontsize=5)
        # ax.text(mid[0]+eps, mid[1], str(isvertex), fontsize=5)
    return edgeid, refpoint, isvertex, dist

##########################################################
def get_viewer_point(snappedpt, viewversor, step, ax=None):
    """Get the  viewer position in the street

    Args:
    snappedpt(np.ndarray): 2d coordinates
    viewversor(np.ndarray): 2d versor of the view
    ax(matplotlib.Axis): axis to plot
    """
    return snappetpt - viewversor * step

##########################################################
def plot_view_cone(snappedpt, step, rad, coneangle, viewversor, network, ax=None):
    """Plot the view cone from the @refpoint in the @network

    Args:
    refpoint(np.ndarray(2)): point 2d coordinates
    step(float): distance to the point
    viewversor(np.ndarray): view direction
    network(igraph.Graph): network
    ax(matplotlib.Axis): axis to plot
    """

    info(inspect.stack()[0][3] + '()')
    viewerpt = snappedpt - viewversor * step
    ax.scatter(viewerpt[0], viewerpt[1], c='k', s=5)

    c = np.ones(4, dtype=float) * 0.5
    viewangle = versor2angle(viewversor, unit='degrees')
    patches = [Wedge(viewerpt, rad, viewangle-coneangle/2,
        viewangle+coneangle/2, color=c, linewidth=0)]
    p = PatchCollection(patches, match_original=True)
    ax.add_collection(p)

##########################################################
def get_point_projection(point, p0, versor):
    """Point projection into line

    Args:
    point, p0, versor(str): output dir

    Returns:
    ret
    """
    info(inspect.stack()[0][3] + '()')
    versororth = np.array([-versor[1], versor[0]])
    a = np.concatenate([
        versororth.reshape(len(versororth), 1),
        -versor.reshape(len(versor), 1),
        ], axis=1)
    b = p0 - point
    x = np.linalg.solve(a, b)
    return point + x[0] * versororth

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

    for i, e in enumerate(network.es):
        p0 = np.array([network.vs[e.source]['x'], network.vs[e.source]['y']])
        p1 = np.array([network.vs[e.target]['x'], network.vs[e.target]['y']])
        versor = (p1 - p0) / np.linalg.norm(p1 - p0) # versor
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

#############################################################
def run_experiment(params_):
    info(inspect.stack()[0][3] + '()')
    cityname = 'sao paulo'
    # viewangle = np.pi / 4
    viewversor = np.array([0.0, 1.0]); viewversor /= np.linalg.norm(viewversor)
    nvertices = 100
    nviewers = 10
    step = 0.02
    conerad = 0.05
    coneangle = 90

    # given alpha=heading
    osmfile = get_osm_file(cityname)
    # network = parse_osm_network(osmfile)

    nrows = 1;  ncols = 1
    figscale = 5
    fig, ax = plt.subplots(nrows, ncols,
            figsize=(ncols*figscale, nrows*figscale))

    network = create_dummy_graph(nvertices, 'grg', ax)
    # points = np.array([[1.0, .5]])
    points = np.random.rand(nviewers, 2) + 0.05

    # tree = build_spatial_index(network)
    for point in points:
        edgeid, snappedpt, isvertex, dist = get_nearest_road(point, network, ax)
        plot_view_cone(snappedpt, step, conerad, coneangle, viewversor, network, ax)
        # print(point, roadid, refpoint, dist)
        # roadangle = get_road_angle(roadid, network)

        # plot_vectors(viewangle, roadangle, ax)
        # plot_top_view(point, viewangle, roadangle, network, ax)
        # plot_street_view(point, viewangle, roadangle, network, ax)
        # plt.close()

    plt.savefig(pjoin(params_['outdir'], 'graph.pdf'))

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nprocs', type=int, default=1, help='nprocs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    aux = list(product([args.outdir]))

    np.random.seed(args.seed)
    random.seed(args.seed)
    params = []
    for i, row in enumerate(aux):
        params.append(dict(outdir = row[0],))

    if args.nprocs == 1:
        info('Running serially (nprocs:{})'.format(args.nprocs))
        respaths = [run_experiment(p) for p in params]
    else:
        info('Running in parallel (nprocs:{})'.format(args.nprocs))
        pool = Pool(args.nprocs)
        respaths = pool.map(run_experiment, params)

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()


