#!/usr/bin/env python3
"""Find the view based from a location and a view bearing wrt the true north
"""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
from itertools import product
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from datetime import datetime
import igraph
import random
from scipy.spatial.distance import cdist

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

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
        # print(radius)
        g = igraph.Graph.GRG(n, radius=radius)

    plot_graph(g, ax)
    return g

#############################################################
def plot_graph(g, ax=None):
    info(inspect.stack()[0][3] + '()')
    if not ax: return
    xs = g.vs['x']
    ys = g.vs['y']
    ax.scatter(xs, ys)

    segments = np.zeros((g.ecount(), 2, 2), dtype=float)
    for i, e in enumerate(g.es()):
        segments[i, 0, :] = [xs[e.source], ys[e.source]]
        segments[i, 1, :] = [xs[e.target], ys[e.target]]
    col = matplotlib.collections.LineCollection(segments)
    ax.add_collection(col)
    ax.grid()

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

    Returns:
    int: from list of network
    """
    info(inspect.stack()[0][3] + '()')
    dists = np.zeros(network.ecount(), dtype=float)
    refpoints = np.zeros((network.ecount(), 2), dtype=float)

    for i, e in enumerate(network.es):
        p0 = np.array([network.vs[e.source]['x'], network.vs[e.source]['y']])
        p1 = np.array([network.vs[e.target]['x'], network.vs[e.target]['y']])
        # print(p0, p1)
        # breakpoint()

        versor = (p1 - p0) / np.linalg.norm(p1 - p0) # versor

        projpoint = get_point_projection(point, p0, versor)
        inside = is_inside_segment(projpoint, versor, p0, p1, eps=0.001)
        # candidates = [p0, p1]
        if inside:
            candidates = [projpoint]
        else:
            candidates = [p0, p1]
        print(p0, p1, projpoint, inside)
        localdists = cdist([point], candidates)[0]
        nearestid = np.argmin(localdists)
        dists[i] = localdists[nearestid]
        refpoints[i] = candidates[nearestid]

    edgeid = np.argmin(dists)
    refpoint = refpoints[edgeid]
    dist = dists[edgeid]

    if ax:
        c = [np.random.rand(3,)]
        ax.scatter(point[0], point[1], c=c, s=20)
        ax.scatter(refpoint[0], refpoint[1], c=c, s=15)
    return edgeid, refpoint, dist

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
    viewangle = np.pi / 4
    nvertices = 50

    # given alpha=heading
    osmfile = get_osm_file(cityname)
    # network = parse_osm_network(osmfile)

    nrows = 1;  ncols = 1
    figscale = 5
    fig, ax = plt.subplots(nrows, ncols,
            figsize=(ncols*figscale, nrows*figscale))

    network = create_dummy_graph(nvertices, 'grg', ax)
    # points = [[1.0, .5]]
    points = np.random.rand(5, 2) + 0.05

    points = np.array(points)

    # tree = build_spatial_index(network)
    for point in points:
        roadid, refpoint, dist = get_nearest_road(point, network, ax)
        print(point, roadid, refpoint, dist)
        roadangle = get_road_angle(roadid, network)

        plot_vectors(viewangle, roadangle, ax)
        plot_top_view(point, viewangle, roadangle, network, ax)
        plot_street_view(point, viewangle, roadangle, network, ax)
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


