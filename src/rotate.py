#!/usr/bin/env python3
"""rotate image
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
import imageio

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

#############################################################
def run_experiment(params_):
    info(inspect.stack()[0][3] + '()')
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    img = imageio.imread('data/sample.jpg')
    ax[0].imshow(img)

    from scipy.spatial.transform import Rotation
    import cv2
    rot = Rotation.from_rotvec(-0.0045 * np.array([0, 1, 0]))
    # outsize = (1200, 700)
    outsize = (600, 300)
    transformed = cv2.warpPerspective(img, rot.as_matrix(), outsize)
    ax[1].imshow(transformed)

    plt.savefig('/tmp/out.png')

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nprocs', type=int, default=1, help='nprocs')
    parser.add_argument('--outdir', default='/tmp/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    aux = list(product([args.outdir]))

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

if __name__ == "__main__":
    main()


