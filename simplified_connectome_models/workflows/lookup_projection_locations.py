# Description:   Extracting and processing projection locations from a circuit
# Author:        M. Reimann, C. Pokorny
# Date:          08-10/2021

import os
import numpy as np
import pandas as pd
import voxcell
import sklearn.cluster
import matplotlib.pyplot as plt

from voxcell import VoxcellError

virtual_fibers_fn = "virtual-fibers.csv"
virtual_fibers_xyz = ["x", "y", "z"]
cells_xyz = ["x", "y", "z"]
virtual_fibers_uvw = ["u", "v", "w"]
virtual_fibers_mask = "apron"
virtual_fibers_gids = "sgid"


def projection_locations_3d(projection):
    """
    :param projection: Path to a sonata file of a projection of a circuit model
    :return:
    The gids, the 3d locations and directions associated with the projection fibers
    """
    vfib_file = os.path.join(os.path.split(os.path.abspath(projection))[0], virtual_fibers_fn)
    if not os.path.isfile(vfib_file):
        raise RuntimeError("Cannot find virtual fiber info for the selected projection!")
    vfib = pd.read_csv(vfib_file)

    if virtual_fibers_mask is not None:
        vfib = vfib[~vfib[virtual_fibers_mask]]

    return vfib[virtual_fibers_gids].values, vfib[virtual_fibers_xyz].values, vfib[virtual_fibers_uvw].values


def neuron_locations_3d(circ, target_name):
    """
    :param circ: bluepy.Circuit
    :param target_name: Name of a target that must exist in the Circuit
    :return:
    The gids, 3d locations and directions associated with the neurons of the target.
    At the moment, the directions are simply None.
    """
    pos = circ.cells.positions(group=target_name)
    return pos.index.values, pos[cells_xyz].values, None


def apply_flatmap(xyz, uvw, flatmap_fn, max_translation=2000):
    """
    :param xyz: numpy.array, N x 3: coordinates in 3d space
    :param uvw: numpy.array, N x 3: directions in 3d space. Optional, can be None.
    :param flatmap_fn: str: path to a flatmap in .nrrd format
    :param max_translation: float.
    :return:
    Flat locations of the xyz coordinates in the flatmap. If locations are outside the valid region of
    the flatmap and uvw is provided (i.e. not None), then the invalid locations are gradually translated along the
    directions gived by uvw until they hit the valid volume. max_translation defines the maximum amplitude of that
    translation. Locations that never hit the valid volume will return a flat location of (-1, -1).
    """
    fm = voxcell.VoxelData.load_nrrd(flatmap_fn)
    solution = fm.lookup(xyz)

    if uvw is not None:
        fac = 0
        step = fm.voxel_dimensions[0] / 4
        tl_factors = np.zeros((len(uvw), 1))
        solution = fm.lookup(xyz)
        while np.any(solution < 0) and fac < max_translation:
            try:
                fac += step
                to_update = np.any(solution < 0, axis=1)
                tl_factors[to_update, 0] = fac
                solution[to_update, :] = fm.lookup(xyz[to_update, :] + tl_factors[to_update, :] * uvw[to_update, :])
            except VoxcellError:
                break
    return solution


def mask_results_bb(results, circ, mask_name, flatmap=None):
    """
    :param results: The output of "get_projection_locations"
    :param circ: bluepy.Circuit
    :param mask_name: str: Name of a cell target of projection that serves as a mask
    :param flatmap: str: Path to a flatmap in .nrrd format
    :return:
    The "results" are masked such that only the parts within the bounding box of "mask_name" are returned.
    If a "flatmap" is provided, the bounding box and masking is done in the 2d flat space. Otherwise in 3d space.
    """
    res_gids, res2d, res3d, resdir = results
    _, mask2d, mask3d, _ = get_projection_locations(circ, mask_name, flatmap=flatmap)
    if flatmap is None:
        valid = (res3d >= mask3d.min(axis=0, keepdims=True)) & (res3d <= mask3d.max(axis=0, keepdims=True))
    else:
        mask2d = mask2d[np.all(mask2d >= 0, axis=1)]
        valid = (res2d >= mask2d.min(axis=0, keepdims=True)) & (res2d <= mask2d.max(axis=0, keepdims=True))
    valid = np.all(valid, axis=1)

    res_gids = res_gids[valid]
    if res2d is not None:
        res2d = res2d[valid]
    if res3d is not None:
        res3d = res3d[valid]
    if resdir is not None:
        resdir = resdir[valid]

    return res_gids, res2d, res3d, resdir


def mask_results_dist(results, circ, mask_name, max_dist=None, dist_factor=None, flatmap=None):
    """
    :param results: The output of "get_projection_locations"
    :param circ: bluepy.Circuit
    :param mask_name: str: Name of a cell target of projection that serves as a mask
    :param max_dist: float: (Optional) Maximal distance from the "mask" location that is considered valid.
    If not provided, a value will be estimated using "dist_factor"
    :param dist_factor: float: (Optional, default: 2.0) If "max_dist" is None, this will be used to conduct an estimate.
    :param flatmap: str: Path to a flatmap in .nrrd format
    :return:
    The "results" are masked such that only the parts within "max_dist" of locations associated with "mask_name"
    are returned.
    If a "flatmap" is provided, the distances are calculated in the 2d flat space. Otherwise in 3d space.
    """
    from scipy.spatial import KDTree

    res_gids, res2d, res3d, resdir = results
    _, mask2d, mask3d, _ = get_projection_locations(circ, mask_name, flatmap=flatmap)

    if flatmap is None:
        use_res = res3d
        use_mask = mask3d
    else:
        use_res = res2d
        use_mask = mask2d

    if dist_factor is None:
        dist_factor = 2.0 # (default)

    t_res = KDTree(use_res)
    t_mask = KDTree(use_mask)
    if max_dist is None:
        dists, _ = t_res.query(use_res, 2)
        max_dist = dist_factor * dists[:, 1].mean()
    actives = t_mask.query_ball_tree(t_res, max_dist)
    actives = np.unique(np.hstack(actives).astype(int))

    res_gids = res_gids[actives]
    if res2d is not None:
        res2d = res2d[actives]
    if res3d is not None:
        res3d = res3d[actives]
    if resdir is not None:
        resdir = resdir[actives]

    return res_gids, res2d, res3d, resdir


def get_projection_locations(circ, projection_name, mask=None, mask_type=None, flatmap=None):
    """
    :param circ: bluepy.Circuit
    :param projection_name: str: Name of a projection or a cell target in the Circuitg
    :param mask: str: Name of a cell target of projection to serve as a mask
    :param mask_type: str: Type of mask to apply: "bbox" (default) or "dist"
    :param flatmap: str: Path to a flatmap in .nrrd format
    :return:
    The gids, 2d flat locations, 3d locations and 3d directions associated with the projection or cell target.
    If "flatmap" is provided, it is used to determine 2d flat locations, otherwise the 2d flat locations are None.
    If "projection_name" is the name of a cell target, then the 3d directions are None.
    If "mask" is provided, then only results defined by the locations and type of the mask are returned.
        If "mask" and "flatmap" are provided, then this masking is happening in the flat 2d space.
        If only "mask" is provided, then this masking is happening in 3d space. This is not recommended.
    """
    circ_proj = circ.config["projections"]

    if projection_name in circ_proj:
        gids, pos3d, dir3d = projection_locations_3d(circ_proj[projection_name])
    else:
        gids, pos3d, dir3d = neuron_locations_3d(circ, projection_name)

    if flatmap is not None:
        pos2d = apply_flatmap(pos3d, dir3d, flatmap)
    else:
        pos2d = None

    if mask_type is None:
        mask_type = 'bbox'
    
    if mask is not None:
        if mask_type == 'bbox':
            gids, pos2d, pos3d, dir3d = mask_results_bb((gids, pos2d, pos3d, dir3d),
                                                     circ, mask, flatmap=flatmap)
        elif mask_type.find('dist') == 0:
            mask_spec = mask_type.replace('dist', '')
            if len(mask_spec) > 0: # Extract distance factor, e.g. mask_type = 'dist2.0'
                dist_factor = float(mask_spec)
            else:
                dist_factor = None
            gids, pos2d, pos3d, dir3d = mask_results_dist((gids, pos2d, pos3d, dir3d),
                                    circ, mask, dist_factor=dist_factor, flatmap=flatmap)
        else:
            raise RuntimeError(f"Mask type {mask_type} unknown!")

    return gids, pos2d, pos3d, dir3d


def cluster_by_locations(gids, pos, n_clusters=None, n_per_cluster=None, cluster_seed=0):
    """
    :param gids: numpy.array, N x 1: List projection gids
    :param pos: numpy.array, N x D: D-dim locations of the projection fibers (2d or 3d)
    :param n_clusters: int: Number if clusters. Optional, can be None if n_per_cluster is given.
    :param n_per_cluster: int: Number of fibers per cluster. Optional, can be None if n_clusters is given.
    :param cluster_seed: int: Random seed of k-means clustering. Optional, default: 0.
    :return:
    The list of lists of gids belonging to a cluster of nearby fibers, D-dim cluster centroids, and cluster
    indices associated with the clusters of projection fibers.
    Either "n_clusters" or "n_per_cluster" needs to be specified to determine the resulting number of clusters.
    "pos" can be either 2d or 3d positions.
    """
    if n_clusters is None:
        if n_per_cluster is None:
            raise RuntimeError("Need to specify number of clusters or mean number of fibers per cluster")
        n_clusters = int(round(float(len(gids)) / n_per_cluster))

    if n_clusters == len(gids): # No clustering (i.e., 1 fiber = 1 cluster)
        gids_list = [[i] for i in range(len(gids))]
        cluster_pos = pos
        cluster_idx = np.arange(len(gids))
    else:
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=cluster_seed).fit(pos)
        gids_list = [gids[kmeans.labels_ == i] for i in range(n_clusters)]
        cluster_pos = kmeans.cluster_centers_
        cluster_idx = kmeans.labels_

    return gids_list, cluster_pos, cluster_idx


def plot_clusters_of_fibers(grp_idx, grp_pos, pos2d, pos3d, pos2d_all, pos3d_all, save_path=None):
    """
    Plots spatial clusters (groups) of nearby fibers
    """
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    num_groups = grp_pos.shape[0]
    grp_colors = plt.cm.jet(np.linspace(0, 1, num_groups))
    np.random.seed(0) # Just for color permutation
    grp_colors = grp_colors[np.random.permutation(num_groups), :]

    if pos2d is not None:
        plt.figure(figsize=(5, 5))
        plt.plot(pos2d_all[:, 0], pos2d_all[:, 1], '.', color='grey', markersize=1)
        for i in range(num_groups):
            plt.plot(pos2d[grp_idx == i, 0], pos2d[grp_idx == i, 1], '.', color=grp_colors[i, :], markersize=1)
        if grp_pos.shape[1] == 2:
            for i in range(num_groups):
                plt.plot(grp_pos[i, 0], grp_pos[i, 1], 'x', color=grp_colors[i, :])
        plt.axis('image')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Clusters of fibers')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'clusters_2d.png'), dpi=300)

    if pos3d is not None:
        plt.figure(figsize=(5, 5))
        plt.subplot(1, 1, 1, projection='3d')
        plt.plot(pos3d_all[:, 0], pos3d_all[:, 1], pos3d_all[:, 2], '.', color='grey', markersize=1)
        for i in range(num_groups):
            plt.plot(pos3d[grp_idx == i, 0], pos3d[grp_idx == i, 1], pos3d[grp_idx == i, 2], '.', color=grp_colors[i, :], markersize=1)
        if grp_pos.shape[1] == 3:
            for i in range(num_groups):
                plt.plot(grp_pos[i, 0], grp_pos[i, 1], grp_pos[i, 2], 'x', color=grp_colors[i, :])
        plt.gca().set_xlabel('x')
        plt.gca().set_ylabel('y')
        plt.gca().set_zlabel('z')
        plt.title('Clusters of fibers')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'clusters_3d.png'), dpi=300)


def plot_cluster_size_distribution(grp_idx, save_path=None):
    """
    Plot distribution of cluster (group) sizes
    """
    num_clusters = np.max(grp_idx) + 1

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    cl_hist = plt.hist(grp_idx, bins=np.arange(-0.5, num_clusters + 0.5, 1))[0]
    plt.text(0.99 * np.max(plt.xlim()), 0.99 * np.max(plt.ylim()), f'MIN: {np.min(cl_hist)}\nMAX: {np.max(cl_hist)}\nMEAN: {np.mean(cl_hist):.1f}\nSTD: {np.std(cl_hist):.1f}\nCOV: {np.std(cl_hist) / np.mean(cl_hist):.1f}', ha='right', va='top')
    plt.xlim(plt.xlim()) # Freeze axis limits
    plt.plot(plt.xlim(), np.full(2, np.mean(cl_hist)), '--', color='tab:red')
    plt.xlabel('Cluster idx')
    plt.ylabel('Cluster size')
    plt.title(f'Cluster sizes (N={num_clusters})')

    plt.subplot(1, 2, 2)
    plt.hist(cl_hist, bins=np.arange(-0.5, np.max(cl_hist) + 1.5))
    plt.ylim(plt.ylim()) # Freeze axis limits
    plt.plot(np.full(2, np.mean(cl_hist)), plt.ylim(), '--', color='tab:red')
    plt.xlabel('Cluster size')
    plt.ylabel('Count')
    plt.title('Cluster size distribution')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'cluster_sizes.png'), dpi=300)
