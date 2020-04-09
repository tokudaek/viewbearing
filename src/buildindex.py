import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
import matplotlib.pyplot as plt


def snap_to_line(points, lines, tolerance=100, sindex=None):
    """
    Attempt to snap a line to the nearest line, within tolerance distance.
    Lines must be in a planar (not geographic) projection and points
    must be in the same projection.
    Parameters
    ----------
    points : GeoPandas.DataFrame
        points to snap
    lines : GeoPandas.DataFrame
        lines to snap against
    tolerance : int, optional (default: 100)
        maximum distance between line and point that can still be snapped
    Returns
    -------
    geopandas.GeoDataFrame
        output data frame containing:
        * all columns from points except geometry
        * geometry: snapped geometry
        * snap_dist: distance between original point and snapped location
        * nearby: number of nearby lines within tolerance
        * any columns joined from lines
    """

    # get list of columns to copy from flowlines
    line_columns = lines.columns[lines.columns != "geometry"].to_list()

    # generate spatial index if it is missing
    if sindex is None:
        sindex = lines.sindex
        # Note: the spatial index is ALWAYS based on the integer index of the
        # geometries and NOT their index

    # generate a window around each point
    window = points.bounds + [-tolerance, -tolerance, tolerance, tolerance]
    # get a list of the line ordinal line indexes (integer index, not actual index) for each window
    hits = window.apply(lambda row: list(sindex.intersection(row)), axis=1)

    # transpose from a list of hits to one entry per hit
    # this implicitly drops any that did not get hits
    tmp = pd.DataFrame(
        {
            # index of points table
            "pt_idx": np.repeat(hits.index, hits.apply(len)),
            # ordinal position of line - access via iloc
            "line_i": np.concatenate(hits.values),
        }
    )

    # reset the index on lines to get ordinal position, and join to lines and points
    tmp = tmp.join(lines.reset_index(drop=True), on="line_i").join(
        points.geometry.rename("point"), on="pt_idx"
    )
    tmp = gpd.GeoDataFrame(tmp, geometry="geometry", crs=points.crs)
    tmp["snap_dist"] = tmp.geometry.distance(gpd.GeoSeries(tmp.point))

    # drop any that are beyond tolerance and sort by distance
    tmp = tmp.loc[tmp.snap_dist <= tolerance].sort_values(by=["pt_idx", "snap_dist"])

    # find the nearest line for every point, and count number of lines that are within tolerance
    by_pt = tmp.groupby("pt_idx")
    closest = gpd.GeoDataFrame(
        by_pt.first().join(by_pt.size().rename("nearby")), geometry="geometry"
    )

    # now snap to the line
    # project() calculates the distance on the line closest to the point
    # interpolate() generates the point actually on the line at that point
    snapped_pt = closest.interpolate(
        closest.geometry.project(gpd.GeoSeries(closest.point))
    )
    snapped = gpd.GeoDataFrame(
        closest[line_columns + ["snap_dist", "nearby"]], geometry=snapped_pt
    )

    # NOTE: this drops any points that didn't get snapped
    return points.drop(columns=["geometry"]).join(snapped).dropna(subset=["geometry"])


points = gpd.GeoDataFrame([Point(2, 2), Point(9, 8)], columns=['geometry'])
lines = gpd.GeoDataFrame([
    LineString([[0, 1], [1, 0]]),
    LineString([[0, 10], [10, 15]]),
    ], columns=['geometry'])
ret = snap_to_line(points, lines, tolerance=100, sindex=None)
print(ret)
