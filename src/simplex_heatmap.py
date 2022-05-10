import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker, cm
import numpy as np

# adapted from http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_AREA = 0.5 * 1 * 0.75**0.5
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])

# For each corner of the triangle, the pair of other corners
_pairs = [_corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points


def tri_area(xy, pair):
    return 0.5 * np.linalg.norm(np.cross(*(pair - xy)))


def xy2bc(xy, tol=1e-7):
    '''Converts 2D Cartesian coordinates to barycentric.

    Arguments:

        `xy`: A length-2 sequence containing the x and y value.
    '''
    coords = np.array([tri_area(xy, p) for p in _pairs]) / _AREA
    return np.clip(coords, tol, 1.0 - tol)


def draw_heatmap(xy_fun, ref_point=np.array([1/3, 1/3, 1/3]), take_sqrt=True, border=False, nlevels=50, subdiv=4, ax=None, *args, **kwargs):
    '''Draws filled contour map over an equilateral triangle (2-simplex).

    Arguments:

        `xy_fun`: A bivariate function, e.g. one of the `d2_*` functions in metrics.py but wrapped so that it only takes `x` and `y`, or one of the `k_*` functions in kernels.py.

        `take_sqrt`: A boolean indicating whether to take square root of the return value of `d2_fun` or not.

        `border` (bool): If True, the simplex border is drawn.

        `nlevels` (int): Number of contours to draw.

        `subdiv` (int): Recursion level for the subdivision. Each triangle is divided into 4**subdiv child triangles.

        args: Additional args passed on to `d2_fun`.

        kwargs: Keyword args passed on to `plt.triplot`.
    '''
    ax = ax or plt.gca()
    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    if take_sqrt:
        pvals = [np.sqrt(xy_fun(xy2bc(xy), ref_point, *args))
                 for xy in zip(trimesh.x, trimesh.y)]
    else:
        pvals = [xy_fun(xy2bc(xy), ref_point, *args)
                 for xy in zip(trimesh.x, trimesh.y)]
    # Note (from documentation https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tricontourf.html):
    #   If a colormap is used, the Normalize instance scales the level values to the canonical colormap range [0, 1] for mapping to colors.
    #   If not given, the default linear scaling is used.
    ax.axis('equal')
    ax.tricontourf(trimesh, pvals, nlevels, cmap='jet', **kwargs)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.75**0.5)
    ax.axis('off')
    # plt.tricontourf(trimesh, pvals, nlevels, cmap='jet', **kwargs)
    # plt.axis('equal')
    # plt.xlim(0, 1)
    # plt.ylim(0, 0.75**0.5)
    # plt.axis('off')
    if border is True:
        ax.triplot(_triangle, linewidth=1)
        # plt.triplot(_triangle, linewidth=1)


def draw_points(X, barycentric=True, border=False, fmt='k.', ax=None, **kwargs):
    '''Plots a set of points in the simplex.

    Arguments:

        `X` (ndarray): A 2xN array (if in Cartesian coords) or 3xN array
                       (if in barycentric coords) of points to plot.

        `barycentric` (bool): Indicates if `X` is in barycentric coords.

        `border` (bool): If True, the simplex border is drawn.

        `fmt`: '[marker][line][color]' see Notes in https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html.

        kwargs: Keyword args passed on to `plt.plot`.
    '''
    ax = ax or plt.gca()
    if barycentric is True:
        X = X.dot(_corners)
    if len(X.shape) == 1:  # dimension fix when N == 1
        X = X[np.newaxis, :]
    ax.plot(X[:, 0], X[:, 1], fmt, ms=5, **kwargs)
    ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.75**0.5)
    ax.axis('off')
    if border is True:
        ax.triplot(_triangle, linewidth=1, color='grey')
    return ax
    # plt.plot(X[:, 0], X[:, 1], fmt, ms=5, **kwargs)
    # plt.axis('equal')
    # plt.xlim(0, 1)
    # plt.ylim(0, 0.75**0.5)
    # plt.axis('off')
    # if border is True:
    #     plt.triplot(_triangle, linewidth=1, color='black')
