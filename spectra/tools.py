import csv
import itertools
import numpy as np

from glob import glob


def read_csv(inp, header=True):
    """
    Reads a csv file.

    :param inp: input file
    :param header: inp contains a header
    :return:
        :titles: titles of the columns
        :xs: x-values (1- or 2-dim np.array)
        :ys: y-values (1- or 2-dim np.array, matches x)
    """
    titles = None
    try:
        with open(inp) as f:
            reader = csv.reader(f)
            if header:
                titles = next(reader)

            xs, ys = [], []
            for x, *y in reader:
                xs.append(float(x))
                ys.append([float(y_val) for y_val in y])
    except ValueError as e:
        raise ValueError(f'Error reading value in {inp}.') from e

    ys = np.array(ys).T
    xs = np.array(xs)

    if titles is None:
        titles = [''] * len(xs)

    return titles, xs, ys


def read_csvs(inps, header=True):
    """
    Read an iterable of CSVs (or only one if a string).

    :param inps: input file(s) to read
    :param header: inp contains a header
    :return: titles, xs, ys
    """
    titles = []
    if isinstance(inps, str):
        titles, xs_list, ys_list = read_csv(inps, header)
        titles = titles[1:]
        xs_list = (np.ones(ys_list.shape) * xs_list)
    else:
        xs_list, ys_list = [], []
        for inp in inps:
            ts, xs, ys = read_csv(inp, header)
            xs = (np.ones(ys.shape) * xs)
            titles.extend(ts[1:])
            if ys.shape[1] == 1:
                xs_list.append(xs)
                ys_list.append(ys)
            else:
                for x_vals, y_vals in zip(xs, ys):
                    xs_list.append(x_vals)
                    ys_list.append(y_vals)

    xs = np.array(xs_list)
    ys = np.array(ys_list)

    # Sanity checks
    assert len(xs) == len(ys)
    assert len(ys) == len(titles)

    return titles, xs, ys


def glob_read_csvs(inps, header=True):
    """
    Use glob to find CSVs and then reads them.

    :param inps: a string or list of strings that can be read by glob
    :param header: inp contains a header
    :return: titles, xs, ys, file_names
    """
    if isinstance(inps, str):
        inps = [inps]
    file_names = list(itertools.chain(*(glob(inp) for inp in inps)))
    titles, xs, ys = read_csvs(file_names)

    return titles, np.array(xs), np.array(ys), file_names


def y_at_x(x_point, xs, ys):
    """
    Determine the y-value at a specified x. If in between xs, choose the first
    past it. Assumes xs are ordered.

    :param x_point: x-value for which the y-value is desired
    :param xs: x-values
    :param ys: y-values
    :return: desired y-value
    """
    if len(xs) != len(ys):
        raise ValueError(f'xs and ys must be of the same length, got: {len(xs)} and {len(ys)}')

    return ys[index_of_x(x_point, xs)]


def index_of_x(x_point, xs):
    """
    Determine the index of value(s) in an ordered list. If in between xs,
    choose the first past it (larger). Assumes xs are ordered.

    :param x_point: value(s) to find
    :param xs: list to search in
    :return: index of the nearest x_point
    """
    # If in reverse order
    revd = False
    if xs[0] > xs[-1]:
        xs = xs[::-1]
        revd = True

    try:
        x_iter = iter(x_point)
    except TypeError:
        x_iter = [x_point]

    for x in x_iter:
        if x < xs[0] or x > xs[-1]:
            raise IndexError(f'x_point not in xs, x_point: {x}, xs: ({xs[0]}→{xs[-1]})')

    if revd:
        return len(xs) - np.searchsorted(xs, x_point) - 1
    return np.searchsorted(xs, x_point)


def integrate(xs, ys, x_range=None):
    """
    Integrate a set of ys on the xs.

    Note: if x_range does not fall exactly on values in x, it finds the next largest x value.

    :param xs: x-values
    :param ys: y-values
    :param x_range: range of x_values to integrate over
    :return: integration
    """
    if len(xs) != len(ys):
        raise ValueError(f'xs and ys must be of the same length, got: {len(xs)} and {len(ys)}')

    if x_range is not None:
        begin, end = x_range
        if begin < xs[0]:
            raise IndexError(f'x_range starts before first value in xs ({begin} > {xs[0]}')
        start = index_of_x(begin, xs)
        finish = index_of_x(end, xs)

        xs = xs[start:finish + 1]
        ys = ys[start:finish + 1]

    return np.trapz(ys, xs)


def smooth_curve(ys, box_pts=True):
    """
    Smooth a curve.

    Assumes that the ys are uniformly distributed. Returns output of length
    `max(ys, box_pts)`, boundary effects are visible.

    Note: ys must be > box_pts

    :param ys: points to smooth
    :param box_pts: number of data points to convolve, if True, use 3
    :return: smoothed points
    """
    if box_pts is True:
        box_pts = 3

    box = np.ones(box_pts) / box_pts
    return np.convolve(ys, box, mode='same')


def cull(vals, n):
    """
    Cull `vals` to have `n` "evenly" spaced values.
    If not evenly divisible, spread them out as evenly as possible.

    :var vals: the values to cull
    :var n: number of values to keep
    :yield: culled values
    """
    yield from (vals[i] for i in np.linspace(0.5, len(vals) - 0.5, n, dtype=int))
