import csv
import numpy as np


def read_csv(inp, header=True):
    """
    Read a csv file.
    :param inp: input file
    :return:
        :titles: titles of the columns
        :xs: x-values (1- or 2-dim np.array)
        :ys: y-values (1- or 2-dim np.array, matches x)
    """
    titles = None
    with open(inp) as f:
        reader = csv.reader(f)
        if header:
            titles = next(reader)

        xs, ys = [], []
        for x, *y in reader:
            xs.append(float(x))
            ys.append([float(y_val) for y_val in y])
    ys = np.array(ys).T
    xs = (np.ones(ys.shape) * xs)

    if titles is None:
        titles = ['']*len(xs)

    return titles, xs, ys


def read_csvs(inps, header=True):
    """
    Read an iterable of CSVs (or only one if a string)
    :param inps: input file(s) to read
    :return: titles, xs, ys
    """
    titles = []
    if isinstance(inps, str):
        ts, xs_list, ys_list = read_csv(inps)
        titles = ts[1:]
    else:
        xs_list, ys_list = [], []
        for inp in inps:
            ts, xs, ys = read_csv(inp)
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


def y_at_x(x_point, xs, ys):
    """
    Determine the y-value at a specified x. If in between xs, choose the first
    past it. Assumes xs are ordered.

    :param x_point: x-value for which the y-value is desired
    :param xs: x-values
    :param ys: y-values
    """
    if len(xs) != len(ys):
        raise ValueError(f'xs and ys must be of the same length, got: {len(xs)} and {len(ys)}')

    if xs[0] > xs[-1]:
        xs = xs[::-1]

    if x_point < xs[0] or x_point > xs[-1]:
        raise IndexError(f'x_point not in xs, x_point:{x_point}, xs: ({xs[0]}→{xs[-1]})')

    for x, y in zip(xs, ys):
        if x >= x_point:
            return y


def integrate(xs, ys, x_range=None):
    """
    Integrate a set of ys on the xs.
    :param xs, ys: x- and y-values
    :param x_range: range of x_values to integrate over
    :return: integration
    """
    assert len(xs) == len(ys)

    if x_range is not None:
        begin, end = x_range
        start, finish = None, None
        if begin < xs[0]:
            raise KeyError(f'x_range starts before first value in xs ({begin} > {xs[0]}')

        for i, x in enumerate(xs):
            if x >= begin:
                start = i
                break
        for i, x in enumerate(xs[start:], start=start):
            if x >= end:
                finish = i
                break

        if end is None:
            raise KeyError('Invalid end')

        xs = xs[start:finish]
        ys = ys[start:finish]

    return np.trapz(ys, xs)
