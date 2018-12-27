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
    with open(inp) as f:
        reader = csv.reader(f)
        titles = None
        if header:
            titles = next(reader)

        xs, ys = [], []
        for x, *y in reader:
            xs.append(float(x))
            ys.append([float(y_val) for y_val in y])
    ys = np.array(ys).T
    xs = (np.ones(ys.shape) * xs)

    if header:
        return titles, xs, ys
    return xs, ys


def read_csvs(inps, header=True):
    """
    Read an iterable of CSVs (or only one if a string)
    :param inps: input file(s) to read
    :return: xs, ys, names
    """
    names = []
    if isinstance(inps, str):
        titles, xs_list, ys_list = read_csv(inps)
        names = titles[1:]
    else:
        xs_list, ys_list = [], []
        for inp in inps:
            titles, xs, ys = read_csv(inp)
            names.extend(titles[1:])
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
    assert len(ys) == len(names)

    return xs, ys, names
