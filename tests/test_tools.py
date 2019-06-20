import sys
import numpy as np

from pytest import raises

sys.path.insert(0, '..')

from spectra.tools import index_of_x, integrate, read_csv, read_csvs, y_at_x, cull


def setup():
    pass


def teardown():
    pass


def test_read_csv(tmpdir):
    p = tmpdir.mkdir("sub").join("test.csv")
    data = [
        ['A', 'B', 'C', 'D'],
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]

    data_str = '\n'.join(','.join(map(str, row)) for row in data)
    p.write(data_str)
    assert p.read() == data_str
    assert len(tmpdir.listdir()) == 1

    csv = read_csv(p)
    assert csv[0] == data[0]
    assert np.all(csv[1] == [[1,5]]*3)
    assert np.all(csv[2] == [[2, 6], [3, 7], [4, 8]])


def test_read_csvs(tmpdir):
    p1 = tmpdir.mkdir("sub").join("test1.csv")
    p2 = tmpdir.join("test2.csv")
    data = [
        ['A', 'B', 'C', 'D'],
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]

    data_str = '\n'.join(','.join(map(str, row)) for row in data)
    p1.write(data_str)
    p2.write(data_str)
    assert len(tmpdir.listdir()) == 2

    csv = read_csvs([p1, p2])
    assert csv[0] == data[0][1:]*2
    assert np.all(csv[1] == [[1,5]]*6)
    assert np.all(csv[2] == [[2, 6], [3, 7], [4, 8]]*2)


def test_index_of_x():
    with raises(IndexError):
        index_of_x(0, [])
    with raises(IndexError):
        index_of_x(0, [1, 2, 3])

    assert 0 == index_of_x(1, [1, 2, 3])
    assert 2 == index_of_x(1, [3, 2, 1])


def test_y_at_x():
    xs = [1, 2, 3]
    ys = [4, 5, 6]
    with raises(IndexError):
        y_at_x(0, [], [])

    with raises(ValueError):
        y_at_x(0, xs, [])

    with raises(IndexError):
        y_at_x(0, xs, ys)

    assert 4 == y_at_x(1,   xs, ys)
    assert 5 == y_at_x(2,   xs, ys)
    assert 6 == y_at_x(2.5, xs, ys)
    assert 4 == y_at_x(3,   xs[::-1], ys)
    assert 5 == y_at_x(2,   xs[::-1], ys)
    assert 4 == y_at_x(2.5, xs[::-1], ys)

    assert y_at_x(2.1, xs, ys) == y_at_x(2.1, xs[::-1], ys[::-1])


def test_integrate():
    assert 0 == integrate([], [])

    with raises(ValueError):
        integrate([1, 2, 3], [])

    assert 10 == integrate([1, 2, 3], [4, 5, 6])

    with raises(IndexError):
        integrate([1, 2, 3], [4, 5, 6], [0, 1])
    with raises(IndexError):
        integrate([1, 2, 3], [4, 5, 6], [1, 9])

    assert 4.5 == integrate([1, 2, 3], [4, 5, 6], [1, 1.1])


def test_cull():
    assert list(cull(range(10), 3)) == [0, 5, 9]
