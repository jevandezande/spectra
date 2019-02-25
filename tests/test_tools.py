import sys
from pytest import raises
import numpy as np

sys.path.insert(0, '..')
from spectra.tools import read_csv, read_csvs, y_at_x


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


def test_y_at_x():
    with raises(IndexError):
        y_at_x(0, [], [])

    with raises(ValueError):
        y_at_x(0, [1, 2, 3], [])

    with raises(IndexError):
        y_at_x(0, [1, 2, 3], [4, 5, 6])

    assert 4 == y_at_x(1,   [1, 2, 3], [4, 5, 6])
    assert 5 == y_at_x(2,   [1, 2, 3], [4, 5, 6])
    assert 6 == y_at_x(2.5, [1, 2, 3], [4, 5, 6])
    assert 6 == y_at_x(3,   [1, 2, 3], [4, 5, 6])
    assert 5 == y_at_x(2,   [3, 2, 1], [4, 5, 6])
    assert 6 == y_at_x(2.5, [3, 2, 1], [4, 5, 6])