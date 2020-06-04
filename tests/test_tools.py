import sys

from numpy.testing import assert_almost_equal as aae

from pytest import raises

sys.path.insert(0, '..')

from spectra.tools import (cull, glob_read_csvs, index_of_x, integrate,
                           read_csv, read_csvs, y_at_x)


def setup():
    pass


def teardown():
    pass


def test_read_csv(tmp_path):
    path = f'{tmp_path}/test.csv'
    data = [
        ['A', 'B', 'C', 'D'],
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(','.join(map(str, row)) for row in data))

    csv = read_csv(path)
    assert csv[0] == data[0]
    aae(csv[1], [1, 5])
    aae(csv[2], [[2, 6], [3, 7], [4, 8]])

    csv = read_csv('tests/files/1-butanol + N 3400/1.00% T12/Round 1/Thu Jul 25 14-53-51 2019 (GMT-04-00).CSV')


def test_read_csvs(tmpdir):
    p1 = tmpdir.mkdir("sub").join("test1.csv")
    p2 = tmpdir.join("test2.csv")
    data1 = [
        ['A', 'B', 'C', 'D'],
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]
    data2 = [
        ['A', 'B'],
        [6, 7],
        [8, 9]
    ]

    data_str1 = '\n'.join(','.join(map(str, row)) for row in data1)
    data_str2 = '\n'.join(','.join(map(str, row)) for row in data2)
    p1.write(data_str1)
    p2.write(data_str2)
    assert len(tmpdir.listdir()) == 2

    read_csvs(str(p1))
    read_csvs([str(p1)])
    read_csvs(str(p2))
    titles, xs, ys = read_csvs([p1, p2])
    assert titles == ['B', 'C', 'D', 'B']
    aae(xs, [[1, 5], [1, 5], [1, 5], [6, 8]])
    aae(ys, [[2, 6], [3, 7], [4, 8], [7, 9]])


def test_glob_read_csvs():
    file_dir1 = 'tests/files/1-butanol + N 3400/1.00% T12/Round 1'
    titles, xs, ys, file_names = glob_read_csvs(f'{file_dir1}/Thu Jul 25 14*.CSV')
    assert len(file_names) == 18

    file_dir2 = 'tests/files/1-butanol + N 3400/0.03% T12/Round 1'
    titles, xs, ys, file_names = glob_read_csvs([f'{file_dir2}/Thu Aug 01 08-49*.CSV'])
    assert len(file_names) == 3


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

    assert 4 == y_at_x(1, xs, ys)
    assert 5 == y_at_x(2, xs, ys)
    assert 6 == y_at_x(2.5, xs, ys)
    assert 4 == y_at_x(3, xs[::-1], ys)
    assert 5 == y_at_x(2, xs[::-1], ys)
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
