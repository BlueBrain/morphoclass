from __future__ import annotations

import pytest
import tmd.Tree.Tree


@pytest.fixture()
def my_list():
    return []


@pytest.fixture()
def simple_apical():
    x = [0, 0, 5, 10, -10]
    y = [0, 10, 15, 20, 20]
    z = [0, 0, 0, 0, 0]
    d = [1, 1, 1, 1, 1]
    t = [4, 4, 4, 4, 4]
    p = [-1, 0, 1, 2, 1]

    return tmd.Tree.Tree(x=x, y=y, z=z, d=d, t=t, p=p)


@pytest.fixture()
def simple_apical_2():
    x = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 5, -10]
    y = [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 25, 20]
    z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    d = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    t = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    p = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 1]

    return tmd.Tree.Tree(x=x, y=y, z=z, d=d, t=t, p=p)


@pytest.fixture()
def simple_apical_3():
    x = [0, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -13, -14, -15, 10]
    y = [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 20]
    z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    d = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    t = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    p = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1]

    return tmd.Tree.Tree(x=x, y=y, z=z, d=d, t=t, p=p)
