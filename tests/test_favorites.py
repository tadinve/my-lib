import pytest

from my_lib import add, fib, flatten


def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0


def test_fib():
    assert fib(0) == 0
    assert fib(1) == 1
    assert fib(10) == 55
    with pytest.raises(ValueError):
        fib(-1)


def test_flatten():
    assert flatten([[1, 2], (3, 4)]) == [1, 2, 3, 4]
    assert flatten([]) == []
