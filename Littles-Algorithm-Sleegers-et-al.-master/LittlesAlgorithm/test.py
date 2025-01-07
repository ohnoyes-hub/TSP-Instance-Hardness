import unittest
import numpy as np
import numpy.ma as ma

from .algorithm import get_minimal_route, reduce_matrix, find_all_subtours

def assert_equal_masked_array(array1, array2):
  assert np.alltrue((array1 == array2).compressed())

class TestReduceMatrix(unittest.TestCase):

  def test_reduce_matrix_1(self):
    """
    Test if the test matrix is reduced right and if the corresponding
    lower bound is correct.
    """
    test_matrix = np.array([
      [np.inf, 27, 43, 16, 30, 26],
      [7, np.inf, 16, 1, 30, 25],
      [20, 13, np.inf, 35, 5, 0],
      [21, 16, 25, np.inf, 18, 18],
      [12, 46, 27, 48, np.inf, 5],
      [23, 5, 5, 9, 5, np.inf]
    ])
    result_should_be = np.array([
      [np.inf, 11, 27, 0, 14, 10],
      [1, np.inf, 15, 0, 29, 24],
      [15, 13, np.inf, 35, 5, 0],
      [0, 0, 9, np.inf, 2, 2],
      [2, 41, 22, 43, np.inf, 0],
      [13, 0, 0, 4, 0, np.inf]
    ])
    result, reduction = reduce_matrix(test_matrix)
    np.testing.assert_equal(result, result_should_be)
    self.assertEqual(48, reduction)

  def test_reduce_matrix_2(self):
    """
    Test if the test matrix is reduced right and if the corresponding
    lower bound is correct.
    """
    test_matrix = ma.array(
      np.array([
        [np.inf, 11, 27, 0, 14, 10],
        [1, np.inf, 15, 0, 29, 24],
        [15, 13, np.inf, 35, 5, 0],
        [np.inf, 0, 9, np.inf, 2, 2],
        [2, 41, 22, 43, np.inf, 0],
        [13, 0, 0, 4, 0, np.inf]
      ]),
      mask=np.array(
        [
          [1, 1, 1, 1, 1, 1],
          [0, 0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0, 0]
        ]
      )
    )
    result_should_be = ma.array(
      np.array([
        [np.inf, 11, 27, 0, 14, 10],
        [0, np.inf, 14, 0, 28, 23],
        [15, 13, np.inf, 35, 5, 0],
        [np.inf, 0, 9, np.inf, 2, 2],
        [2, 41, 22, 43, np.inf, 0],
        [13, 0, 0, 4, 0, np.inf]
      ]),
      mask=np.array(
        [
          [1, 1, 1, 1, 1, 1],
          [0, 0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0, 0]
        ]
      )
    )
    result, reduction = reduce_matrix(test_matrix)
    assert_equal_masked_array(result, result_should_be)
    self.assertEqual(1, reduction)

  def test_find_all_subtours(self):

    included = [(1, 2), (2, 3), (3, 4), (9, 8), (8, 7), (7, 6)]
    result_should_be = [(1, 4), (9, 6)]
    result = find_all_subtours(included)

    self.assertEqual(result, result_should_be)

if __name__ == '__main__':
  unittest.main()
