import unittest
import numpy as np
from typing import List

from augmentation.augMethods import get_shape_contour


class AugmentationTest(unittest.TestCase):
    def test_get_contours_from_shape(self) -> None:
        points: List[List[float]] = [[0, 10],
                                     [10, 10],
                                     [10, 0],
                                     [0, 0]]
        cnt: np.ndarray = get_shape_contour(points)
        self.assertTrue(cnt.shape == (len(points), 1, 2))
        for i in range(len(points)):
            self.assertEqual(points[i][0], cnt[i][0][0])
            self.assertEqual(points[i][1], cnt[i][0][1])
