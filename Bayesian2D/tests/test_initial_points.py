import unittest
from Bayesian2D.tools import initial_points
import numpy


def opt_func(x, y):
	return x**2 + y**2

class TestInitialPoints(unittest.TestCase):

	def setUp(self):
		self.starting_n = 5
		self.x_bounds = ([-5, 5])
		self.y_bounds = ([-2, 2])

		
	def tearDown(self):
		self.starting_n = 0
		self.opt_func = None
		self.x_bounds = 0
		self.y_bounds = 0

		
	def test_output_type(self):

		XY, z = initial_points(self.starting_n, opt_func, self.x_bounds, self.y_bounds)
		self.assertEqual(type(XY), numpy.ndarray and type(z), numpy.ndarray )
