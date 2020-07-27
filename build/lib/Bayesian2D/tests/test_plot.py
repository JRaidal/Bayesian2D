import unittest
from Bayesian2D.tools import plot
import numpy


class TestPlot(unittest.TestCase):

	def setUp(self):
		self.x_bounds = ([-5, 5])
		self.y_bounds = ([-2, 2])

		
	def tearDown(self):
		self.x_bounds = 0
		self.y_bounds = 0

		
	def test_output_type(self):
		def plot_func(x, y):
			return x**2 + y**2
		self.assertIsNone(plot(plot_func, self.x_bounds, self.y_bounds))
