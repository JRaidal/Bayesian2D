import unittest
from Bayesian2D.tools import results, Bayesian2D
import numpy

class TestBayesian2D(unittest.TestCase):

	def setUp(self):
		self.max_min = 'maximum'
		self.function = 'Rosenbrock'
		self.starting_n = 5
		self.x_bounds = ([-5, 5])
		self.y_bounds = ([-2, 2])
		self.iterations = 10
		self.exploration = 5


	def tearDown(self):
		self.starting_n = 0
		self.x_bounds = 0
		self.y_bounds = 0
		self.function = None
		self.iterations = 0
		self.exploration = 0
		self.max_min = None

		
	def test_output_type_xy(self):
		x, y, z_best = Bayesian2D(self.x_bounds, self.y_bounds, self.starting_n, self.iterations, self.max_min, self.exploration, self.function)
		self.assertEqual(type(x), float and type(y), float)

	def test_output_type_z(self):
		x, y, z_best = Bayesian2D(self.x_bounds, self.y_bounds, self.starting_n, self.iterations, self.max_min, self.exploration, self.function)
		self.assertEqual(type(z_best), numpy.float64)

