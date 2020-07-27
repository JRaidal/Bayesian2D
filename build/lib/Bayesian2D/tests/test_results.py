import unittest
from Bayesian2D.tools import results, acquisition, optimize
import numpy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import Matern

class TestResults(unittest.TestCase):

	def setUp(self):
		self.max_min = 'maximum'
		self.model = GaussianProcessRegressor(kernel = Matern(), alpha = 1e-10)
		self.starting_n = 5
		self.x_bounds = ([-5, 5])
		self.y_bounds = ([-2, 2])
		self.iterations = 10
		self.e = 5
		self.acquisition = acquisition


	def tearDown(self):
		self.starting_n = 0
		self.x_bounds = 0
		self.y_bounds = 0
		self.model = None
		self.iterations = 0
		self.e = 0
		self.max_min = None

		
	def test_output_type_xy(self):
		def opt_func(x, y):
			return x**2 + y**2
		x, y, z = results(opt_func, self.acquisition, self.starting_n, self.x_bounds, self.y_bounds, self.iterations, self.e, self.model, self.max_min)
		self.assertEqual(type(x), float and type(y), float)

	def test_output_type_z(self):
		def opt_func(x, y):
			return x**2 + y**2
		x, y, z = results(opt_func, self.acquisition, self.starting_n, self.x_bounds, self.y_bounds, self.iterations, self.e, self.model, self.max_min)
		self.assertEqual(type(z), numpy.float64)
