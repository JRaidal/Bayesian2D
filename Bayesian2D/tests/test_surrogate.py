import unittest
from Bayesian2D.tools import surrogate, acquisition
import numpy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import Matern

class TestSurrogate(unittest.TestCase):

	def setUp(self):
		self.XY = numpy.array([[3, 4], [5, 7]])
		self.model = GaussianProcessRegressor(kernel = Matern(), alpha = 1e-10)


	def tearDown(self):
		self.XY = 0
		self.model = None

		
	def test_output_type(self):
		mean, std = surrogate(self.model, self.XY)
		self.assertEqual(type(mean), numpy.ndarray and type(std), numpy.ndarray )
