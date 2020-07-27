import unittest
from Bayesian2D.tools import acquisition
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import Matern

class TestAcquisition(unittest.TestCase):

	def setUp(self):
		self.XY = np.array([[1, 2], [3, 4]])
		self.x_bounds = ([-10, 10])
		self.y_bounds = ([-10, 10])
		self.e = 0.5
		self.max_min = 'minimum' 
		self.model = GaussianProcessRegressor(kernel = Matern(), alpha = 1e-10)

		
	def tearDown(self):
		self.XY = 0
		self.x_bounds = 0
		self.y_bounds = 0
		self.e = 0
		self.max_min = '' 
		
	def test_output_float(self):
		
		x, y = acquisition(self.XY, self.x_bounds, self.y_bounds, self.e, self.model, self.max_min)

		self.assertEqual(type(x), float and type(y), float)

