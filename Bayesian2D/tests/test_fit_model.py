import unittest
from Bayesian2D.tools import fit_model
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import Matern

class TestFitModel(unittest.TestCase):

	def setUp(self):
		self.model = GaussianProcessRegressor(kernel = Matern(), alpha = 1e-10)
		self.data_input = np.array([[1, 3], [4, 7]])
		self.data_output = np.array([15 , 6])

		
	def tearDown(self):
		self.data_input = 0
		self.data_output = 0

		
	def test_output_None(self):

		self.assertIsNone(fit_model(self.model, self.data_input, self.data_output))

