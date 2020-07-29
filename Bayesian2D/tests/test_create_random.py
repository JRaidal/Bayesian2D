import unittest
from Bayesian2D.tools.create_random import create_random
import numpy
from numpy.random import uniform

class TestCreateRandom(unittest.TestCase):
	def setUp(self):
		self.x_bounds = ([-5, 5])
		self.y_bounds = ([-2, 2])
		self.n_random = 10000

	def tearDown(self):
		self.n_random = 0
		self.x_bounds = 0
		self.y_bounds = 0

		
	def test_output_type(self):
		Ysamples = create_random(self.x_bounds, self.y_bounds, self.n_random)
		self.assertEquals(type(Ysamples), numpy.ndarray)

	def test_output_length(self):
		XYsamples = create_random(self.x_bounds, self.y_bounds, self.n_random)
		self.assertEquals(len(XYsamples), self.n_random)

