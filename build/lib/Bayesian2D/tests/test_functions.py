import unittest

from Bayesian2D.tools import functions


class TestFunctions(unittest.TestCase):

	def setUp(self):
		self.x_float = 2.2
		self.y_float = 3.3
		self.x_int = 2
		self.y_int = 3
		
		
	def tearDown(self):
		self.x = 0
		self.y = 0
		
	def test_output_float(self):
		# functions(self.x, self.y, 'Rosenbrock') == float
		self.assertEqual(type(functions(self.x_float, self.y_float, 'Rosenbrock')) , float)
	def test_output_int(self):	
		self.assertEqual(type(functions(self.x_int, self.y_int, 'Rosenbrock')), int)



