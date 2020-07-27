
from setuptools import setup, find_packages

setup(
    name='Bayesian2D',
    version='0.2.5',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Package used to find the maximum or minimum of any 2D function using Bayesian optimization',
    long_description=open('README.txt').read(),
    install_requires=['numpy', 'datetime', 'scipy', 'sklearn', 'matplotlib'],
    author='Juhan Raidal',
    author_email='myemail@example.com'
)
