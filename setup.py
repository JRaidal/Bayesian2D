
from setuptools import setup, find_packages

setup(
    name='Bayesian2D',
    version='0.3.0',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Package used to find the maximum or minimum of any 2D function using Bayesian optimization',
    long_description=open('README.md').read(),
    url='https://github.com/JRaidal/Bayesian2D',
    install_requires=['numpy', 'datetime', 'scipy', 'sklearn', 'matplotlib'],
    author='Juhan Raidal',
    author_email='juhantiitus.raidal@gmail.com'
)
