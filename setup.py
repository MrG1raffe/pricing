from setuptools import setup, find_packages

setup(
   name='pricing',
   version='1.0',
   description='Option pricing models',
   author='MrG1raffe',
   author_email='dimitri.sotnikov@gmail.com',
   packages=find_packages(),
   install_requires=[
      'numpy>=1.23.0',
      'typing',
      'scipy',
      'pandas',
      'dataclasses',
      'matplotlib',
      'numba>=0.58.1',
      'py_vollib>=1.0.0',
      'tqdm',
      'simulation @ git+https://github.com/MrG1raffe/simulation.git'
      #'signature @ git+https://github.com/MrG1raffe/signature.git'
   ],
)