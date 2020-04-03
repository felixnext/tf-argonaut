'''
Setup Script for the Argo

This will install the module to the local python distribution.
'''


import os
from setuptools import setup
from setuptools import find_packages


__status__      = "Package"
__copyright__   = "Copyright 2020"
__license__     = "Apache License 2.0"
__version__     = "0.2.0"

# 01101100 00110000 00110000 01110000
__author__      = "Felix Geilert"


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='tf-argonaut',
      version=__version__,
      description='Tensorflow Experimentation Pipeline',
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords='tensorflow experimentation computervision',
      url='https://github.com/felixnext/tf-argonaut',
      author='Felix Geilert',
      license='Apache License 2.0',
      packages=find_packages(),
      entry_points={
          'console_scripts': ['argo-datavis=argonaut.utils.visualization:visualize_dataset'],
      },
      install_requires=[ 'numpy', 'fire', 'bunch', 'progressbar2', 'pandas', 'selectivesearch' ],
      include_package_data=True,
      zip_safe=False)