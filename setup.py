#!/usr/bin/env python3

from setuptools import setup

# Function to read the requirements from 'requirements.txt'
def read_requirements():
    with open('requirements.txt', 'r') as req:
        return req.read().splitlines()

setup(name='Odin',
      version='0.0.0',
      description="Odin is",
      author='Dario Cazzani',
      packages=['odin'],
      classifiers=[
        "TBD"
      ],
      install_requires=read_requirements(),
      python_requires='>=3.11',
      include_package_data=True)