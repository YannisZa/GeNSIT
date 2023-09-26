import os

from setuptools import find_packages, setup

from multiresticodm import __version__

requirementPath="requirements.txt"
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='multiresticodm',
    version=__version__,
    description="A command line tool for origin destination matrix inference.",
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'multiresticodm = multiresticodm.main:cli'
        ],
    },
)
