from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'torchvision>=0.12.0',
    'pillow>=9.0.1',
    'tqdm>=4.64.0',
    'mlflow>=1.26.0',
    'yapf>=0.31.0',
    'pylint>=2.12.2',
    'tangled-up-in-unicode==0.1.0',
    'h5py',
    'typing-extensions',
    'wheel',
    'transformers',
    'datasets',
    'tqdm',
    'cloudml-hypertune'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application.'
)