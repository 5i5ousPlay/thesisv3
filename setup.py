from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'prototype package for Bomediano And Friends Thesis'
AUTHOR = '5i5ousplay'


setup(
    name='thesisv3',
    packages=find_packages(exclude=['tests', 'test_*']),
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    install_requires=[
        'contourpy==1.3.1',
        'cvxopt==1.3.2',
        'cycler==0.12.1',
        'Cython==3.0.11',
        'fonttools==4.55.0',
        'future==1.0.0',
        'GraKeL==0.1.10',
        'joblib==1.4.2',
        'kiwisolver==1.4.7',
        'matplotlib==3.9.2',
        'networkx==3.4.2',
        'numpy==1.26.4',
        'packaging==24.2',
        'pandas==2.2.3',
        'pillow==11.0.0',
        'pyparsing==3.2.0',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.2',
        'scikit-learn==1.5.2',
        'scipy==1.13.0',
        'six==1.16.0',
        'threadpoolctl==3.5.0',
        'tzdata==2024.2'
        ]
    )
