#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'sympl',
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='chomp',
    version='0.1.0',
    description="Closing Higher Orders with a Machine Learning (ML) Parameterization.",
    long_description=readme + '\n\n' + history,
    author="Jeremy McGibbon",
    author_email='mcgibbon@uw.edu',
    url='https://github.com/mcgibbon/chomp',
    packages=[
        'chomp',
    ],
    package_dir={'chomp':
                 'chomp'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='chomp',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
