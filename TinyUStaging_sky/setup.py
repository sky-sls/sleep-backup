from setuptools import setup, find_packages
from ustaging import __version__

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

with open("requirements.txt") as req_file:
    requirements = list(filter(None, req_file.read().split("\n")))

setup(
    name='ustaging',
    version=__version__,
    description='TinyUStaging: An Efficient Model for Sleep Staging with Single-Channel EEG and EOG.',
    long_description=readme + "\n\n" + history,
    author='LJY 203509',
    author_email='ljyljyok@gmail.com',
    url='https://github.com/ljyljy/TinyUStaging',
    license="LICENSE.txt",
    packages=find_packages(),
    package_dir={'ustaging':
                 'ustaging'},
    include_package_data=True,
    setup_requires=["setuptools_git>=0.3",],
    entry_points={
       'console_scripts': [
           'sky=ustaging.bin.us:entry_func',
       ],
    },
    install_requires=requirements,
    classifiers=['Environment :: Console',
                 'Operating System :: POSIX',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7'
                 'License :: OSI Approved :: MIT License']
)
