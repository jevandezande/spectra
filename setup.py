from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='spectra',
    version='0.2',
    description='Utility for analyzing and plotting 1D spectra.',
    long_description=readme(),
    packages=['spectra'],
    url='',
    license='MIT',
    author='Jonathon Vandezande',
    author_email='jevandezande@gmail.com',
    install_requires=['matplotlib', 'natsort', 'numpy', 'pytest', 'scipy'],
    tests_require=['pytest'],
)
