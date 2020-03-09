from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='spectra',
    version='0.3',
    description='Utility for analyzing and plotting 1D spectra.',
    long_description=readme(),
    packages=['spectra'],
    url='',
    license='MIT',
    author='Jonathon Vandezande',
    author_email='jevandezande@gmail.com',
    scripts=['bin/plot_spectra', 'bin/correlate_spectra'],
    install_requires=['lmfit', 'matplotlib', 'natsort', 'numpy', 'pytest', 'scipy'],
    tests_require=['pytest'],
)
