from setuptools import setup

setup(
    name='spectra',
    version='0.1',
    packages=['spectra'],
    url='',
    license='MIT',
    author='Jonathon Vandezande',
    author_email='jevandezande@gmail.com',
    description='A simple tool for plotting spectra',
    install_requires=['matplotlib', 'natsort', 'numpy', 'pytest', 'scipy'],
)
