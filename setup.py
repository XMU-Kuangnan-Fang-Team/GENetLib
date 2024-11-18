from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='GENetLib',
    version='1.1.4',
    description='A Python Library for Gene–environment Interaction Analysis via Deep Learning',
    author='Yuhao Zhong',
    author_email='Barry57@163.com',
    url='https://github.com/Barry57/GENetLib/',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'scikit-learn',
        'scipy',
        'matplotlib',
        'setuptools' 
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
    ],
)
