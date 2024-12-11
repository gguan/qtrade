from setuptools import setup, find_packages

setup(
    name='qtrade',
    version='0.1.0',
    description='A quantitative trading framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/qtrade',
    packages=find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.2.0',
        'scipy>=1.6.0',
        'matplotlib>=3.3.0',
        'bokeh>=3.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.0.0',
            'black>=21.0.0',
            'isort>=5.0.0',
            'flake8>=3.9.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'qtrade=qtrade.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
)