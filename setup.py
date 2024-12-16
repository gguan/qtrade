from setuptools import setup, find_packages

setup(
    name='qtrade',
    version='0.1.0',
    description='A Python library for backtesting trading strategies and applying reinforcement learning to trading.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Guan Guan',
    author_email='guanguan1114@gmail.com',
    url='https://github.com/yourusername/qtrade',
    packages=find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'pandas',
        'scipy>=1.6.0',
        'matplotlib>=3.3.0',
        'bokeh>=3.1.1',
        'tqdm>=4.0.0',
        'gymnasium>=1.0.0',
        'mplfinance>=0.12.10b0',
    ],
    extras_require={
        'test': [
            'pytest',
            'coverage'
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
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
)