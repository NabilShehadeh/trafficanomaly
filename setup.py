# Setup script for Traffic Anomaly

from setuptools import setup, find_packages

setup(
    name='TrafficAnomaly',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
            'traffic-anomaly=traffic_anomaly:main',
        ],
    },
)