from setuptools import setup
from glob import glob
import os
package_name = 'yolopyinference_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files.
        ('share/' + package_name, glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='admin',
    maintainer_email='kaikai.liu@sjsu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'YoloDecoder = yolopyinference_ros.YoloDecoder:main',
            'TrtDetector = yolopyinference_ros.trtdetect:main',
            'image_visualizer = yolopyinference_ros.visualization:main'
        ],
    },
)
