import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sfm_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='donggil',
    maintainer_email='user@todo.todo',
    description='SfM Data Collection Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'auto_scanner = sfm_pipeline.auto_scanner:main',
            'colmap_converter = sfm_pipeline.colmap_converter:main',
            'ai_depth_scanner = sfm_pipeline.ai_depth_scanner:main',
            'gaussian_data_collector = sfm_pipeline.gaussian_data_collector:main',
            'one_shot_grasp = sfm_pipeline.one_shot_grasp:main',
            'grasp_executor = sfm_pipeline.grasp_executor:main',
        ],
    },
)