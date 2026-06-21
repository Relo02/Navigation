import glob

from setuptools import setup

package_name = 'g1_local_map'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lorenzo Ortolani',
    maintainer_email='lorenzo.ortolani@talosrobotics.ai',
    description='Local rolling 3D voxel map (DLIO deskewed cloud -> ground-removed '
                'obstacle cloud + voxel grid + costmap) for the a_star_mpc planner.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'local_voxel_map_node = g1_local_map.local_voxel_map_node:main',
            'ground_removal_node = g1_local_map.ground_removal_node:main',
        ],
    },
)
