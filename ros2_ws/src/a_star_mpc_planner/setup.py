import glob

from setuptools import setup

package_name = 'a_star_mpc_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lorenzo Ortolani',
    maintainer_email='lorenzo.ortolani@talosrobotics.ai',
    description='A* + MPC local planner for the Unitree G1 humanoid, adapted to '
                'the Navigation DLIO stack (pose from /dlio/odom_node/odom, '
                'obstacles from g1_local_map /local_voxel_map/obstacles, velocity '
                'out as Twist on /mpc/cmd_vel → AMO WebSocket via '
                'g1_sim_bridge/cmd_vel_to_amo_node).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'a_star_node = a_star_mpc_planner.a_star_node:main',
            'mpc_node = a_star_mpc_planner.mpc_node:main',
        ],
    },
)
