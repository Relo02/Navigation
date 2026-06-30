import glob

from setuptools import setup

package_name = 'g1_sim_bridge'

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
    description='DLIO sensor bridges: Isaac Sim QoS relay (sim) + Livox IMU '
                'g->m/s^2 rescale (real robot).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'isaac_dlio_qos_relay_node = '
            'g1_sim_bridge.isaac_dlio_qos_relay_node:main',
            'imu_rescale_node = g1_sim_bridge.imu_rescale_node:main',
            'cmd_vel_to_amo_node = g1_sim_bridge.cmd_vel_to_amo_node:main',
            'joy_to_cmdvel_node = g1_sim_bridge.joy_to_cmdvel_node:main',
            'estop_keyboard_node = g1_sim_bridge.estop_keyboard_node:main',
        ],
    },
)
