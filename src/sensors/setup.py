from setuptools import find_packages, setup

package_name = 'sensors'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ee1240935',
    maintainer_email='ee1240935@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lidar_shape_detector = sensors.lidar_shape_detector:main',
            'shape_detector_task2A = sensors.shape_detector_task2A:main',
            'task1B = sensors.task1B:main',
            'arm_manipulator_task2B = sensors.arm_manipulator_task2B:main',
            'arm_coordination_task2B = sensors.arm_coordination_task2B:main',
            'aruco_fruits_task2B = sensors.aruco_fruits_task2B:main',
            'joint_angle_logger = sensors.joint_angle_logger:main',
            'arm_manipulator_task3B = sensors.arm_manipulator_task3B:main',
            'arm_perception_task2B = sensors.arm_perception_task2B:main',
            'fruits_task3A = sensors.fruits_task3A:main',
            'idk = sensors.idk:main',
            'moving_end_effector = sensors.moving_end_effector:main',
            'arm_manipulator_task4C = sensors.arm_manipulator_task4C:main',
            'csv_joint_replay_sequential = sensors.csv_joint_replay_sequential:main',
            'rotating_gautam = sensors.rotating_gautam:main',
            'csv_joint_hardware = sensors.csv_joint_hardware:main',
            'joint_jogger_only_rotation_hardware_animesh = sensors.joint_jogger_only_rotation_hardware_animesh:main',
            'joint_jogger_only_rotation_simulation_animesh = sensors.joint_jogger_only_rotation_simulation_animesh:main',
        ],
    },
)
