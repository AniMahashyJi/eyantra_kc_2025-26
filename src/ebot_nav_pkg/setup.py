from setuptools import find_packages, setup

package_name = 'ebot_nav_pkg'

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
            'ebot_nav_task3B = ebot_nav_pkg.ebot_nav_task3B:main',
            'ebot_nav_task4B = ebot_nav_pkg.ebot_nav_task4B:main',
            'ebot_nav_task4C = ebot_nav_pkg.ebot_nav_task4C:main',
            'ebot_nav_task3B_multithreading = ebot_nav_pkg.ebot_nav_task3B_multithreading:main',
            'ebot_nav_task5 = ebot_nav_pkg.ebot_nav_task5:main',
        ],
    },
)
