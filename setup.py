from setuptools import setup

package_name = 'yolo_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='YOLO segmentation node for real-time video processing',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segmentation_node = yolo_segmentation.segmentation_node:main',
        ],
    },
)
