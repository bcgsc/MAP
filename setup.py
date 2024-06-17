import shutil
from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import atexit

def cleanup():
    egg_info_dir = os.path.join(os.getcwd(), 'map.egg-info')
    if os.path.exists(egg_info_dir):
        print(f"Removing {egg_info_dir}")
        shutil.rmtree(egg_info_dir)
        print(f"{egg_info_dir} removed")
    build_dir = os.path.join(os.getcwd(), 'build')
    if os.path.exists(build_dir):
        print(f"Removing {build_dir}")
        shutil.rmtree(build_dir)
        print(f"{build_dir} removed")

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        atexit.register(cleanup)

setup(
    name='map',
    version='1.0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'map=map.main:main',
        ],
    },
    install_requires=[
        'pandas',
        'numpy',
        'progress',
        'openpyxl',
        'xlsxwriter',
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
)