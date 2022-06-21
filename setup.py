from setuptools import setup

setup(
    name='opencv_multicam_recorder',
    version='0.1.0',
    author='Javier Felip Leon',
    author_email='javier.felip.leon@gmail.com',
    packages=['opencv_multicam_recorder'],
    scripts=[],
    url='https://github.com/jfelip/opencv-multicam-recorder.git',
    license='LICENSE',
    description='An awesome package that does something',
    long_description=open('README.md').read(),
    install_requires=[
        "opencv-python",
        "numpy",
    ],
)