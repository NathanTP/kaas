import setuptools
import shutil
import subprocess as sp
import pathlib

VERSION = '0.0.1'

with open("requirements.txt", "r") as f:
    install_requires = f.readlines()

if shutil.which('nvcc') is not None:
    install_requires.append('pycuda')

with open("../README.md", "r") as fh:
    long_description = fh.read()

#XXX
# sp.run(['make'], cwd=pathlib.Path(__file__).parent / 'kaas/cutlass', check=True)

setuptools.setup(
    name="kaas",
    version=VERSION,
    author="Nathan Pemberton",
    author_email="nathanp@berkeley.edu",
    description="Kernel-as-a-Service: A GPU-native serverless function",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nathantp/kaas.git",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.8',
    include_package_data=True
)
