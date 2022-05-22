from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="JPEG",
    author="Robin Senn",
    author_email="rsenn@hs-mittweida.de",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy == 1.20.3", "Pillow == 8.2.0", "Cython == 0.29.23", "blessings==1.7"],
    scripts=["bin/jpeg-cli"],
    ext_modules=cythonize(
        Extension(
            "color_space_transform",
            sources=["src/color_space_transform.pyx"],
            include_dirs=[numpy.get_include()],
        )
    ),
)
