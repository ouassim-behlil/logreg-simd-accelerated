import sys
import platform
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Pick compiler flags based on platform
if platform.system() == "Windows":
    extra_args = ["/O2", "/arch:AVX2", "/std:c++17"]
else:
    extra_args = ["-O3", "-march=native", "-std=c++17"]

ext_modules = [
    Pybind11Extension(
        "logreg",
        sources=[
            "bindings/py_logreg.cpp",
            "logreg/LogisticRegression.cpp",
            "logreg/dispatcher.cpp",
            "logreg/dot_product.cpp",
            "logreg/vect_sigmoid.cpp",
            "utils/aligned_alloc.cpp",
        ],
        include_dirs=["logreg/include"],
        extra_compile_args=extra_args,
        language="c++",
    ),
]

setup(
    name="logreg",
    version="0.1.0",
    author="ouassim",
    description="SIMD-accelerated binary logistic regression with NumPy bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=["numpy"],
)
