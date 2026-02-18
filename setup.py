from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

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
        extra_compile_args=["-O3", "-march=native", "-std=c++17"],
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
