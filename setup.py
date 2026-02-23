from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "utilities_cpp",
        ["compute_fluxes_moles_pc_conc.cpp"],
        extra_compile_args=['-O3', '-fopenmp'], # Linux/Mac (use /openmp for Windows MSVC)
        extra_link_args=['-lgomp'],
    ),
]

setup(name="utilities_cpp", ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
