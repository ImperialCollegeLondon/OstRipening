import os
import shutil
import glob
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

class BuildExtAndMove(build_ext):
    """Custom build class to move the .so file to the parent directory."""
    def run(self):
        super().run()
        # Look for the compiled shared library in the current folder
        # Files usually look like cluster_cpp.cpython-310-x86_64-linux-gnu.so
        ext_files = glob.glob("cluster_cpp*.so") + glob.glob("cluster_cpp*.pyd")
        
        for filename in ext_files:
            destination = os.path.join("..", filename)
            print(f"Moving {filename} to {destination}")
            # Use copy/move to place it in the parent folder
            shutil.copy2(filename, destination)

ext_modules = [
    Pybind11Extension(
        "cluster_cpp",
        ["module_wrapper.cpp", "clusterManipulation.cpp", "compute_fluxes_moles_pc_conc.cpp"],
        cxx_std=11,  # Ensure this matches your C++ version
        extra_compile_args=['-fopenmp'], # Tells the compiler to recognize OpenMP
        extra_link_args=['-fopenmp'],    # Tells the linker to include the OpenMP library
    ),
]

setup(
    name="cluster_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtAndMove},
)

