#include <pybind11/pybind11.h>

// Forward declare the registration functions
void init_cluster_manipulation(pybind11::module &m);
void init_flux_calculations(pybind11::module &m);

PYBIND11_MODULE(cluster_cpp, m) {
    m.doc() = "My combined C++ extension";

    // Initialize each sub-part of your module
    init_cluster_manipulation(m);
    init_flux_calculations(m);
}