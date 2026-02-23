#include "cluster.h"
#include <pybind11/numpy.h>  // <--- This MUST be here for py::array_t
#include <pybind11/stl.h>
#include <vector>
#include <omp.h> // For parallel processing


namespace py = pybind11;

struct EventResult {
    bool triggerShrinkage;
    bool triggerGrowth;
};

std::pair<bool, bool> checkEvents(
    const std::vector<bool>& validToShrink,
    const std::vector<bool>& validToGrow,
    const std::vector<int>& clustSize,
    const std::vector<double>& clustPc,
    const std::vector<double>& clustPcShrink,
    const std::vector<double>& clustPcMax,
    py::array_t<bool> clustersToShrink,
    py::array_t<bool> clustersToGrow,
    const std::vector<double>& clustMoles,
    const std::vector<int>& toImbibe,
    const std::vector<double>& volarray,
    double factor,
    int nClusters) 
{
    auto res_shrink = clustersToShrink.mutable_unchecked<1>();
    auto res_grow = clustersToGrow.mutable_unchecked<1>();
    bool globalShrink = false;
    bool globalGrow = false;

    #pragma omp parallel for reduction(|:globalShrink, globalGrow)
    for (int k = 0; k < nClusters; ++k) {
        res_shrink[k] = false;
        res_grow[k] = false;

        if (validToShrink[k]) {
            if (clustPc[k] < clustPcShrink[k]) {
                size_t targetIdx = static_cast<size_t>(toImbibe[k]);
                if ((clustMoles[k] / volarray[targetIdx] / factor) > 1.0) {
                    continue;
                }
                res_shrink[k] = true;
                globalShrink = true;
            }     
        } else if (validToGrow[k]) {
            if (clustPc[k] > clustPcMax[k]) {
                res_grow[k] = true;
                globalGrow = true;
            }
        }
    }

    return {globalShrink, globalGrow};
}


void init_cluster_manipulation(pybind11::module &m) {
    m.doc() = "C++ implementation of cluster shrinkage and growth events";

    // Register the result struct so Python can read the fields
    py::class_<EventResult>(m, "EventResult")
        .def_readonly("triggerShrinkage", &EventResult::triggerShrinkage)
        .def_readonly("triggerGrowth", &EventResult::triggerGrowth);

    // Bind the function
    m.def("checkEvents", &checkEvents, 
          "A function to check cluster shrinkage/growth",
          py::arg("validToShrink"),
          py::arg("validToGrow"),
          py::arg("clustSize"),
          py::arg("clustPc"),
          py::arg("clustPcShrink"),
          py::arg("clustPcMax"),
          py::arg("clustersToShrink"),
          py::arg("clustersToGrow"),
          py::arg("clustMoles"),
          py::arg("toImbibe"),
          py::arg("volarray"),
          py::arg("factor"),
          py::arg("nClusters"));
}