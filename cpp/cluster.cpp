#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <set>
#include <cstdint>
#include <omp.h>

namespace py = pybind11;

struct Cluster {
    int nClusters = 0;
    std::vector<int> clusterSize;
    std::vector<int> members;
    std::vector<int> members_pos;
    std::vector<int> members_offset;
    std::vector<double> pc;
    std::vector<uint8_t> trappedStatus;
    std::vector<uint8_t> connected;
    std::vector<double> pcMax;
    std::vector<double> pcShrink;
    std::vector<int> clusterID;
    std::set<int> availableID;

    Cluster() = default;

    // --- RESIZE LOGIC ---
    void resize(int sizeToAdd) {
        int oldSize = static_cast<int>(pc.size());
        int newSize = oldSize + sizeToAdd;

        clusterSize.resize(newSize, 0);
        members_offset.resize(newSize, -5);
        pc.resize(newSize, 0.0);
        trappedStatus.resize(newSize, 0);
        connected.resize(newSize, 0);
        pcMax.resize(newSize, 0.0);
        pcShrink.resize(newSize, 0.0);

        for (int c = oldSize; c < newSize; ++c) {
            if (c > 0) availableID.insert(c); 
        }
        nClusters = newSize;
    }


    // --- PARALLEL EVENT CHECKING ---
    py::dict checkEvents(const std::vector<uint8_t>& validToShrink, 
                        const std::vector<uint8_t>& validToGrow,
                        double initialAqAvgPres) {
        
        std::vector<uint8_t> clustersToShrink(nClusters, 0);
        std::vector<uint8_t> clustersToGrow(nClusters, 0);
        bool triggerShrink = false;
        bool triggerGrow = false;

        #pragma omp parallel for reduction(|:triggerShrink, triggerGrow)
        for (int k = 0; k < nClusters; ++k) {
            if (validToShrink[k]) {
                if ((clusterSize[k] == 1 && pc[k] <= initialAqAvgPres) ||
                    (clusterSize[k] > 1 && pc[k] < pcShrink[k])) {
                    clustersToShrink[k] = 1;
                    triggerShrink = true;
                }
            } else if (validToGrow[k]) {
                if (pc[k] > pcMax[k]) {
                    clustersToGrow[k] = 1;
                    triggerGrow = true;
                }
            }
        }

        py::dict res;
        res["triggerShrinkage"] = triggerShrink;
        res["triggerGrowth"] = triggerGrow;
        res["clustersToShrink"] = py::cast(clustersToShrink);
        res["clustersToGrow"] = py::cast(clustersToGrow);
        return res;
    }
};

// --- PYBIND11 MODULE DEFINITION ---
PYBIND11_MODULE(cluster_cpp, m) {
    py::class_<Cluster>(m, "Cluster")
        .def(py::init<>())
        // Expose data as attributes accessible from Python
        .def_readwrite("nClusters", &Cluster::nClusters)
        .def_readwrite("pc", &Cluster::pc)
        .def_readwrite("clusterSize", &Cluster::clusterSize)
        .def_readwrite("clusterID", &Cluster::clusterID)
        .def_readwrite("availableID", &Cluster::availableID)
        .def_readwrite("trappedStatus", &Cluster::trappedStatus)
        .def_readwrite("connected", &Cluster::connected)
        .def("resize", &Cluster::resize)
        .def("clustering", &Cluster::clustering)
        .def("check_events", &Cluster::checkEvents);
}