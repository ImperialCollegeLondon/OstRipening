#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <iostream>


namespace py = pybind11;

double fast_flux_kernel_cpp(
    double D, double H, double RT, 
    py::array_t<double> flux, py::array_t<double> netFlux, py::array_t<double> len_tij_valid,
    py::array_t<double> gConc0, py::array_t<double> gConc1, py::array_t<float> volarray,
    py::array_t<int> TPValid, py::array_t<int> TValid, py::array_t<int32_t> clusterID,
    py::array_t<double> clust_moles, py::array_t<double> clust_volume, py::array_t<float> clust_pc,
    py::array_t<int32_t> clust_members, py::array_t<int32_t> clust_size,  
    py::array_t<int32_t> members_offset, py::array_t<bool> elemToUpdateW, 
    py::array_t<bool> elemToUpdateNW,
    py::array_t<double> aqMoles, py::array_t<float> minMoles,
    double imposedP, double moles_tol, double _dt) 
{
    // Get pointers to data
    auto p_flux = flux.mutable_data();
    auto p_netFlux = netFlux.mutable_data();
    auto p_len = len_tij_valid.data();
    auto p_g0 = gConc0.data();
    auto p_g1 = gConc1.mutable_data();
    auto p_vol = volarray.data();
    auto p_TPV = TPValid.data();
    auto p_TV = TValid.data();
    auto p_cID = clusterID.data();
    auto p_cMoles = clust_moles.mutable_data();
    auto p_cVol = clust_volume.data();
    auto p_cPc = clust_pc.mutable_data();
    auto p_cMem = clust_members.data();
    auto p_cSize = clust_size.data();
    auto p_cOff = members_offset.data();
    auto p_upW = elemToUpdateW.data();
    auto p_upNW = elemToUpdateNW.data();
    auto p_aqM = aqMoles.mutable_data();
    auto p_minM = minMoles.data();

    int fSize = flux.size();
    int nfSize = netFlux.size();
    int nClusters = clust_moles.size();

    // Compute Fluxes (Parallel)
    #pragma omp parallel for
    for (int i = 0; i < fSize; ++i) {
        int32_t p = p_TPV[i];
        int32_t t = p_TV[i];
        double f = D * p_len[i] * (p_g0[p] - p_g0[t]);

        if (f > 0) {
            if ((p_aqM[p] - (double)p_minM[p]) < moles_tol) f = 0.0;
        } else if (f < 0) {
            if ((p_aqM[t] - (double)p_minM[t]) < moles_tol) f = 0.0;
        }
        
        if (std::abs(f) < 1e-25) f = 0.0;
        p_flux[i] = f;
    }

    // Compute netFlux (Atomic to prevent race conditions)
    // Note: netFlux must be zeroed out in Python before calling this
    #pragma omp parallel for
    for (int i = 0; i < fSize; ++i) {
        double f = p_flux[i];
        int32_t p = p_TPV[i];
        int32_t t = p_TV[i];

        #pragma omp atomic
        p_netFlux[t] += f;

        #pragma omp atomic
        p_netFlux[p] -= f;
    }

    // Determine Adaptive Time Step
    double dt = _dt;
    bool found_neg = false;
    #pragma omp parallel
    {
        double local_dt = _dt;
        bool local_found = false;
        #pragma omp for
        for (int e = 0; e < nfSize; ++e) {
            if (p_netFlux[e] < 0.0) {

                double diff = (double)p_minM[e] - p_aqM[e];
                if (diff < -1e-18) { 
                    double tratio = diff / p_netFlux[e];
                    
                    if (tratio < 0.0) tratio = 0.0; 

                    if (tratio < local_dt) {
                        local_dt = tratio;
                        local_found = true;
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            if (local_found) {
                found_neg = true;
                if (local_dt < dt) {
                    dt = local_dt;
                }
            }
        }
    }
    if (found_neg) {
        if (dt < 0.0) dt = 0.0;
    } else {
        dt = _dt;
    }
    // Update Moles and Concentrations
    double total_deficit = 0.0;
    #pragma omp parallel for
    for (int e = 0; e < nfSize; ++e) {
        double nMoles = p_netFlux[e] * dt;
        if (p_upNW[e]) {
            int32_t k = p_cID[e];
            #pragma omp atomic
            p_cMoles[k] += nMoles;
        } else if (p_upW[e]) {
            p_aqM[e] += nMoles;
            if (p_aqM[e] < 0.0) {
                total_deficit += -p_aqM[e];
                p_aqM[e] = 0.0;
                p_g1[e] = 0.0;
            } else {
            p_g1[e] = p_aqM[e] / (double)p_vol[e];
            }
        }
    }

    // Update Cluster Pressures and Member Concentrations
    #pragma omp parallel for
    for (int k = 0; k < nClusters; ++k) {
        if (p_cSize[k] == 0 || p_cVol[k] == 0.0) continue;
        if (p_cMoles[k] < 0.0){
            total_deficit += -p_cMoles[k];
            p_cMoles[k] = 0.0;
        }
        
        p_cPc[k] = (float)((p_cMoles[k] * RT / p_cVol[k]) - imposedP);
        if (p_cPc[k] < 1e-3f) p_cPc[k] = 1e-3f;

        int32_t st = p_cOff[k];
        double cluster_conc = H * (double)p_cPc[k];
        for (int j = 0; j < p_cSize[k]; ++j) {
            p_g1[p_cMem[st + j]] = cluster_conc;
        }
    }

    if (total_deficit > 0.0){
        for (int e = 0; e < nfSize; ++e) {
            if (p_upW[e] && p_aqM[e]>0) {
                double take = std::min(0.01*p_aqM[e], total_deficit);
                p_aqM[e] -= take;
                p_g1[e] = p_aqM[e] / (double)p_vol[e];
                total_deficit -= take;
                if (total_deficit <= 0.0) break;
            }
        }

        if (total_deficit > 0.0){
            std::cout << "\n  total_deficit:  " << total_deficit << " ";
        }
    }

    return dt;
}

void init_flux_calculations(pybind11::module &m) {
    m.def("compute_fluxes_moles_pc_conc", &fast_flux_kernel_cpp, "High-precision simulation kernel");
}