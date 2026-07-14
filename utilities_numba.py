import numpy as np
from scipy.sparse import csr_matrix
import warnings
from functools import reduce
from numba import njit, prange
from numba.types import int32, float32, boolean, void, Tuple, int64, float64
import os
import sys
from math import pi, cos, sin, acos
from time import time_ns


''' Do not call this function in quasi-static mode, its an abridged version
    and overideTrapping condition has been assumed. Updating of values have 
    been modified. '''  
@njit(void(
    int32[:], float32[:], float64, boolean[:], boolean[:], float32[:],
    float32[:], float32[:], float32[:], float32[:], boolean,
    int64, float64, float32[:], float32[:], int32[:]), 
    cache=True, nogil=True)
def create_films_numba(
    mem, halfAng, Pc, m_exists, m_inited, m_initOrMaxPcHist,
    m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, is_oil_inj,
    totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr):

    half_pi = pi/2.0
    sigma_over_Pc = sigma / Pc
    n = mem.size
    for i in range(n):
        m = mem[i]
        nCorners = nCorners_arr[m]
        offset = m * 4 
        advAng_m = thetaAdvAng[m]
        recAng_m = thetaRecAng[m]
        conAng = recAng_m if is_oil_inj else advAng_m
        adv_angle = min(pi, advAng_m)
        rec_angle = min(pi, recAng_m)

        for j in range(nCorners):
            f_idx = offset + j
            if m_exists[f_idx] and m_inited[f_idx]:
                continue

            halfAng_ij = halfAng[f_idx]
            if conAng >= (half_pi - halfAng_ij):
                continue

            m_exists[f_idx] = True
            cosTerm = cos(conAng + halfAng_ij)
            inv_sinTerm = 1.0 / sin(halfAng_ij)
            dist = max(sigma_over_Pc * cosTerm * inv_sinTerm, 0.0)
            m_initedApexDist[f_idx] = dist

            if dist > 0.0:
                common = sigma * inv_sinTerm / dist
                advPc[f_idx] = common * cos(adv_angle + halfAng_ij)
                recPc[f_idx] = common * cos(rec_angle + halfAng_ij)
            else:
                advPc[f_idx] = 0.0
                recPc[f_idx] = 0.0

            m_inited[f_idx] = True
            m_initOrMinApexDistHist[f_idx] = dist
            m_initOrMaxPcHist[f_idx] = Pc
            

@njit(float64(float64, float64, float64), inline='always')
def val_clamped(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)
    

            
''' Do not call this function in quasi-static mode, its an abridged version
    and overideTrapping condition has been assumed '''    
@njit(void(
    int32[:], float32[:], float64, boolean[:], float32[:],
    float32[:], float32[:], float32[:], float32[:], 
    int64, float64, float32[:], float32[:], int32[:],
    float32[:], float32[:], float64, float64, boolean), 
    cache=True, nogil=True)
def corner_apex_numba(
    mem, halfAng, pc_value, m_exists, m_initOrMaxPcHist,
    m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist,
    totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr, 
    conAng_out, apexDist_out, molecular_length, _delta, accurate):

    delta = 0.0 if accurate else _delta
    sigma_over_Pc = sigma / pc_value
    inv_sigma_over_Pc = 1.0/sigma_over_Pc
    n = mem.size
    for i in range(n):
        m = mem[i]
        offset = m * 4
        
        advAng_m = thetaAdvAng[m]
        recAng_m = thetaRecAng[m]
        rec_angle = min(pi, recAng_m)
        nCorners = nCorners_arr[m]
        for j in range(nCorners):
            f_idx = offset + j
            ha_ij = halfAng[f_idx]
            sinHa_ij = sin(ha_ij)
            
            if abs(sinHa_ij) < 1e-10:
                conAng_out[f_idx] = 0.0
                apexDist_out[f_idx] = 0.0
                continue
            
            inv_sinHa_ij = 1.0/sinHa_ij
            conAng_ij = 0.0
            apexDist_ij = 0.0
            advPc_ij = advPc[f_idx]
            recPc_ij = recPc[f_idx]
            iniApxDist_ij = m_initedApexDist[f_idx]

            if not m_exists[f_idx]:
                apexDist_ij = molecular_length
            
            elif (advPc_ij - delta <= pc_value) and (pc_value <= recPc_ij + delta):
                apexDist_ij = iniApxDist_ij
                part = val_clamped(apexDist_ij * sinHa_ij * inv_sigma_over_Pc, -0.999999, 0.999999)
                conAng_ij = val_clamped(acos(part) - ha_ij, 0.0, pi)
                #part = max(-0.999999, min(0.999999, (apexDist_ij * sinHa_ij * inv_sigma_over_Pc)))
                #conAng_ij = max(0.0, min(pi, acos(part) - ha_ij))

            elif pc_value < advPc_ij:
                conAng_ij = advAng_m
                apexDist_ij = sigma_over_Pc * cos(conAng_ij + ha_ij) * inv_sinHa_ij
                if apexDist_ij < iniApxDist_ij:
                    apexDist_ij = iniApxDist_ij
                    part = val_clamped(apexDist_ij * sinHa_ij * inv_sigma_over_Pc, -0.999999, 0.999999)
                    conAng_ij = val_clamped(acos(part) - ha_ij, 0.0, pi)
                    #part = max(-0.999999, min(0.999999, (apexDist_ij * sinHa_ij * inv_sigma_over_Pc)))
                    #conAng_ij = max(0.0, min(pi, acos(part) - ha_ij))

            elif pc_value > m_initOrMaxPcHist[f_idx]:
                conAng_ij = rec_angle
                apexDist_ij = sigma_over_Pc * cos(conAng_ij + ha_ij) * inv_sinHa_ij

            elif pc_value > recPc_ij:
                conAng_ij = recAng_m
                apexDist_ij = sigma_over_Pc * cos(conAng_ij + ha_ij) * inv_sinHa_ij
                minApexHist = m_initOrMinApexDistHist[f_idx]
                
                if apexDist_ij > iniApxDist_ij:
                    apexDist_ij = iniApxDist_ij
                    part = val_clamped(apexDist_ij * sinHa_ij * inv_sigma_over_Pc, -0.999999, 0.999999)
                    conAng_ij = val_clamped(acos(part) - ha_ij, 0.0, pi)
                    #part = max(-0.999999, min(0.999999, (apexDist_ij * sinHa_ij * inv_sigma_over_Pc)))
                    #conAng_ij = max(0.0, min(pi, acos(part) - ha_ij))
                elif apexDist_ij < minApexHist:
                    part = val_clamped(minApexHist * sinHa_ij * inv_sigma_over_Pc, -0.999999, 0.999999)
                    conAng_ij = val_clamped(acos(part) - ha_ij, 0.0, pi)
                    #part = max(-0.999999, min(0.999999, (minApexHist * sinHa_ij * inv_sigma_over_Pc)))
                    #conAng_ij = max(0.0, min(pi, acos(part) - ha_ij))
                    apexDist_ij = minApexHist
            else:
                apexDist_ij = sigma_over_Pc * cos(conAng_ij + ha_ij) * inv_sinHa_ij

            # Final assignment
            conAng_out[f_idx] = conAng_ij
            apexDist_out[f_idx] = apexDist_ij



''' Do not call this function in quasi-static mode, its an abridged version and 
    overideTrapping condition has been assumed. Update of conductance has been 
    removed and only necessary update for Ostwald ripening is ensured.'''   
@njit(void(
    int32[:], float32[:], float32[:], float32[:], boolean[:], int64, 
    float32[:], float32[:], float32[:], boolean, int32[:]), 
    cache=True, nogil=True)
def calcAreaW_numba(
    mem, halfAng, conAng_arr, apexDist_arr, m_exists, totElements, 
    areaSPhase, _cornArea, _centerArea, is_oil_inj, nCorners_arr):

    half_pi = pi / 2.0
    n = mem.size
    for i in range(n):
        m = mem[i]        
        cornA = 0.0
        nCorners = nCorners_arr[m]

        if nCorners > 0:
            offset = m * 4
            for j in range(nCorners):
                f_idx = offset + j
                
                if not m_exists[f_idx]:
                    continue

                halfAng_ij = halfAng[f_idx]
                conAng_ij = conAng_arr[f_idx]
                sin_ha = sin(halfAng_ij)
                cos_ha = cos(halfAng_ij)
                
                term0 = conAng_ij + halfAng_ij
                term1 = term0 - half_pi
                abs_term1 = abs(term1)
                cos_term0 = cos(term0)
                term2 = sin_ha * cos_ha
                term3 = sin_ha/cos_term0

                cornerGstar_ij = term2 / (4.0 * ((1.0 + sin_ha) ** 2))
                if abs_term1 < 0.01:
                    dimlessCornerA_ij = term2
                else:
                    dimlessCornerA_ij = (term3 ** 2) * ((cos(conAng_ij)/term3) + term1)

                if abs_term1 > 0.01:
                    cornerG_ij = dimlessCornerA_ij / (4.0 * ((1.0 - (term3 * term1)) ** 2))
                else:
                    cornerG_ij = cornerGstar_ij

                if cornerG_ij != 0.0:
                    apexDist_ij = apexDist_arr[f_idx]
                    cornA += (apexDist_ij ** 2) * dimlessCornerA_ij


            if cornA > areaSPhase[m]:
                cornA = areaSPhase[m]
            elif  cornA < 0.0:
                cornA = 0.0

            _cornArea[m] = cornA
                
        else:
            _cornArea[m] = 0.0
            cornA = 0.0
            #if is_oil_inj:
            #    _cornArea[m] = 0.0
            #    cornA = 0.0
            #else:
            #    cornA = areaSPhase[m]
            #    _cornArea[m] = cornA
                
        _centerArea[m] = areaSPhase[m] - cornA


@njit(void(
    int32[:], float32[:], float64, boolean[:], boolean[:], 
    float32[:], float32[:], float32[:], float32[:], float32[:], 
    boolean, int64, float64, float32[:], float32[:], int32[:], float32[:],
    float32[:], float32[:], float32[:], float32[:], float64, float64, boolean, boolean
    ), cache=True)
def update_areas_conductances_numba(mem, halfAng, pc_value, m_exists, m_inited, 
    m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist,
    is_oil_inj, totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr, areaSPhase, 
    _cornArea, _centerArea, conAng_cur, apexDist_cur, molecular_length, _delta, 
    accurate, first_time):

    if first_time:
        create_films_numba(mem, halfAng, pc_value, m_exists, m_inited, m_initOrMaxPcHist,
            m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, is_oil_inj,
            totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr)

    corner_apex_numba(
        mem, halfAng, pc_value, m_exists, m_initOrMaxPcHist,
        m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist,
        totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr, 
        conAng_cur, apexDist_cur, molecular_length, _delta, accurate)
    
    calcAreaW_numba(
        mem, halfAng, conAng_cur, apexDist_cur, m_exists, totElements, 
        areaSPhase, _cornArea, _centerArea, is_oil_inj, nCorners_arr)


''' arrr must be true for only element with gas '''
@njit(Tuple((float64, float64))(
    int64, int32[:], float64, float32[:], boolean[:], boolean[:],
    float32[:], float32[:], float32[:], float32[:], float32[:],
    boolean, int64, float64, float32[:], float32[:], int32[:], float32[:],
    float32[:], float32[:], float32[:], float32[:], float64[:], float64[:], int32[:],
    float64, float64, float64, float64, boolean, boolean), cache=True)
def returnClusterVolMoles(k, mem, pc_value, halfAng, m_exists, m_inited, 
        m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist,
        is_oil_inj, totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr, areaSPhase, 
         _cornArea, _centerArea, conAng_cur, apexDist_cur, volarray, satList, clusterID,
        imposedP, molecular_length, _delta, RT, updateSat, first_time):

    update_areas_conductances_numba(mem, halfAng, pc_value, m_exists, m_inited, 
        m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist,
        is_oil_inj, totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr, areaSPhase, 
        _cornArea, _centerArea, conAng_cur, apexDist_cur, molecular_length, _delta, 
        True, first_time)

    cVol = 0.0
    n = mem.size
    #totalVol = 0.0
    for i in range(n):
        m = mem[i]
        sat_i = _centerArea[m]/areaSPhase[m]
        vol_m = volarray[m]
        cVol += sat_i*vol_m
        #totalVol += vol_m
        if updateSat: satList[m] = 1-sat_i
         
    cMoles = (imposedP+pc_value)*cVol/RT

    return cVol, cMoles






