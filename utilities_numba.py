from numba import njit, prange, jit, types, typeof
from numba.experimental import jitclass
from numba.types import int32, float64, Array, boolean
from numba_specs import *
import numpy as np

import pnflowPy.utilities as do
import cluster as doClust

@njit(updateToDrainImbibe_spec, parallel=True, cache=True)
def updateToDrainImbibe_numba(arrkeys, members, neighbours, toDrainArray, toImbibeArray, 
	pcShrinkArray, pcMaxArray, PcI, PcD, Rarray, totElements, tempMemArray, parallel, pc_min, pc_max):
    ''' updates the toDrain and toImbibe arrays'''

    def _func(i):
        k = arrkeys[i]
        mem = np.where(members[k])[0]
        if mem.size==0:
            toImbibeArray[k] = toDrainArray[k] = totElements+1
            pcShrinkArray[k], pcMaxArray[k] = pc_min, pc_max
            return
        
        tempMemArray[mem] = True
        if mem.size==1:
            pcShrinkArray[k] = max(PcI[mem[0]], 1e-3)
            toImbibeArray[k] = mem[0]
        else:
            imb = mem[np.argmin(Rarray[mem])]
            #imb = mem[np.argmax(PcI[mem])]
            toImbibeArray[k] = imb
            if k==0: pcShrinkArray[k] = 1e-3
            else:
                #pcShrinkArray[k] = max(max(PcI[imb], PcI[mem].max()), 1e-3)
                pcShrinkArray[k] = max(PcI[imb], 1e-3)
                          
        neigh = np.where(neighbours[k])[0]
        if neigh.size==0:
            toDrainArray[k] = totElements+1
            pcMaxArray[k] = pc_max
            return
            
        drain = neigh[np.argmin(PcD[neigh])]
        toDrainArray[k] = drain
        pcMaxArray[k] = PcD[drain]
   
    n = arrkeys.size
    if not parallel:
        for i in range(n):
            _func(i)
    else:
        for i in prange(n):
            _func(i)


@njit(parallel=True, cache=True)
def zeroInvalidFlux(flux, aqueousMoles, minMoles, TPValid, TValid, nValid, moles_tol):
    for i in prange(nValid):
        mask = (flux[i]>0.0)
        pt = TPValid[i]*mask + TValid[i]*(1-mask)
        mask = (aqueousMoles[pt]-minMoles[pt]>=moles_tol)
        flux[i] *= mask


@njit(cache=True)
def updateFluxes(flux, netFlux, netFluxClusters, TPValid, TValid, toUpdateNW, clusterID, nValid, D, len_tij_valid, gasConc0):
    netFlux[:] = 0.0
    netFluxClusters[:] = 0.0
    for i in range(nValid):
        p, t = TPValid[i], TValid[i]
        flux[i] = D*len_tij_valid[i]*(gasConc0[p] - gasConc0[t])
        netFlux[t] += flux[i]
        netFlux[p] -= flux[i]
        if toUpdateNW[t]:
            netFluxClusters[clusterID[t]] += flux[i]
        if toUpdateNW[p]:
            netFluxClusters[clusterID[p]] -= flux[i]


@njit(parallel=True, cache=True)
def compFlux(flux, TPValid, TValid, len_tij_valid, gasConc, D, nValid, aqueousMoles, minMoles, moles_tol):
    for i in prange(nValid):
        p, t = TPValid[i], TValid[i]
        fl = D*len_tij_valid[i]*(gasConc[p] - gasConc[t])
        mask = (fl>0.0 and (aqueousMoles[p]-minMoles[p] >= moles_tol)) or (
            fl<0.0 and (aqueousMoles[t]-minMoles[t] >= moles_tol))
        flux[i] = fl*mask


@njit(fastmath=True, cache=True, parallel=True)
def corner_apex_numba_abridged(arr, halfAng, sinHalfAng, sinHalfAngcosHalfAng, Pc, clusterID, m_cornExists, 
    m_initOrMaxPcHist, advPc, initedApexDist, cornA, sigma, thetaAdvAng, thetaRecAng, 
    areaSPhase, maxCornerArea, nCorners, half_pi):

    n = arr.size
    cornA[arr] = 0.0
    
    for i in prange(n):
        idx = arr[i]
        k = clusterID[idx]
        Pc_i = Pc[clusterID[idx]]
        sigma_over_Pc = sigma / Pc_i
                
        for j in range(nCorners):
            sinHalfAng_ij = sinHalfAng[idx, j]
            if (np.abs(sinHalfAng_ij) < 1e-10) or not m_cornExists[idx, j]:
                continue
            halfAng_ij = halfAng[idx, j]
        
            # cond2
            if Pc_i < advPc[idx, j]:
                conAng_ij = thetaAdvAng[idx]
                term0 = conAng_ij + halfAng_ij
                apexDist_ij = sigma_over_Pc * np.cos(term0)/sinHalfAng_ij
                initedApexDist_ij = initedApexDist[idx, j]

                if apexDist_ij < initedApexDist_ij:
                    part = max(
                        min(initedApexDist_ij * sinHalfAng_ij / sigma_over_Pc, 0.999999), -0.999999)
                    conAng_ij = max(min(np.arccos(part) - halfAng_ij, np.pi), 0.0)
                    apexDist_ij = initedApexDist_ij
            
            # cond3
            elif Pc_i > m_initOrMaxPcHist[idx, j]:
                conAng_ij = min(np.pi, thetaRecAng[idx])
                term0 = conAng_ij + halfAng_ij
                apexDist_ij = sigma_over_Pc*np.cos(term0)/sinHalfAng_ij

            else:
                print("unusual case!!!")
                # from IPython import embed; embed()

            term1 = term0 - half_pi #          
            if abs(term1) < 0.01:
                dimlessCornerA_ij = sinHalfAngcosHalfAng[idx, j]
            else:
                cos_term0 = np.cos(term0) #
                dimlessCornerA_ij = (
                    (sinHalfAng_ij/cos_term0)**2.0 *
                    (np.cos(conAng_ij)*cos_term0/sinHalfAng_ij + term1))

            cornA[idx] += (apexDist_ij**2.0)*dimlessCornerA_ij

        if cornA[idx] > areaSPhase[idx]:
            cornA[idx] = maxCornerArea[idx]
        elif cornA[idx] > maxCornerArea[idx]:
            maxCornerArea[idx] = cornA[idx]

    return cornA[arr]


@njit(computeCornerArea_spec, cache=True)
def computeCornerArea_numba(arr, newPc, cornA, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
    m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, 
    clusterNW_pc, clusterW_ID, clusterNW_ID, areaSPhase, maxCornerArea, sigma, thetaAdvAng, thetaRecAng, 
    _delta, isSquare, isTriangle, muw, MOLECULAR_LENGTH, overideTrapping):

    arrS = arr[isSquare[arr]]
    arrT = arr[isTriangle[arr]]
    
    if arrS.any():
        conAngPS, apexDistPS = do.corner_apex_numba(
            arrS, m_halfAngles, newPc, m_cornExists, m_initOrMaxPcHist, m_initOrMinApexDistHist, 
            m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, clusterNW_pc,
            clusterW_ID, clusterNW_ID, sigma, thetaAdvAng, thetaRecAng, _delta, overideTrapping, 
            MOLECULAR_LENGTH, 4)
        cornA[arrS], _ = do.calcAreaW_numba(
            arrS, m_halfAngles, conAngPS, m_cornExists, apexDistPS, muw, 4)
        
        
    if arrT.any():    
        conAngPT, apexDistPT = do.corner_apex_numba(
            arrT, m_halfAngles, newPc, m_cornExists, m_initOrMaxPcHist, m_initOrMinApexDistHist, 
            m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, clusterNW_pc, 
            clusterW_ID, clusterNW_ID, sigma, thetaAdvAng, thetaRecAng, _delta, overideTrapping, 
            MOLECULAR_LENGTH, 3)
        cornA[arrT], _ = do.calcAreaW_numba(
            arrT, m_halfAngles, conAngPT, m_cornExists, apexDistPT, muw, 3)
        
    cond = cornA > areaSPhase
    if cond.any():
        arr = np.where(cond)[0]
        cornA[arr] = maxCornerArea[arr]

    cond = ~cond & (cornA>maxCornerArea)
    if cond.any():
        arr = np.where(cond)[0]
        maxCornerArea[arr] = cornA[arr]


@njit(computeCornerArea1_spec, cache=True)
def computeCornerArea_numba1(arr, newPc, clusterID, cornA, m_halfAngles, sinHalfAng, sinHalfAngcosHalfAng, 
    m_cornExists, m_initOrMaxPcHist, m_advPc, m_initedApexDist, areaSPhase, maxCornerArea, sigma, 
    thetaAdvAng, thetaRecAng, isSquare, isTriangle, half_pi):

    arrS = arr[isSquare[arr]]
    arrT = arr[isTriangle[arr]]
    
    if arrS.any():
        corner_apex_numba_abridged(arrS, m_halfAngles, sinHalfAng, sinHalfAngcosHalfAng, newPc, clusterID, 
            m_cornExists, m_initOrMaxPcHist, m_advPc, m_initedApexDist, cornA, sigma, thetaAdvAng, thetaRecAng, 
            areaSPhase, maxCornerArea, 4, half_pi)
        
    if arrT.any():
        corner_apex_numba_abridged(arrT, m_halfAngles, sinHalfAng, sinHalfAngcosHalfAng, newPc, clusterID, 
            m_cornExists, m_initOrMaxPcHist, m_advPc, m_initedApexDist, cornA, sigma, thetaAdvAng, thetaRecAng, 
            areaSPhase, maxCornerArea, 3, half_pi)
        

@njit(computeVolume_spec, cache=True, fastmath=True, parallel=True)
def computeVolume_numba(keys, memkeys, memID, newPc, cornA, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
    m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, clusterNW_pc, 
    clusterW_ID, clusterNW_ID, areaSPhase, maxCornerArea, volarray, sigma, thetaAdvAng, thetaRecAng, PcD, 
    tempClustVol, tempMemVol, satList, _delta, isSquare, isTriangle, muw, MOLECULAR_LENGTH, 
    overidetrapping, updateSat):
    
    nElem = memID.size
    computeCornerArea_numba(memID, newPc[clusterNW_ID], cornA, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
        m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, 
        clusterNW_pc, clusterW_ID, clusterNW_ID, areaSPhase, maxCornerArea, sigma, thetaAdvAng, thetaRecAng, 
        _delta, isSquare, isTriangle, muw, MOLECULAR_LENGTH, overidetrapping)
    tempMemVol[memID] = (1-cornA[memID]/areaSPhase[memID])*volarray[memID]

    if updateSat:
        satList[memID] = 1 - tempMemVol[memID]/volarray[memID]

    tempClustVol[keys] = 0.0
    for i in range(nElem):
        tempClustVol[memkeys[i]] += tempMemVol[memID[i]]

    return tempClustVol[keys]


@njit(computeVolume_spec1, cache=True, fastmath=True, parallel=True)
def computeVolume_numba1(keys, memkeys, memID, newPc, cornA, m_halfAngles, sinHalfAng, sinHalfAngcosHalfAng, 
    m_cornExists, m_initOrMaxPcHist, m_advPc, m_initedApexDist, clusterNW_ID,areaSPhase, maxCornerArea, volarray, sigma, 
    thetaAdvAng, thetaRecAng, PcD, criticalVolume, tempClustVol, tempMemVol, satList, isSquare, isTriangle, updateSat, half_pi):
    
    nElem = memID.size
    cond = (newPc[memkeys] <= PcD[memID])
    memID_free = memID[cond]
    tempMemVol[memID_free] = newPc[memkeys[cond]]/PcD[memID_free]*criticalVolume[memID_free]
    memID_left = memID[~cond]

    computeCornerArea_numba1(memID_left, newPc, clusterNW_ID, cornA, m_halfAngles, sinHalfAng, 
        sinHalfAngcosHalfAng, m_cornExists, m_initOrMaxPcHist, m_advPc, m_initedApexDist, areaSPhase, 
        maxCornerArea, sigma, thetaAdvAng, thetaRecAng, isSquare, isTriangle, half_pi)

    tempMemVol[memID_left] = (1-cornA[memID_left]/areaSPhase[memID_left])*volarray[memID_left]

    if updateSat:
        satList[memID] = 1 - tempMemVol[memID]/volarray[memID]

    tempClustVol[keys] = 0.0
    for i in range(nElem):
        tempClustVol[memkeys[i]] += tempMemVol[memID[i]]

    return tempClustVol[keys]


@njit(computeVolume_single_spec, cache=True, fastmath=True)
def computeVolume_single_numba(memID, newPc, cornA, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
    m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, clusterNW_pc, 
    clusterW_ID, clusterNW_ID, areaSPhase, maxCornerArea, volarray, sigma, thetaAdvAng, thetaRecAng, PcD, 
    criticalVolume, tempMemVol, satList, _delta, isSquare, isTriangle, muw, MOLECULAR_LENGTH, 
    overidetrapping, updateSat):

    cond = (newPc[memID] <= PcD[memID])
    memID_free = memID[cond]
    tempMemVol[memID_free] = newPc[memID_free]/PcD[memID_free]*criticalVolume[memID_free]
    memID_left = memID[~cond]

    computeCornerArea_numba(memID_left, newPc, cornA, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
        m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, 
        clusterNW_pc, clusterW_ID, clusterNW_ID, areaSPhase, maxCornerArea, sigma, thetaAdvAng, thetaRecAng, 
        _delta, isSquare, isTriangle, muw, MOLECULAR_LENGTH, overidetrapping)
    tempMemVol[memID_left] = (1-cornA[memID_left]/areaSPhase[memID_left])*volarray[memID_left]

    if updateSat:
        satList[memID] = 1 - tempMemVol[memID]/volarray[memID]

    return tempMemVol[memID].sum()


# @njit(updateVolumeShrinkMax_spec, cache=True)
# def updateVolumeShrinkMax_numba(arrkeys, members, neighbours, toDrainArray, toImbibeArray, 
#         pcShrinkArray, pcMaxArray, valClust, valClustD, clustSize, clusterID, volShrinkArray, volMaxArray,
#         PcI, PcD, maxGasVolume, criticalVolume, totElements, tempMemArray, parallel, pc_min, pc_max):
#     tempMemArray.fill(False)
#     updateToDrainImbibe_numba(arrkeys, members, neighbours, toDrainArray, toImbibeArray, 
#         pcShrinkArray, pcMaxArray, PcI, PcD, totElements, tempMemArray, parallel, pc_min, pc_max)
#     valClust[arrkeys] = (clustSize[arrkeys]>0)
#     valClustD[arrkeys] = valClust[arrkeys] & (toDrainArray[arrkeys] <= totElements)

#     mem = tempMemArray & valClustD[clusterID]
#     nwID = np.flatnonzero(mem)
#     volMaxArray[arrkeys] = np.bincount(clusterID[nwID], maxGasVolume[nwID], arrkeys.max()+1)[arrkeys]

#     mem[toImbibeArray[arrkeys]] = False
#     nwID = np.flatnonzero(mem)
#     volShrinkArray[arrkeys] = np.maximum(
#         1.0e-30, np.bincount(clusterID[nwID], criticalVolume[nwID], arrkeys.max()+1)[arrkeys])


# @njit(updateMolesShrinkGrowth_spec, cache=True)
# def updateMolesShrinkGrowth_numba(arrkeys, pcMaxArray, pcShrinkArray, volMaxArray, volShrinkArray, 
#     molesShrinkArray, molesMaxArray, toImbibeArray, totalVolume, volarray, imposedP, RT, H):
#     molesMaxArray[arrkeys] = ((imposedP+pcMaxArray[arrkeys])*volMaxArray[arrkeys]/RT + 
#                     H*pcMaxArray[arrkeys]*(totalVolume[arrkeys]-volMaxArray[arrkeys]))
#     totalVolume_k = totalVolume[arrkeys] - volarray[toImbibeArray[arrkeys]]
#     molesShrinkArray[arrkeys] = np.maximum(
#             (imposedP + pcShrinkArray[arrkeys])*volShrinkArray[arrkeys]/RT +
#             H*pcShrinkArray[arrkeys]*(totalVolume_k-volShrinkArray[arrkeys]), 1e-22)


# @njit(updateEventsThreshold_spec, cache=True)
# def updateEventsThreshold_shrinkage_numba(arrkeys, members, neighbours, toDrainArray, toImbibeArray, 
#         pcShrinkArray, pcMaxArray, valClust, valClustD, clustSize, clusterID, volShrinkArray, volMaxArray,
#         molesShrinkArray, molesMaxArray, molesArray, totalVolume, volarray, PcI, PcD, maxGasVolume, 
#         criticalVolume, totElements, imposedP, RT, H, tempMemArray, parallel, pc_min, pc_max):

#     oldToImbibe = toImbibeArray[arrkeys]
#     updateVolumeShrinkMax_numba(arrkeys, members, neighbours, toDrainArray, toImbibeArray, 
#         pcShrinkArray, pcMaxArray, valClust, valClustD, clustSize, clusterID, volShrinkArray, volMaxArray,
#         PcI, PcD, maxGasVolume, criticalVolume, totElements, tempMemArray, parallel, pc_min, pc_max)

#     keys1 = arrkeys[(oldToImbibe==toDrainArray[arrkeys])]
#     oldMolesShrink = molesShrinkArray[keys1]
#     updateMolesShrinkGrowth_numba(arrkeys, pcMaxArray, pcShrinkArray, volMaxArray, volShrinkArray, 
#         molesShrinkArray, molesMaxArray, toImbibeArray, totalVolume, volarray, imposedP, RT, H)
#     molesMaxArray[keys1] = np.maximum(molesMaxArray[keys1], oldMolesShrink+1e-22)
#     molesShrinkArray[arrkeys] = np.maximum(np.minimum(molesShrinkArray[arrkeys], molesArray[arrkeys]-1e-22), 1e-22)


# @njit(updateEventsThreshold_spec, cache=True)
# def updateEventsThreshold_growth_numba(arrkeys, members, neighbours, toDrainArray, toImbibeArray, 
#         pcShrinkArray, pcMaxArray, valClust, valClustD, clustSize, clusterID, volShrinkArray, volMaxArray,
#         molesShrinkArray, molesMaxArray, molesArray, totalVolume, volarray, PcI, PcD, maxGasVolume, 
#         criticalVolume, totElements, imposedP, RT, H, tempMemArray, parallel, pc_min, pc_max):

#     oldToDrain = toDrainArray[arrkeys]
#     updateVolumeShrinkMax_numba(arrkeys, members, neighbours, toDrainArray, toImbibeArray, 
#         pcShrinkArray, pcMaxArray, valClust, valClustD, clustSize, clusterID, volShrinkArray, volMaxArray,
#         PcI, PcD, maxGasVolume, criticalVolume, totElements, tempMemArray, parallel, pc_min, pc_max)

#     keys1 = arrkeys[(oldToDrain==toImbibeArray[arrkeys])]
#     oldMolesMax = molesMaxArray[keys1]
#     updateMolesShrinkGrowth_numba(arrkeys, pcMaxArray, pcShrinkArray, volMaxArray, volShrinkArray, 
#         molesShrinkArray, molesMaxArray, toImbibeArray, totalVolume, volarray, imposedP, RT, H)
#     molesMaxArray[keys1] = np.maximum(molesMaxArray[keys1], oldMolesMax+1e-22)
#     molesShrinkArray[keys1] = np.maximum(np.minimum(molesShrinkArray[keys1], oldMolesMax-1e-22), 1e-22)


@njit(returnVolMoles_spec, cache=True, fastmath=True)
def returnVolMoles(keys, memkeys, memID, pc, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
    m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, 
    clusterW_pc, clusterNW_pc, clusterW_ID, clusterNW_ID, areaSPhase, maxCornerArea, 
    volarray, sigma, thetaAdvAng, thetaRecAng, PcD, satList, tempClustVol, tempMemVol, 
    _delta, isSquare, isTriangle, cornA, muw, MOLECULAR_LENGTH, imposedP, RT):
    vol = computeVolume_numba(keys, memkeys, memID, pc, cornA, m_halfAngles,
        m_cornExists, m_initOrMaxPcHist, m_initOrMinApexDistHist, m_advPc, m_recPc, 
        m_initedApexDist, trappedW, trappedNW, clusterW_pc, clusterNW_pc, 
        clusterW_ID, clusterNW_ID, areaSPhase, maxCornerArea, volarray, 
        sigma, thetaAdvAng, thetaRecAng, PcD, tempClustVol, tempMemVol, 
        satList, _delta, isSquare, isTriangle, muw, MOLECULAR_LENGTH, True, False)
    moles = (imposedP+pc[keys])*vol/RT

    return vol, moles


@njit(findPc_spec, cache=True, fastmath=True, parallel=True)
def findPcVolume(keys, memkeys, memID, targetMoles, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
    m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, clusterNW_pc, 
    clusterW_ID, clusterNW_ID, clusterVolume, pcShrink, pcMax, areaSPhase, maxCornerArea, volarray, sigma, 
    thetaAdvAng, thetaRecAng, PcD, satList, tempClustVol, tempMemVol, _delta, isSquare, isTriangle, cornA, muw, 
    MOLECULAR_LENGTH, imposedP, RT, H, moles_tol, pc_tol, max_iter, notdone, pc1, pc2, pcNext):

    pc1[keys] = clusterNW_pc[keys]
    vol1, moles1 = returnVolMoles(keys, memkeys, memID, pc1, m_halfAngles, m_cornExists, 
        m_initOrMaxPcHist, m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, 
        trappedW, trappedNW, clusterW_pc, clusterNW_pc, clusterW_ID, clusterNW_ID, 
        areaSPhase, maxCornerArea, volarray, sigma, thetaAdvAng, thetaRecAng, PcD, satList, 
        tempClustVol, tempMemVol, _delta, isSquare, isTriangle, cornA, muw, MOLECULAR_LENGTH, 
        imposedP, RT)
    f_1 = moles1 - targetMoles[keys]

    notdone_keys = (np.abs(f_1) > moles_tol)
    pc2[keys] = pc1[keys]
    keys_notdone = keys[notdone_keys]
    pc2[keys_notdone] = (pcMax[keys_notdone]+1e-3)*(f_1[notdone_keys]<0)+(
        np.maximum(pcShrink[keys_notdone]-1e-3, 1e-3)*(f_1[notdone_keys]>0))
    vol2, moles2 = returnVolMoles(keys, memkeys, memID, pc2, m_halfAngles, m_cornExists, 
        m_initOrMaxPcHist, m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, 
        trappedW, trappedNW, clusterW_pc, clusterNW_pc, clusterW_ID, clusterNW_ID, 
        areaSPhase, maxCornerArea, volarray, sigma, thetaAdvAng, thetaRecAng, PcD, satList, 
        tempClustVol, tempMemVol, _delta, isSquare, isTriangle, cornA, muw, MOLECULAR_LENGTH, 
        imposedP, RT)
    f_2 = moles2 - targetMoles[keys]

    notdone_keys &= (f_1*f_2<0)
    notdone[keys] = notdone_keys
    keys_notdone = keys[notdone_keys]
    done = (~notdone_keys)
    if done.any():
        keys_done = keys[done]
        clusterNW_pc[keys_done] = pc2[keys_done]
        clusterVolume[keys_done] = vol2[done]
        cond = notdone[memkeys]
        memID_done = memID[~cond]
        satList[memID_done] = cornA[memID_done]/areaSPhase[memID_done]
        memkeys = memkeys[cond]
        memID = memID[cond]
        pcNext[keys_done] = pc2[keys_done]
    
    for _ in range(max_iter):
        pcNext[keys_notdone] = (pc1[keys_notdone]+pc2[keys_notdone])/2
        vol2[notdone_keys], moles2[notdone_keys] = returnVolMoles(
            keys_notdone, memkeys, memID, pcNext, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
            m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, 
            trappedW, trappedNW, clusterW_pc, clusterNW_pc, clusterW_ID, clusterNW_ID, 
            areaSPhase, maxCornerArea, volarray, sigma, thetaAdvAng, thetaRecAng, PcD, satList, 
            tempClustVol, tempMemVol, _delta, isSquare, isTriangle, cornA, muw, MOLECULAR_LENGTH, 
            imposedP, RT)
        f_Next = moles2 - targetMoles[keys]

        cond = notdone_keys & (f_1*f_Next>0)
        pc1[keys_notdone] = np.where(cond[notdone_keys], pcNext[keys_notdone], pc1[keys_notdone])
        pc2[keys_notdone] = np.where(cond[notdone_keys], pc2[keys_notdone], pcNext[keys_notdone])
    
        notdone[keys_notdone] = ((np.abs(f_2[notdone_keys]) > moles_tol)&
            (np.abs(pc1[keys_notdone]-pc2[keys_notdone])>pc_tol))
        notdone_keys = notdone[keys]
        keys_notdone = keys[notdone_keys]
        if not keys_notdone.any(): break
        cond = notdone[memkeys]
        memID_done = memID[~cond]
        satList[memID_done] = cornA[memID_done]/areaSPhase[memID_done]
        memkeys = memkeys[cond]
        memID = memID[cond]

    keys_left = keys[~done]
    clusterNW_pc[keys_left] = pcNext[keys_left]
    clusterVolume[keys_left] = vol2[~done]

    return


@njit(findPc1_spec, cache=True, fastmath=True, parallel=True)
def findPc1(keys, memkeys, memID, targetMoles, m_halfAngles, sinHalfAng, sinHalfAngcosHalfAng, m_cornExists, m_initOrMaxPcHist, 
    m_advPc, m_initedApexDist, clusterVolume, clusterNW_ID, clusterNW_pc, areaSPhase, maxCornerArea, volarray, sigma, thetaAdvAng, thetaRecAng,
    PcD, criticalVolume, satList, tempClustVol, tempMemVol, isSquare, isTriangle, totalVolume, cornA, imposedP, RT, H, 
    moles_tol, pc_tol, max_iter, notdone, pc1, pc2, pcNext, eps, half_pi):

    pc1[keys] = clusterNW_pc[keys]
    vol1 = computeVolume_numba1(keys, memkeys, memID, pc1, cornA, m_halfAngles, sinHalfAng, sinHalfAngcosHalfAng, 
        m_cornExists, m_initOrMaxPcHist, m_advPc, m_initedApexDist, clusterNW_ID, areaSPhase, maxCornerArea, volarray, sigma, 
        thetaAdvAng, thetaRecAng, PcD, criticalVolume, tempClustVol, tempMemVol, satList, isSquare, isTriangle, False, half_pi)

    nkeys = keys.size
    moles1 = ((imposedP+pc1[keys])*vol1/RT + H*pc1[keys]*(totalVolume[keys]-vol1))
    f_1 = moles1 - targetMoles[keys]
    notdone[keys] = (np.abs(f_1) > moles_tol)
    done = ~notdone[keys]
    clusterVolume[keys[done]] = vol1[done]
    keys_notdone = keys[notdone[keys]]
    pc2[keys_notdone] = 1.1*pc1[keys_notdone]
    cond = notdone[memkeys]
    if not cond.all():
        memkeys = memkeys[cond]
        memID = memID[cond]
    
    for _ in range(max_iter):
        vol2 = computeVolume_numba1(keys, memkeys, memID, pc2, cornA, m_halfAngles, sinHalfAng, sinHalfAngcosHalfAng, 
            m_cornExists, m_initOrMaxPcHist, m_advPc, m_initedApexDist, clusterNW_ID, areaSPhase, maxCornerArea, volarray, sigma, 
            thetaAdvAng, thetaRecAng, PcD, criticalVolume, tempClustVol, tempMemVol, satList, isSquare, isTriangle, False, half_pi)
        
        for i in prange(nkeys):
            k = keys[i]
            if not notdone[k]: continue
            moles_k = ((imposedP + pc2[k])*vol2[i]/RT + H*pc2[k]*(totalVolume[k]-vol2[i]))
            f_2_i = moles_k - targetMoles[k]
            if (abs(f_2_i) <= moles_tol) or (np.abs(pc2[k]-pc1[k])<=pc_tol):
                notdone[k] = False
                clusterNW_pc[k] = pc2[k]
                clusterVolume[k] = vol2[i]                
            else:
                denom_i = f_2_i - f_1[i]
                if abs(denom_i) < eps:
                    denom_i = eps if denom_i >= 0.0 else -eps
                delta_k = f_2_i * (pc2[k] - pc1[k]) / denom_i
                pcNext[k] = pc2[k] - max(-0.5*pc2[k], min(delta_k, 0.5*pc2[k]))
                pc1[k], f_1[i] = pc2[k], f_2_i
                pc2[k] = pcNext[k]

        if not notdone[keys].any():
            break

        cond = notdone[memkeys]
        memkeys = memkeys[cond]
        memID = memID[cond]
        
    cond = notdone[keys]
    keys_left = keys[cond]
    clusterNW_pc[keys_left] = pc2[keys_left]
    clusterVolume[keys_left] = vol2[cond]
    
    return

# @njit(findPc_single_spec, cache=True, fastmath=True)
# def findPc_single(key, memID, targetMoles, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
#     m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, clusterNW_pc, 
#     clusterW_ID, clusterNW_ID, clusterVolume, areaSPhase, maxCornerArea, volarray, sigma, thetaAdvAng, thetaRecAng,
#     PcD, criticalVolume, satList, tempMemVol, _delta, isSquare, isTriangle, totalVolume, cornA, muw, 
#     MOLECULAR_LENGTH, imposedP, RT, H, moles_tol, pc_tol, max_iter, pc1, pc2, pcNext, eps):
    
#     pc1[key] = clusterNW_pc[key]
#     vol1 = computeVolume_single_numba(memID, pc1[clusterNW_ID], cornA, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
#         m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, clusterNW_pc, 
#         clusterW_ID, clusterNW_ID, areaSPhase, maxCornerArea, volarray, sigma, thetaAdvAng, thetaRecAng, PcD, 
#         criticalVolume, tempMemVol, satList, _delta, isSquare, isTriangle, muw, MOLECULAR_LENGTH, True, False)
    
#     moles1 = ((imposedP+pc1[key])*vol1/RT + H*pc1[key]*(totalVolume[key]-vol1))
#     f_1 = moles1 - targetMoles[key]
#     if (abs(f_1) <= moles_tol):
#         clusterVolume[key] = vol1
#         return
    
#     pc2[key] = 1.1*pc1[key]
#     for _ in range(max_iter):
#         vol2 = computeVolume_single_numba(memID, pc2[clusterNW_ID], cornA, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
#             m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, clusterNW_pc,
#             clusterW_ID, clusterNW_ID, areaSPhase, maxCornerArea, volarray, sigma, thetaAdvAng, thetaRecAng, PcD, 
#             criticalVolume, tempMemVol, satList, _delta, isSquare, isTriangle, muw, MOLECULAR_LENGTH, True, False)
        
#         moles_k = ((imposedP + pc2[key])*vol2/RT + H*pc2[key]*(totalVolume[key]-vol2))
#         f_2 = moles_k - targetMoles[key]
#         if (abs(f_2) <= moles_tol) or (abs(pc2[key]-pc1[key])<=pc_tol):
#             break
        
#         denom_i = f_2 - f_1
#         if abs(denom_i) < eps:
#             denom_i = np.copysign(eps, denom_i)
#         delta_k = f_2 * (pc2[key] - pc1[key]) / denom_i
#         pcNext[key] = pc2[key] - max(-0.5*pc2[key], min(delta_k, 0.5*pc2[key]))
#         pc1[key], f_1 = pc2[key], f_2
#         pc2[key] = pcNext[key]

#     clusterNW_pc[key] = pc2[key]
#     clusterVolume[key] = vol2
#     return


# @njit(updateClusterPcVolume_spec, cache=True, fastmath=True)
# def updateClusterPcVolume_numba(memID, keys, targetMoles, cornA, m_halfAngles, m_cornExists,
#     m_initOrMaxPcHist, m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, 
#     clusterW_pc, clusterNW_pc, clusterVolume, clusterW_ID, clusterNW_ID, totalVolume, areaSPhase, maxCornerArea, 
#     volarray, sigma, thetaAdvAng, thetaRecAng, PcD, criticalVolume, satList, tempClustVol, tempMemVol, 
#     _delta, isSquare, isTriangle, muw, MOLECULAR_LENGTH, imposedP, RT, H, moles_tol, pc_tol, max_iter, 
#     notdone, pc1, pc2, pcNext, eps):

#     ''' update the cluster pc and volume according to the moles in each cluster '''
#     memkeys = clusterNW_ID[memID]
#     findPc(keys, memkeys, memID, targetMoles, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
#         m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, 
#         clusterW_pc, clusterNW_pc, clusterW_ID, clusterNW_ID, clusterVolume, areaSPhase, maxCornerArea, 
#         volarray, sigma, thetaAdvAng, thetaRecAng, PcD, criticalVolume, atList, tempClustVol, tempMemVol, 
#         _delta, isSquare, isTriangle, totalVolume, cornA, muw, MOLECULAR_LENGTH, imposedP, RT, H, 
#         moles_tol, pc_tol, max_iter, notdone, pc1, pc2, pcNext, eps)


@njit(updateClusterPcVolume1_spec, cache=True, fastmath=True)
def updateClusterPcVolume_numba1(memID, keys, targetMoles, cornA, m_halfAngles, sinHalfAng, sinHalfAngcosHalfAng,
    m_cornExists, m_initOrMaxPcHist, m_advPc, m_initedApexDist, clusterVolume, clusterNW_ID, clusterNW_pc, totalVolume, areaSPhase, maxCornerArea, 
    volarray, sigma, thetaAdvAng, thetaRecAng, PcD, criticalVolume, satList, tempClustVol, tempMemVol, isSquare, isTriangle, 
    imposedP, RT, H, moles_tol, pc_tol, max_iter, notdone, pc1, pc2, pcNext, eps, half_pi):

    ''' update the cluster pc and volume according to the moles in each cluster '''
    memkeys = clusterNW_ID[memID]
    findPc1(keys, memkeys, memID, targetMoles, m_halfAngles, sinHalfAng, sinHalfAngcosHalfAng, m_cornExists, m_initOrMaxPcHist, 
        m_advPc, m_initedApexDist, clusterVolume, clusterNW_ID, clusterNW_pc, areaSPhase, maxCornerArea, volarray, sigma, thetaAdvAng, thetaRecAng,
        PcD, criticalVolume, satList, tempClustVol, tempMemVol, isSquare, isTriangle, totalVolume, cornA, imposedP, RT, H, 
        moles_tol, pc_tol, max_iter, notdone, pc1, pc2, pcNext, eps, half_pi)


# @njit(updateClusterPcVolume_spec, cache=True, fastmath=True, parallel=True)
# def updateClusterPcVolume_numba(memID, keys, targetMoles, cornA, m_halfAngles, m_cornExists, m_initOrMaxPcHist, 
#     m_initOrMinApexDistHist, m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, clusterNW_pc, 
#     clusterVolume, clusterW_ID, clusterNW_ID, totalVolume, areaSPhase, maxCornerArea, volarray, sigma, 
#     thetaAdvAng, thetaRecAng, PcD, criticalVolume, satList, tempClustVol, tempMemVol, _delta, isSquare, isTriangle, 
#     muw, MOLECULAR_LENGTH, imposedP, RT, H, moles_tol, pc_tol, max_iter, notdone, pc1, pc2, pcNext, eps):

#     ''' update the cluster pc and volume according to the moles in each cluster '''
#     memkeys = clusterNW_ID[memID]
#     nkeys = keys.size
#     for i in prange(nkeys):
#         k = keys[i]
#         memID_i = memID[memkeys==k] 
#         findPc_single(k, memID_i, targetMoles, m_halfAngles, m_cornExists, m_initOrMaxPcHist, m_initOrMinApexDistHist, 
#             m_advPc, m_recPc, m_initedApexDist, trappedW, trappedNW, clusterW_pc, clusterNW_pc, clusterW_ID, clusterNW_ID,
#             clusterVolume, areaSPhase, maxCornerArea, volarray, sigma, thetaAdvAng, thetaRecAng, PcD, criticalVolume, satList, 
#             tempMemVol, _delta, isSquare, isTriangle, totalVolume, cornA, muw, MOLECULAR_LENGTH, imposedP, RT, H, moles_tol, 
#             pc_tol, max_iter, pc1, pc2, pcNext, eps)
        

        

    





@njit(fillAdjoiningPores_spec, cache=True)
def fillAdjoiningPores(keys, toDrain, isPore, TPConnections, nPores, done, tempID, hasFluid):
    cond = ~isPore[toDrain]
    toDrainT = toDrain[cond] # throats
    keysT = keys[cond]
    conP = TPConnections[toDrainT-nPores]
    #done = ntwkJit.tempMemP
    done.fill(False)
    #tempID = ntwkJit.tempID
    #hasFluid = clustJit.hasFluid
    for i in range(conP.shape[0]):
        k = keysT[i]
        for j in range(2):
            p = conP[i, j]
            if p > 0 and not hasFluid[p] and not done[p]:
                done[p] = True
                tempID[p] = k
    
    _done = np.flatnonzero(done).astype(np.int32)

    return np.append(keys, tempID[_done]), np.append(toDrain, _done)


@njit(volume_bincount_spec, cache=True)
def volume_bincount(memID, keys, clustID, areaPhase, areaSPhase, volarray):
    return np.bincount(clustID[memID], areaPhase/areaSPhase[memID]*volarray[memID])[keys]


@njit(mapOldNewKeys_spec, cache=True)
def mapOldNewKeys(oldkeys, newkeys, notdone):
    ''' returns unique pairs of old and new keys '''
    n = oldkeys.size
    notdone[newkeys] = True
    cond = np.zeros(n, boolean)
    for i in range(n):
        if notdone[newkeys[i]]:
            cond[i] = True
            notdone[newkeys[i]] = False
            
    return oldkeys[cond], newkeys[cond]
    

