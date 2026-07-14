import numpy as np
from numba import njit, prange
from numba.types import int32, float64

from pnflowPy.cluster import Cluster as clust
import pnflowPy.utilities as do
from pnflowPy.cluster import *

from . import temp

from .utilities_numba import *

import warnings
from numba.core.errors import NumbaDebugInfoWarning
warnings.filterwarnings("ignore", category=NumbaDebugInfoWarning)



_temp = temp.TempArrays()

class Cluster(clust):
    def __init__(self, network, phase=1, nClusters=None): 
        super().__init__(network, phase, nClusters)
        temp.TempArrays.formTempNetworkArrays(
            network.nPores, network.nThroats, network.totElements, 
            network.connectivity_graph_flat.size, network.m_halfAngles)
        self.imbEvents = 0
        self.drainEvents = 0

    def __neighbours__(self, network, keys=None):
        if keys is None:
            keys = np.flatnonzero(self.sizes>0).astype('int32')
        self.updateClusterNeighbours(keys, network)

    def resizeClusters(self, size, ostMode=False, update=False):
        super().resizeClusters(size)
        
        if ostMode:
            oldSize, newSize = self.nClusters-size, self.nClusters

            if self.phase==0: return

            attrList = ['pcShrink', 'pcMax', 'volume', 'volShrink', 'volMax', 'volTotal', 
                        'minPcArray', 'moles', 'molesShrink', 'molesMax', 'netClustMoles', 
                        'netMolesAfterLastUpdate', 'toDrain', 'toImbibe', 'valClust',
                        'valClustD', 'validToShrink', 'validToGrow', 'prequilibratedMoles','pcAfterLastUpdate']
            dtypes = [np.float32, np.float32, np.float64, np.float32, np.float32, np.float64, 
                    np.float32, np.float64, np.float32, np.float32, np.float64, np.float64,
                    np.int32, np.int32, np.bool_, np.bool_, np.bool_, np.bool_, np.float64,
                    np.float32]
            
            for i, attr in enumerate(attrList):
                if not hasattr(self, attr):
                    setattr(self, attr, np.zeros(newSize, dtype=dtypes[i]))
                else:
                    tempArr = getattr(self, attr)
                    newArr = np.zeros(newSize, dtype=dtypes[i])
                    newArr[:oldSize] = tempArr
                    setattr(self, attr, newArr)

            if not update:
                temp.TempArrays.formTempClustArrays(newSize)
            else:
                temp.TempArrays.updateTempClustArrays(newSize)


    def updateClusterProperties(self, keys, ntwk, pc_min, pc_max, shrinkMode):
        return updateClusterProperties_numba(keys, self.sizes, self.volume, 
            self.volShrink, self.volMax, self.volTotal, self.pc, self.pcShrink, self.pcMax,
            self.hasFluid, self.toImbibe, self.toDrain, self.members, 
            self.mem_offsets, self.valClust, self.valClustD, ntwk.totElements, 
            ntwk.connectivity_graph_flat, ntwk.cg_offsets, self.minPcArray, ntwk.PcI, ntwk.PcD, 
            ntwk.entryPress_freezed, ntwk.volarray, pc_min, pc_max, shrinkMode)
            
    def update_areas(self, arrr, pc_value, is_oil_inj, accurate, first_time, ntwk):
        update_areas_conductances_numba(arrr, ntwk.m_halfAngles, pc_value, ntwk.m_cornExists, ntwk.m_inited, 
            ntwk.m_initOrMaxPcHist, ntwk.m_initOrMinApexDistHist, ntwk.m_advPc, ntwk.m_recPc, ntwk.m_initedApexDist,
            is_oil_inj, ntwk.totElements, ntwk.sigma, ntwk.thetaAdvAng, ntwk.thetaRecAng, ntwk.nCorners_arr, ntwk.areaSPhase, 
            ntwk._cornArea, ntwk._centerArea, ntwk.conAng_cur, ntwk.apexDist_cur, ntwk.MOLECULAR_LENGTH, ntwk._delta, 
            accurate, first_time)


    def removeMembers(self, keys, toImbibe, ntwk):
        return removeMembers_numba(keys, toImbibe, self.sizes, self.moles, self.prequilibratedMoles,
            self.hasFluid, self.clusterID, self.pc, self.members, self.mem_offsets, self.heads_arr,
            self.next_elem_arr, self.trapped, self.trappedStatus, self.conn, 
            self.connected, self.neighbours_updated, ntwk.aqueousMoles, ntwk.minMoles, 
            ntwk.gasConc, ntwk.volarray, ntwk.connectivity_graph_flat, ntwk.cg_offsets, 
            ntwk.toInlet, ntwk.toOutlet, ntwk.toInBdr, ntwk.toOutBdr, ntwk.elemToUpdateW, 
            ntwk.elemToUpdateNW,_temp.tmepID, _temp.done, _temp.visited, ntwk.initAqConc)


    def addMembers(self, keys, toAdd, ntwk):
        
        '''Ensure the previous phase is removed from an element first before filling 
            with another phase in the case where the element can only hold one phase
            at a time (circles). Also, allow automatic filling of adjacent pore if 
            not already filled and has lower entry pressure'''
        
      
        updated_keys, updated_toAdd, toRemove_keys, toRemove = addMembers_helper(
            keys, toAdd, self.hasFluid, ntwk.isCircle, ntwk.isPore, 
            ntwk.connectivity_graph_flat, ntwk.cg_offsets, ntwk.PcD, 
            ntwk.elemToUpdateW, ntwk.elemToUpdateNW, ntwk.fluid, self.phase)
                
        if toRemove.size > 0:
            if self.phase==1:
                cP = ntwk.cWP
                for i in toRemove:
                    cP.unfill_phase(i , cP.pc[cP.clusterID[i]])
            else:
                cP = ntwk.cNWP
                cP.removeMembers(toRemove_keys, toRemove, ntwk)

        old_keys = self.clusterID.copy()
        while True:
            status, keys_to_update, new_keys = addMembers_numba(updated_keys, updated_toAdd,
                self.pc[updated_keys], self.sizes, self.moles, self.prequilibratedMoles, 
                self.hasFluid, self.clusterID, self.pc,  self.members, self.mem_offsets, 
                self.heads_arr, self.next_elem_arr, self.trapped, self.trappedStatus, 
                self.conn, self.connected, self.valClust, 
                ntwk.aqueousMoles, ntwk.connectivity_graph_flat, ntwk.cg_offsets, 
                ntwk.toInlet, ntwk.toOutlet, ntwk.toInBdr, ntwk.toOutBdr, ntwk.volarray, _temp.tempID, 
                _temp.done, _temp.visited, self.neighbours_updated, ntwk.initAqConc)
            
            if status==0: break            
            needed = int(np.ceil(ntwk.totElements/5000)*100)
            self.resizeClusters(needed)

        updateMoles_numba(new_keys, old_keys, self.sizes, self.members, self.mem_offsets, self.moles)

        return keys_to_update, new_keys


    def returnClusterPcVolume(self, keys, ntwk, is_oil_inj, moles_tol, pc_tol):
        
        returnClusterPcVolume_numba(keys, ntwk.m_halfAngles, ntwk.m_cornExists,
            ntwk.m_inited, ntwk.m_initOrMaxPcHist, ntwk.m_initOrMinApexDistHist,
            ntwk.m_advPc, ntwk.m_recPc, ntwk.m_initedApexDist, is_oil_inj,
            ntwk.totElements, ntwk.sigma, ntwk.thetaAdvAng, ntwk.thetaRecAng, 
            ntwk.nCorners_arr, ntwk.areaSPhase, ntwk._cornArea, ntwk._centerArea, 
            ntwk.conAng_cur, ntwk.apexDist_cur, ntwk.volarray, ntwk.satList, 
            self.clusterID, self.volume, self.pc, self.pcShrink, self.pcMax, 
            self.moles, self.members, self.mem_offsets, self.sizes, ntwk.gasConc, 
            ntwk.aqueousMoles, ntwk.imposedP, ntwk.MOLECULAR_LENGTH, ntwk._delta, 
            ntwk.H, ntwk.RT, self.minPcArray, ntwk.maxPc, moles_tol, pc_tol)


    def distributeMoles(self, pair, shrinkType, is_oil_inj, moles_tol, ntwk):
        distributeMoles_numba(pair[:,0], pair[:,1], shrinkType, ntwk.m_halfAngles, 
            ntwk.m_cornExists, ntwk.m_inited, ntwk.m_initOrMaxPcHist, 
            ntwk.m_initOrMinApexDistHist, ntwk.m_advPc, ntwk.m_recPc, ntwk.m_initedApexDist, 
            is_oil_inj, ntwk.totElements, ntwk.sigma, ntwk.thetaAdvAng, ntwk.thetaRecAng, 
            ntwk.nCorners_arr, ntwk.areaSPhase, ntwk._cornArea, ntwk._centerArea, 
            ntwk.conAng_cur, ntwk.apexDist_cur, ntwk.volarray, ntwk.satList, self.clusterID, 
            self.pcShrink, self.pcMax, self.moles, self.members, self.mem_offsets, 
            self.sizes, ntwk.imposedP, ntwk.MOLECULAR_LENGTH, ntwk._delta, ntwk.RT, moles_tol)
            
    def adjustClusterPcVolume(self, ntwk, moles_tol, pc_tol):
        adjustClusterPcVolume_numba(self.netMolesAfterLastUpdate,
            ntwk.m_halfAngles, ntwk.m_cornExists, ntwk.m_inited, ntwk.m_initOrMaxPcHist, 
            ntwk.m_initOrMinApexDistHist, ntwk.m_advPc, ntwk.m_recPc, ntwk.m_initedApexDist, 
            ntwk.totElements, ntwk.sigma, ntwk.thetaAdvAng, ntwk.thetaRecAng, 
            ntwk.nCorners_arr, ntwk.areaSPhase, ntwk._cornArea, ntwk._centerArea, 
            ntwk.conAng_cur, ntwk.apexDist_cur, ntwk.volarray, ntwk.satList,
            self.clusterID, self.volume, self.pc, self.pcShrink, self.pcMax, self.moles, 
            self.members, self.mem_offsets, self.sizes, _temp.tempCID, ntwk.gasConc, ntwk.aqueousMoles,
            ntwk.imposedP, ntwk.MOLECULAR_LENGTH, ntwk._delta, ntwk.H, ntwk.RT, self.minPcArray, 
            ntwk.maxPc, moles_tol, pc_tol)
            
        
    def shrinkCluster(self, keys, ntwk, moles_tol=1e-20, pc_tol=1e-4, 
                    max_iter=50, pc_min=1e-3, pc_max=1e30):

        ''' This function manages the shrinkage of clusters which could lead to 
            reduction in size, fragmentation of clusters or total disappearance. 
            This function has assumed the shrinkage of phase 1  !!!'''
            
        toImbibe = self.toImbibe[keys]
        
        # fill toImbibe with phase 0
        cWP = ntwk.cWP
        ntwk.fluid[toImbibe] = 0
        cWP.hasFluid[toImbibe] = True
        old_clusterID = self.clusterID.copy()
        _temp.tempID[:] = -5

        cWP.doClusteringAssigning(toImbibe, self.pc[keys], _temp.tempID, True, ntwk)
        ntwk.satList[toImbibe] = 1.0

        # remove phase 1 from toImbibe
        _temp.tempID[:] = -5
        status, keys_to_update = removeMembers_numba(keys, toImbibe, self.sizes, self.moles, 
            self.prequilibratedMoles, self.hasFluid, self.clusterID, self.pc, 
            self.members, self.mem_offsets, self.heads_arr, self.next_elem_arr, 
            self.trapped, self.trappedStatus, self.conn, self.connected, 
            self.neighbours_updated, ntwk.aqueousMoles, ntwk.minMoles, ntwk.gasConc, 
            ntwk.volarray, ntwk.connectivity_graph_flat, ntwk.cg_offsets, 
            ntwk.toInlet, ntwk.toOutlet, ntwk.toInBdr, ntwk.toOutBdr, 
            ntwk.elemToUpdateW, ntwk.elemToUpdateNW, 
            _temp.tempID, _temp.done, _temp.visited, ntwk.initAqConc)
        
   
        # update properties of clusters including the neighbours, toImbibe, toDrain
        mem = self.updateClusterProperties(keys_to_update, ntwk, pc_min, pc_max, True)
        self.netMolesAfterLastUpdate[keys_to_update] = 0.0
                    
        if mem.size > 0:
            pair, shrinkType = mapOldNewKeys(mem, old_clusterID, self.clusterID, self.nClusters)
            newkeys = pair[:, 1]
            self.distributeMoles(pair, shrinkType, False, moles_tol, ntwk)
            self.returnClusterPcVolume(newkeys, ntwk, False, moles_tol, pc_tol)
            
     

    def growCluster(self, keys, ntwk, moles_tol=1e-23, pc_tol=1e-4, max_iter=50, 
                    pc_min=1.0e-3, pc_max=1.0e30):
        ''' update toDrain element(s) !!! '''
        
        toDrain = self.toDrain[keys]
        ntwk.fluid[toDrain] = 1 
        keys_to_update, new_keys = self.addMembers(keys, toDrain, ntwk)        
        _ = self.updateClusterProperties(keys_to_update, ntwk, pc_min, pc_max, False)
        self.netMolesAfterLastUpdate[keys_to_update] = 0.0
        self.returnClusterPcVolume(new_keys, ntwk, True, moles_tol, pc_tol)
        
        nkeys = toDrain.size
        for i in range(nkeys):
            m = toDrain[i]
            k = self.clusterID[m]            
            if self.pc[k] <= self.pcShrink[k] or np.isclose(self.pc[k], pc_min):
                ntwk.entryPress_freezed[m] = True
                self.pcShrink[k] = pc_min
                
           
           

@njit(int32[:](int32[:], int32[:], boolean[:], int32[:], int32[:], int32[:]), cache=True)
def checkCoalescence(keys, toDrain, hasFluid, clusterID, cg_data, cg_offsets):
    coalesced_keys = np.empty(keys.size*10, dtype=np.int32)
    j = 0
    for i in range(keys.size):
        k = keys[i]
        e = toDrain[i]
        for n in cg_data[cg_offsets[e]:cg_offsets[e+1]]:
            if hasFluid[n] and clusterID[n] != k:
                coalesced_keys[j] = k
                j += 1
    return np.unique(coalesced_keys[:j])
    
    
@njit((boolean[:], float64[:], float64[:], int32[:], int32[:], int32[:], float64[:]), parallel=True, cache=True)
def assignSaturation_numba(isCircle, satList, clust_volume, clust_sizes, members, members_offsets, volarray):
    nClusters = clust_volume.size
    for k in prange(nClusters):
        size_k = clust_sizes[k]
        if size_k==0: continue
        target_vol = clust_volume[k]
        volTotal = 0.0
        j = 0
        mem = np.empty(size_k, dtype=np.int32)

        for m in members[members_offsets[k]:members_offsets[k+1]]:
            if isCircle[m]:
                target_vol -= volarray[m]
                satList[m] = 0.0
            else:
                mem[j] = m
                volTotal += volarray[m]
                j += 1
                
        mem = mem[:j]
        sat_w = max(0.0, 1.0 - target_vol/volTotal)
        for m in mem:
            satList[m] = sat_w     
    


@njit(Tuple((int32[:,:], boolean[:]))(int32[:], int32[:], int32[:], int64), cache=True)
def mapOldNewKeys(mem, oldIDS, newIDS, nClusters):
    ''' returns unique pairs of old and new keys '''
    notdone = np.ones(nClusters, dtype=np.bool_)
    n = mem.size
    pair = np.empty((n, 2), dtype=np.int32)
    first_new = np.full(nClusters, -5, dtype=np.int32)
    shrinkType = np.zeros(nClusters, dtype=np.bool_)  # True if cleavage occurred otherwise False
    j = 0
    for i in range(n):
        m = mem[i]
        old_id = oldIDS[m] 
        new_id = newIDS[m]
        fn = first_new[old_id]

        if fn == -5:
            first_new[old_id] = new_id
            pair[j, 0] = old_id
            pair[j, 1] = new_id
            notdone[new_id] = False
            j += 1

        elif notdone[new_id]:
            shrinkType[old_id] = True
            pair[j, 0] = old_id
            pair[j, 1] = new_id
            notdone[new_id] = False
            j += 1

    return pair[:j], shrinkType


@njit(void(boolean[:], int32[:], float64[:], float64[:], float64[:], int64), cache=True)
def updateClusterVolume(hasFluid, clusterID, clusterVol, satList, volarray, totElements):
    for i in range(totElements):
        if not hasFluid[i]: continue
        cid = clusterID[i]
        clusterVol[cid] += (1-satList[i])*volarray[i]

               
@njit(float64(int32[:], float32[:], float64[:], float64[:], 
    boolean[:], boolean[:], float32[:], float64, float64, float64), cache=True)
def adjustPcUpdateMolesReturnAvgPc(keys, clusterPc, clusterVol, clusterMoles, 
    valClust, valClustD, pcMax, imposedP, RT, alpha):
    nkeys = keys.size
    pcVolSum = 0.0
    volSum = 0.0
    for i in range(nkeys):
        k = keys[i]
        if not valClust[k]: continue
        if valClustD[k]: 
            clusterPc[k] = alpha*pcMax[k] + (1-alpha)*clusterPc[k]
            pcVolSum += (clusterPc[k]*clusterVol[k])
            volSum += clusterVol[k]

        clusterMoles[k] = (imposedP + clusterPc[k])*clusterVol[k]/RT

    return pcVolSum/volSum


@njit(void(boolean[:], boolean[:], boolean[:], float32[:], float32[:],
    int32[:], float32[:], float64[:], float64[:], float64[:], boolean[:], boolean[:], 
    boolean[:], float64[:], float64[:], float64, float64, float64, float64, int64), 
    parallel=True, cache=True)
def updateArrays(hasFluid, validToShrink, validToGrow, pcShrink, pcMax, 
    clusterID, clusterPc, clusterMoles, volarray, satList, elemToUpdateW, elemToUpdateNW, 
    validVolumes, gasConc, aqueousMoles, initialAvgPres, initAqConc, imposedP, H, totElements):

    for i in prange(totElements):
        if volarray[i]==0.0: continue
        
        validVolumes[i] = True
        if hasFluid[i]: 
            elemToUpdateNW[i] = True
            cid = clusterID[i]
            gasConc[i] = H*(clusterPc[cid] + imposedP)
        else: 
            elemToUpdateW[i] = True
            gasConc[i] = initAqConc

        aqueousMoles[i] = gasConc[i]*volarray[i]


@njit(void(
    int32[:], int32[:], boolean[:], float32[:], boolean[:], boolean[:],
    float32[:], float32[:], float32[:], float32[:], float32[:],
    boolean, int64, float64, float32[:], float32[:], int32[:], float32[:],
    float32[:], float32[:], float32[:], float32[:], float64[:], float64[:], int32[:],
    float32[:], float32[:], float64[:], int32[:], int32[:], int32[:], float64,
    float64, float64, float64, float64), parallel=True, cache=True)
def distributeMoles_numba(old_keys, new_keys, shrinkType, halfAng, m_exists, m_inited, 
    m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, 
    is_oil_inj, totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr, areaSPhase, 
    _cornArea, _centerArea, conAng_cur, apexDist_cur, volarray, satList, clusterID, 
    pcShrinkArr, pcMaxArr, molesArr, members, mem_offsets, sizes, imposedP, 
    molecular_length, _delta, RT, moles_tol):

    uniq_old_keys = np.unique(old_keys)
    nUniq = uniq_old_keys.size
    nkeys = old_keys.size
    Pc_values = np.zeros(totElements, dtype=np.float64)
    accurate = False

    prev_moles = np.empty(nUniq, dtype=np.float64)
    for i in range(nUniq):
        k = uniq_old_keys[i]
        prev_moles[i] = molesArr[k]

    for i in prange(nUniq):
        k = uniq_old_keys[i]
        if shrinkType[k]:
            new_keys_k = np.empty(nkeys, dtype=np.int32)
        else:
            new_keys_k = np.empty(1, dtype=np.int32)

        j = 0
        for n in range(nkeys):
            if old_keys[n] == k:
                new_keys_k[j] = new_keys[n]
                j += 1
        
        if shrinkType[k]:
            new_keys_k = new_keys_k[:j]
            nkSize = new_keys_k.size
            molesShr = np.empty(nkSize, dtype=np.float64)
            molesMax = np.empty(nkSize, dtype=np.float64)
            totalVol = np.empty(nkSize, dtype=np.float64)
            
            for ii in prange(nkSize):
                nk = new_keys_k[ii]
               
                start, end = mem_offsets[nk], mem_offsets[nk+1]
                mem = members[start:end]
                first_time = True

                _, molesShr[ii] = returnClusterVolMoles(nk, mem, pcShrinkArr[nk], halfAng, m_exists, 
                    m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc, 
                    m_initedApexDist, is_oil_inj, totElements, sigma, thetaAdvAng, thetaRecAng, 
                    nCorners_arr, areaSPhase, _cornArea, _centerArea, conAng_cur, apexDist_cur, 
                    volarray, satList, clusterID, imposedP, molecular_length, _delta, RT, accurate, first_time)

                first_time = False
                _, molesMax[ii] = returnClusterVolMoles(nk, mem, pcMaxArr[nk], halfAng, m_exists, 
                    m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc, 
                    m_initedApexDist, is_oil_inj, totElements, sigma, thetaAdvAng, thetaRecAng, 
                    nCorners_arr, areaSPhase, _cornArea, _centerArea, conAng_cur, apexDist_cur, 
                    volarray, satList, clusterID, imposedP, molecular_length, _delta, RT, accurate, first_time)
                
                totalVol[ii] = volarray[mem].sum()

            molesShrCum = molesShr.sum()
            diff = molesMax - molesShr            
            if (molesShr <= moles_tol).any():
                molesArr[new_keys_k] = totalVol/totalVol.sum()*prev_moles[i]                
            elif (prev_moles[i] - molesShrCum >= 0.0) and (diff>0.0).all():                
                molesArr[new_keys_k] = molesShr + diff/diff.sum()*(prev_moles[i] - molesShrCum)
            else:
                molesArr[new_keys_k] = molesShr/molesShrCum*prev_moles[i]

            if sizes[k]==0:
                molesArr[k] = 0.0

        elif new_keys_k[0] != k:
            molesArr[new_keys_k] = prev_moles[i]
            molesArr[k] = 0.0



@njit(Tuple((int32, int32[:]))(
    int32[:], int32[:], int32[:], float64[:], float64[:], boolean[:], int32[:], float32[:],
    int32[:], int32[:], int32[:], int32[:], boolean[:], boolean[:], boolean[:], 
    boolean[:], boolean[:], float64[:], float64[:], float64[:], float64[:],
    int32[:], int32[:], boolean[:], boolean[:], boolean[:], boolean[:], boolean[:], boolean[:], 
    int32[:], boolean[:], int32[:], float64), cache=True)
def removeMembers_numba(keys, toRemove, sizes, moles, prequilibratedMoles, hasFluid, clusterID, clusterPc, 
    members, mem_offsets, heads_arr, next_elem_arr, trapped, trappedStatus, conn, 
    connected, neighbours_updated, aqueousMoles, minMoles, gasConc, volarray,
    cg_data, cg_offsets, toInlet, toOutlet, toInBdr, toOutBdr, etoUpdateW, etoUpdateNW, 
    temp_clusterID, done, visited, initAqConc):
    ''' remove members from clusters and update clusters '''

    nkeys = keys.size
    notAdded = np.ones(sizes.size, dtype=np.bool_)
    to_update = np.full(sizes.size, -5, dtype=np.int32)
    t = 0
    elem_arr = np.zeros(hasFluid.size, dtype=np.int32)
    for i in range(nkeys):
        k, toImb = keys[i], toRemove[i]
        if notAdded[k]:
            notAdded[k] = False
            to_update[t] = k
            t += 1
        if sizes[k]==1:
            aqueousMoles[toImb] = moles[k] #+ prequilibratedMoles[k]
            moles[k] = 0.0
        else:
            aqueousMoles[toImb] = min(moles[k], initAqConc*volarray[toImb])
            moles[k] -= aqueousMoles[toImb]
            
        gasConc[toImb] = aqueousMoles[toImb]/volarray[toImb]
        
        sizes[k] -= 1
        hasFluid[toImb] = False
        clusterID[toImb] = -5
        trapped[toImb] = False
        etoUpdateW[toImb], etoUpdateNW[toImb] = True, False
        

        start, end = cg_offsets[toImb], cg_offsets[toImb+1]
        neigh = np.empty(end-start, dtype=np.int32)
        j = 0
        for idx in range(start, end):
            n = cg_data[idx]
            if hasFluid[n]:
                neigh[j] = n
                j += 1
        neigh = neigh[:j]
            
        if j>0:
            Pc = clusterPc[k]
            start_index = 0
            elem_offset = np.zeros(neigh.size+1, dtype=np.int32)
            is_connected_arr = np.zeros(j, dtype=np.bool_)
            is_trapped_arr = np.ones(j, dtype=np.bool_)
            status, start_index = parallel_exploration_dsu_with_assigning(neigh,  np.asarray([Pc]), 
                elem_arr, elem_offset, hasFluid, cg_data, cg_offsets, sizes, clusterID, clusterPc,
                temp_clusterID, toInlet, toOutlet, toInBdr, toOutBdr, done, visited, 
                members, mem_offsets, heads_arr, next_elem_arr, trapped, trappedStatus, 
                conn,  connected, neighbours_updated, is_connected_arr, is_trapped_arr, True, start_index)

            for n in neigh:
                k = clusterID[n]
                if notAdded[k]:
                    notAdded[k] = False
                    to_update[t] = k
                    t += 1
                clusterPc[k] = Pc            
                if not trappedStatus[k]:
                    clusterPc[k] = Pc
        else:   # on the assumption that only a single-membered cluster could satisfy this condition after shrinkage
            heads_arr[k] = -5
            members[mem_offsets[k]:-1] = members[mem_offsets[k+1]:]
            mem_offsets[k+1:] -= 1
            
    return status, to_update[:t]


@njit(Tuple((int32[:], int32[:], int32[:], int32[:]))(
    int32[:], int32[:], boolean[:], boolean[:], boolean[:], int32[:], int32[:],
    float32[:], boolean[:], boolean[:], int32[:], int64),cache=True)
def addMembers_helper(keys, toAdd, hasFluid, isCircle, isPore, cg_data, cg_offsets, 
    PcD, etoUpdateW, etoUpdateNW, fluidArr, phase):
    nkeys = keys.size
    arr_size = nkeys*2

    toRemove = np.empty(arr_size, dtype=np.int32)
    toRemove_keys = np.empty(arr_size, dtype=np.int32)
    updated_keys = np.empty(arr_size, dtype=np.int32)
    updated_toAdd = np.empty(arr_size, dtype=np.int32)

    j, p = 0, 0
    for i in range(nkeys):
        k, toDra = keys[i], toAdd[i]
        if hasFluid[toDra]: continue
        hasFluid[toDra] = True
        etoUpdateW[toDra], etoUpdateNW[toDra] = False, True
        updated_keys[j] = k
        updated_toAdd[j] = toDra
        j += 1
        if isCircle[toDra]:
            toRemove[p] = toDra
            toRemove_keys[p] = k
            p += 1
        if not isPore[toDra]:
            start, end = cg_offsets[toDra], cg_offsets[toDra+1]
            for n in cg_data[start:end]:
                if not hasFluid[n] and PcD[n] <= PcD[toDra]:
                    hasFluid[n] = True
                    updated_keys[j] = k
                    updated_toAdd[j] = n
                    fluidArr[n] = phase
                    j += 1
                    etoUpdateW[n], etoUpdateNW[n] = etoUpdateNW[n], etoUpdateW[n]
                    if isCircle[n]:
                        toRemove[p] = n
                        toRemove_keys[p] = k
                        p += 1   

    return updated_keys[:j], updated_toAdd[:j], toRemove_keys[:p], toRemove[:p]  


@njit(Tuple((int32, int32[:], int32[:]))(
    int32[:], int32[:], float32[:], int32[:], float64[:], float64[:], boolean[:], int32[:], float32[:],
    int32[:], int32[:], int32[:], int32[:], boolean[:], boolean[:], boolean[:], 
    boolean[:], boolean[:], float64[:], int32[:], int32[:], boolean[:], boolean[:], boolean[:], 
    boolean[:], float64[:], int32[:], boolean[:], int32[:], boolean[:], float64), cache=True)
def addMembers_numba(keys, toAdd, pc_val, sizes, moles, prequilibratedMoles, hasFluid, clusterID, clusterPc,
    members, mem_offsets, heads_arr, next_elem_arr, trapped, trappedStatus, conn, 
    connected, valClust, aqueousMoles, cg_data, cg_offsets, toInlet, toOutlet, toInBdr, 
    toOutBdr, volarray, temp_clusterID, done, visited, neighbours_updated, initAqConc):
    ''' add members to clusters and update clusters '''
    
    nkeys = keys.size
    nClusters = sizes.size
    notdone = np.ones(nClusters, dtype=np.bool_)
    keys_to_update = np.empty(nkeys*5, dtype=np.int32)
    j = 0
    for i in range(nkeys):
        k, toDra = keys[i], toAdd[i]
        equilMoles = 0.0
        moles[k] += (aqueousMoles[toDra]-equilMoles)
        if notdone[k]:
            keys_to_update[j] = k
            notdone[k] = False
            j += 1
        for m in cg_data[cg_offsets[toDra]:cg_offsets[toDra+1]]:
            k = clusterID[m]
            if k>=0 and notdone[k]:
                keys_to_update[j] = k
                notdone[k] = False
                j += 1
    
    temp_clusterID[:] = -5
    start_index = 0
    elem_arr = np.zeros(hasFluid.size, dtype=np.int32)
    elem_offset = np.zeros(nkeys+1, dtype=np.int32)
    is_connected_arr = np.zeros(nkeys, dtype=np.bool_)
    is_trapped_arr = np.ones(nkeys, dtype=np.bool_)
    status, start_index = parallel_exploration_dsu_with_assigning(toAdd, pc_val, 
        elem_arr, elem_offset, hasFluid, cg_data, cg_offsets, sizes, clusterID, clusterPc, 
        temp_clusterID, toInlet, toOutlet, toInBdr, toOutBdr, done, visited, members, mem_offsets, 
        heads_arr, next_elem_arr, trapped, trappedStatus, conn, connected, neighbours_updated, 
        is_connected_arr, is_trapped_arr, True, start_index)
   
        
    for i in range(nkeys):
        curr = toAdd[i]
        k = clusterID[curr]
        if notdone[k]:
            keys_to_update[j] = k
            notdone[k] = False
            j += 1
        
    for i in range(j):
        notdone[keys_to_update[i]] = True
    
    new_keys = np.full(j, -5, dtype=np.int32)
    n = 0
    for i in range(new_keys.size):
        k = keys_to_update[i]
        if sizes[k]>0 and notdone[k]:
            new_keys[n] = k
            notdone[k] = False
            n += 1
    
    return status, keys_to_update[:j], new_keys[:n]


@njit(void(int32[:], int32[:], int32[:], int32[:], int32[:], float64[:]), cache=True)
def updateMoles_numba(keys, old_keys, sizes, members, mem_offsets, molesArr):
    nkeys = keys.size
    done  = np.zeros(sizes.size, dtype=np.bool_)
    old_moles = molesArr.copy()
    for i in range(nkeys):
        k = keys[i]
        for idx in range(mem_offsets[k], mem_offsets[k+1]):
            m = members[idx]
            old_id = old_keys[m]
            if old_id == -5 or k==old_id: continue
            if done[old_id]: continue
            done[old_id] = True
            molesArr[k] += old_moles[old_id]
            molesArr[old_id] -= old_moles[old_id]




@njit(void(
    int32[:], int32[:], float32[:], boolean[:], float64[:], float64[:], int64), 
    cache=True)
def updateVolumeFromSat(keys, clusterID, clustVolume, hasFluid, satList, volarray, totElements):
    ''' This function is written for phase 1. satList is saturation of phase 0'''

    nkeys = keys.size
    for i in range(totElements):
        if not hasFluid[i]: continue
        cid = clusterID[i]
        clustVolume[cid] += (1-satList[i])*volarray[i]


@njit(void(
    int64, int32[:], int32[:], int32[:], int32[:],
    float32[:], float32[:], float32[:], float64[:], float32[:], float32[:], boolean[:], 
    float64[:], int64, float64, float64), cache=True)
def updateToDrainImbibe_single(k, mem, neigh, toDrainArray, toImbibeArray, 
    pcShrinkArray, pcMaxArray, minPcArray, volTotal, PcI, PcD, entryPress_freezed, 
    volarray, totElements, pc_min, pc_max):

    volTotal[k] = 0.0
    if mem.size==0:
        toImbibeArray[k] = toDrainArray[k] = totElements+1
        pcShrinkArray[k], pcMaxArray[k] = pc_min, pc_max
        return

    mem_size, neigh_size = mem.size, neigh.size
    toimb = mem[0]
    max_pc = PcI[toimb]
    min_pc = max_pc
    volTotal[k] = volarray[toimb]
    if mem.size==1:
        freezed = entryPress_freezed[toimb]
        if not freezed:
            pcShrinkArray[k] = max(PcI[toimb], 1e-3)
            toImbibeArray[k] = toimb
    else:
        freezed = False
        for i in range(1, mem_size):
            next_m = mem[i]
            volTotal[k] += volarray[next_m]
            next_pc = PcI[next_m]
            if entryPress_freezed[next_m]: freezed=True
            if max_pc < next_pc:
                max_pc = next_pc
                toimb = next_m
            if next_pc < min_pc:
                min_pc = next_pc
                
                
        if not freezed:
            toImbibeArray[k] = toimb
            pcShrinkArray[k] = max_pc
    
    minPcArray[k] = max(min_pc, 1e-3)        
    if neigh.size==0:
        toDrainArray[k] = totElements+1
        pcMaxArray[k] = pc_max
        return
        
    todra = neigh[0]
    min_pc = PcD[todra]
    for i in range(1, neigh_size):
        next_n = neigh[i]
        next_pc = PcD[next_n]
        if min_pc > next_pc:
            min_pc = next_pc
            todra = next_n

    if not freezed:
        toDrainArray[k] = todra
        pcMaxArray[k] = min_pc



@njit(int32[:](
    int32[:], int32[:], float64[:], float32[:], float32[:], float64[:],
    float32[:], float32[:], float32[:], boolean[:], int32[:], int32[:], 
    int32[:], int32[:], boolean[:], boolean[:], int64, int32[:], int32[:], 
    float32[:], float32[:], float32[:], boolean[:], float64[:], float64, float64, boolean), cache=True)
def updateClusterProperties_numba(keys, sizes, clusterVol, volShrink, volMax, volTotal,
    clusterPc, pcShrink, pcMax, hasFluid, toImbibeArr, toDrainArr,
    members, mem_offsets, valClust, valClustD, totElements, cg_data, cg_offsets, 
    minPcArray, PcI, PcD, entryPress_freezed, volarray, pc_min, pc_max, shrinkMode):
    
    nkeys = keys.size
    if shrinkMode:
        all_members = np.empty(totElements, dtype=np.int32)
    else:
        all_members = np.empty(0, dtype=np.int32)
    j = 0 

    valClust[:] = (sizes>0)
    valClustD &= valClust
    for i in range(nkeys):
        k = keys[i]
        if sizes[k]==0:
            volShrink[k] = volMax[k] = clusterVol[k] = 0.0
            clusterPc[k], pcShrink[k], pcMax[k] = 0.0, pc_min, pc_max
            toImbibeArr[k] = toDrainArr[k] = totElements+1
            valClust[k] = valClustD[k] = False
        else:
            start, end = mem_offsets[k], mem_offsets[k+1]
            mem = members[start:end]
            if shrinkMode:
                for m in mem:
                    all_members[j] = m
                    j += 1

            neigh = getNeighbours(k, mem, hasFluid, cg_data, cg_offsets, totElements)
            valClust[k] = True

            updateToDrainImbibe_single(k, mem, neigh, toDrainArr, toImbibeArr, 
                pcShrink, pcMax, minPcArray, volTotal, PcI, PcD, entryPress_freezed, 
                volarray, totElements, pc_min, pc_max)
           
            if neigh.size>0: 
                valClustD[k] = True

    return all_members[:j]



@njit(Tuple((int64, int64))(
    int32[:], float32[:], float32[:], float32[:], boolean[:],
    boolean[:], boolean[:], int32[:], int32[:], float64), cache=True)
def checkEvents(sizes, clusterPc, pcShrink, pcMax, validToShrink,
    validToGrow, valClust, toShrinkList, toGrowList, initialAvgPres):

    nClusters = validToShrink.size
    i, j = 0, 0
    for k in range(nClusters):
        if not valClust[k]: continue

        if validToGrow[k] and clusterPc[k] > pcMax[k]:
            toGrowList[j] = k
            j += 1
        elif validToShrink[k] and clusterPc[k] < pcShrink[k]:
            toShrinkList[i] = k
            i += 1

    return i, j


@njit(float64(
    float64[:], float64[:], float32[:], float64[:], float64[:],
    int32[:], int32[:], int32[:], float64[:], float64[:], float32[:], int32[:], float64[:],
    float64[:], int32[:], int32[:], boolean[:], boolean[:], 
    float64[:], float64[:], float32[:], float64, boolean[:], float64, float64, float64, float64, float64, float64), 
    fastmath=True, parallel=False, cache=True)
def compute_fluxes_moles_pc_conc_numba(flux, netFlux, len_tij_valid, gasConc, volarray,
    TPValid, TValid, clusterID, clust_moles, clust_volume, clust_pc, clust_size, clust_netMoles,
    cNetMolesAfterLastUpdate, members, mem_offsets, elemToUpdateW, elemToUpdateNW, 
    aqMoles, minMoles, minPcArray, maxPc, notValid, imposedP, D, H, RT, _dt, MOLES_EPS):
    
    fSize = flux.size
    nfSize = netFlux.size

    for e in prange(nfSize):
        netFlux[e] = 0.0

    netflux_Aq = 0.0
    for i in range(fSize):
        p, t = TPValid[i], TValid[i]
        if volarray[p]==0.0 or volarray[t]==0.0:
            f = 0.0
        else:
            f = D * len_tij_valid[i] * (gasConc[p] - gasConc[t])
            
        if f > 0.0:
            if (aqMoles[p] - minMoles[p]) <= 1e-3*aqMoles[p]:
                f = 0.0
        elif f < 0.0:
            if (aqMoles[t] - minMoles[t]) <= 1e-3*aqMoles[t]:
                f = 0.0
        
        flux[i] = f
        netFlux[t] += f
        netFlux[p] -= f
        if elemToUpdateW[t]:
            netflux_Aq += f
        if elemToUpdateW[p]:
            netflux_Aq -= f

    dt = _dt
    for e in range(nfSize):
        if netFlux[e] >= 0.0: continue
        tratio =  (minMoles[e] - aqMoles[e])/netFlux[e]
        if tratio < 0.0:
            return 0.0
        if dt > tratio:
            dt = tratio

    excess = 0.0
    for e in range(nfSize):
        if netFlux[e] == 0.0: continue
        nMoles = netFlux[e] * dt
        if nMoles < -aqMoles[e]:
            excess += (aqMoles[e] + nMoles)
            nMoles = -aqMoles[e]
        
        if elemToUpdateNW[e]:
            k = clusterID[e]
            clust_netMoles[k] += nMoles
        elif elemToUpdateW[e]:
            aqMoles[e] += nMoles
            gasConc[e] = aqMoles[e]/volarray[e]
    
    nClusters = clust_moles.size
    for k in range(nClusters):
        if clust_size[k] == 0 or clust_netMoles[k] == 0.0: continue
        newMole_k = clust_moles[k] + clust_netMoles[k]
        cNetMolesAfterLastUpdate[k] += clust_netMoles[k]
        clust_netMoles[k] = 0.0
        
        if newMole_k < 0.0:
            excess += newMole_k
            newMole_k = 0.0
        clust_moles[k] = newMole_k
                
    if excess != 0.0:
        for e in range(nfSize):
            if elemToUpdateW[e] and (aqMoles[e]+excess>=minMoles[e]):
                aqMoles[e] += excess
                break
        
    return dt   
    

@njit(float64(
    float64[:], float64[:], float32[:], float64[:], float64[:],
    int32[:], int32[:], int32[:], float64[:], float64[:], float32[:], int32[:], float64[:],
    float64[:], int32[:], int32[:], boolean[:], boolean[:], 
    float64[:], float64[:], float32[:], float64, boolean[:], float64, float64, float64, float64, float64, float64), 
    fastmath=True, parallel=False, cache=True)
def compute_fluxes_moles_pc_conc_numba0(flux, netFlux, len_tij_valid, gasConc, volarray,
    TPValid, TValid, clusterID, clust_moles, clust_volume, clust_pc, clust_size, clust_netMoles,
    cNetMolesAfterLastUpdate, members, mem_offsets, elemToUpdateW, elemToUpdateNW, 
    aqMoles, minMoles, minPcArray, maxPc, notValid, imposedP, D, H, RT, _dt, MOLES_EPS):
    
    fSize = flux.size
    nfSize = netFlux.size

    for e in prange(nfSize):
        netFlux[e] = 0.0

    netflux_Aq = 0.0
    for i in range(fSize):
        p, t = TPValid[i], TValid[i]
        if volarray[p]==0.0 or volarray[t]==0.0:
            f = 0.0
        else:
            f = D * len_tij_valid[i] * (gasConc[p] - gasConc[t])
            
        
        # only execute this constraint when both p and t have no gas
        if f > 0.0:
            if (aqMoles[p] - minMoles[p]) <= 1e-3*aqMoles[p]:
                f = 0.0
        elif f < 0.0:
            if (aqMoles[t] - minMoles[t]) <= 1e-3*aqMoles[t]:
                f = 0.0

        flux[i] = f
        netFlux[t] += f
        netFlux[p] -= f
        if elemToUpdateW[t]:
            netflux_Aq += f
        if elemToUpdateW[p]:
            netflux_Aq -= f

    dt = _dt
    for e in range(nfSize):
        if netFlux[e] >= 0.0: continue
        tratio =  (minMoles[e] - aqMoles[e])/netFlux[e]
        if tratio < 0.0:
            return 0.0
        if dt > tratio:
            dt = tratio

    excess = 0.0
    for e in range(nfSize):
        if netFlux[e] == 0.0: continue
        nMoles = netFlux[e] * dt
        if nMoles < -aqMoles[e]:
            excess += (aqMoles[e] + nMoles)
            nMoles = -aqMoles[e]
        
        if elemToUpdateNW[e]:
            k = clusterID[e]
            clust_netMoles[k] += nMoles
        elif elemToUpdateW[e]:
            aqMoles[e] += nMoles
            gasConc[e] = aqMoles[e]/volarray[e]
    
    nClusters = clust_moles.size
    for k in range(nClusters):
        if clust_size[k] == 0 or clust_netMoles[k] == 0.0: continue
        newMole_k = clust_moles[k] + clust_netMoles[k]
        cNetMolesAfterLastUpdate[k] += clust_netMoles[k]
        clust_netMoles[k] = 0.0
        
        if newMole_k < 0.0:
            excess += newMole_k
            newMole_k = 0.0
              
        mem = members[mem_offsets[k]:mem_offsets[k+1]]
        volTotal = volarray[mem].sum()
        
       
        minPc = 1e-3
        if clust_volume[k] > 0.0:
            clust_moles[k] = newMole_k
            pc_k = max(minPc, clust_moles[k]*RT/clust_volume[k] - imposedP)
        elif clust_moles[k] > MOLES_EPS:
            fr = newMole_k/clust_moles[k]
            if fr > 1e-3: fr = 1e-3            
            pc_k = max(minPc, fr*(imposedP+clust_pc[k]) - imposedP)
            clust_moles[k] = newMole_k
        else:
            pc_k = clust_pc[k] 
            clust_moles[k] = newMole_k
            
        if pc_k > maxPc: 
            pc_k = maxPc
        clust_pc[k] = pc_k
        for m in mem:
            gasConc[m] = H*(pc_k+imposedP)
            aqMoles[m] = min(gasConc[m]*volarray[m], volarray[m]/volTotal*newMole_k) 
                
    if excess != 0.0:
        for e in range(nfSize):
            if elemToUpdateW[e] and (aqMoles[e]+excess>=minMoles[e]):
                aqMoles[e] += excess
                break
        
    return dt   


@njit(Tuple((float64, float64, float64, float64, float64, float64))(
    boolean[:], boolean[:], float64[:], float64[:], float64[:],
    float64[:], float64[:], boolean[:], float32[:], float64[:], int64, boolean[:],
    float64, float64, float64), cache=True)
def statistics_numba(elemToUpdateW, elemtoUpdateNW, aqMoles, satList, volarray, 
    clust_moles, prequilibratedMoles, valClust, clusterPc, clusterVol, totElements, isinsideBox,
    totVoidVolume, H, imposedP):
    
    total_moles_in_gas = 0.0
    total_moles_in_aq = 0.0
    total_aq_vol_in_box = 0.0
    total_aq_vol = 0.0
    pcVolSum = 0.0
    volSum = 0.0
    nClusters = valClust.size
    for i in range(totElements):
        if elemToUpdateW[i]:
            total_moles_in_aq += aqMoles[i]
            total_aq_vol += volarray[i]

        if isinsideBox[i]:
            total_aq_vol_in_box += (satList[i]*volarray[i])

    for k in range(nClusters):
        total_moles_in_gas += clust_moles[k]
        if valClust[k]:
            pcVolSum += (clusterPc[k]*clusterVol[k])
            volSum += clusterVol[k]

    total_moles = total_moles_in_gas + total_moles_in_aq
    satW = total_aq_vol_in_box/totVoidVolume
    avgGasPres = pcVolSum/volSum
    avgAqPres = total_moles_in_aq/(H*total_aq_vol) - imposedP

    return total_moles_in_gas, total_moles_in_aq, total_moles, satW, avgGasPres, avgAqPres
    
    
@njit(void(int32[:], float64, float64[:], float32[:], float32[:], float64[:], float64[:], float64[:]), cache=True, inline='always')
def updateSatMolesConc_numba(mem, new_conc, satList, _cornArea, areaSPhase, gasConc, aqMoles, volarray):
    for j in range(mem.size):
        m = mem[j]
        satList[m] = _cornArea[m]/areaSPhase[m]
        gasConc[m] = new_conc
        aqMoles[m] = new_conc*volarray[m]


@njit(void(
    int32[:], float32[:], boolean[:], boolean[:], float32[:],
    float32[:], float32[:], float32[:], float32[:], boolean, int64, 
    float64, float32[:], float32[:], int32[:], float32[:], float32[:], float32[:],
    float32[:], float32[:], float64[:], float64[:], int32[:], float64[:], float32[:],
    float32[:], float32[:], float64[:], int32[:], int32[:], int32[:], float64[:], float64[:],
    float64, float64, float64, float64, float64, float32[:], float64, float64, float64
    ), cache=True)
def returnClusterPcVolume_numba(keys, halfAng, m_exists, m_inited, m_initOrMaxPcHist, 
    m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, is_oil_inj, totElements,
    sigma, thetaAdvAng, thetaRecAng, nCorners_arr, areaSPhase, _cornArea, _centerArea, 
    conAng_cur, apexDist_cur, volarray, satList, clusterID, clusterVol, clusterPc, 
    pcShrink, pcMax, clusterMoles, members, mem_offsets, sizes, gasConc, aqMoles,
    imposedP, molecular_length, _delta, H, RT, minPcArray, maxPc, moles_tol, pc_tol):

    accurate = False
    nClusters = clusterPc.size
    
    nkeys = keys.size
    for i in range(nkeys):
        k = keys[i]
        moles_k = clusterMoles[k]
        start, end = mem_offsets[k], mem_offsets[k+1]
        mem = members[start:end]
        first_time = True
        minPc = 1e-3
        
        # at current pc
        currPc = clusterPc[k]        
        volC, molesC = returnClusterVolMoles(k, mem, currPc, halfAng, m_exists, m_inited, 
            m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist,
            is_oil_inj, totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr, areaSPhase, 
            _cornArea, _centerArea, conAng_cur, apexDist_cur, volarray, satList, clusterID,
            imposedP, molecular_length, _delta, RT, True, first_time)
        first_time = False
        
        if abs(molesC - moles_k) <= moles_tol:
            clusterVol[k] = volC
            new_conc = H * (imposedP + currPc)
            updateSatMolesConc_numba(mem, new_conc, satList, _cornArea, areaSPhase, gasConc, aqMoles, volarray)
            continue
        elif moles_k < molesC:
            vol2, moles2, pc2 = volC, molesC, currPc
            
            # at minimum possible pc                 
            vol1, moles1 = returnClusterVolMoles(k, mem, minPc, halfAng, m_exists, m_inited, 
                m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist,
                is_oil_inj, totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr, areaSPhase, 
                _cornArea, _centerArea, conAng_cur, apexDist_cur, volarray, satList, clusterID,
                imposedP, molecular_length, _delta, RT, False, first_time)
            if moles_k < moles1:
                clusterVol[k] = vol1
                clusterPc[k] = minPc
                new_conc = H * (imposedP + minPc)
                updateSatMolesConc_numba(mem, new_conc, satList, _cornArea, areaSPhase, gasConc, aqMoles, volarray)
                continue
            else:
                pc1 = minPc
        else:
            vol1, moles1, pc1 = volC, molesC, currPc
            
            # at maximum allowable pc
            vol2, moles2 = returnClusterVolMoles(k, mem, maxPc, halfAng, m_exists, m_inited, 
                m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, 
                is_oil_inj, totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr, areaSPhase, 
                _cornArea, _centerArea, conAng_cur, apexDist_cur, volarray, satList, clusterID, 
                imposedP, molecular_length, _delta, RT, False, first_time)
            if moles_k > moles2:
                clusterVol[k] = vol2
                clusterPc[k] = maxPc
                new_conc = H * (imposedP + maxPc)
                updateSatMolesConc_numba(mem, new_conc, satList, _cornArea, areaSPhase, gasConc, aqMoles, volarray)
                continue
            else:
                pc2 = maxPc
        
        f1 = moles1 - moles_k
        f2 = moles2 - moles_k
        if (abs(pc2 - pc1) <= pc_tol):
            clusterVol[k] = volC
            new_conc = H * (imposedP + currPc)
            updateSatMolesConc_numba(mem, new_conc, satList, _cornArea, areaSPhase, gasConc, aqMoles, volarray)
            continue
        else:
            for r in range(12):                
                denom = (f2 - f1)
                if denom == 0.0:
                    # fallback to bisection if secant slope is degenerate
                    pcNext = 0.5 * (pc1 + pc2)
                else:
                    # Secant step inside the bracket
                    pcNext = (pc1 * f2 - pc2 * f1) / denom

                # Safety: If roundoff pushes pcNext outside, bisect instead
                if not (pc1 < pcNext < pc2):
                    pcNext = 0.5 * (pc1 + pc2)
                    
                volN, molesN = returnClusterVolMoles(k, mem, pcNext, halfAng, m_exists, 
                    m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc, 
                    m_initedApexDist, is_oil_inj, totElements, sigma, thetaAdvAng, thetaRecAng, 
                    nCorners_arr, areaSPhase, _cornArea, _centerArea, conAng_cur, apexDist_cur, 
                    volarray, satList, clusterID, imposedP, molecular_length, _delta, 
                    RT, False, first_time)
                    
                fN = molesN - moles_k
                if abs(fN) < moles_tol or abs(pc1 - pc2) <= pc_tol:
                    break
                                    
                if fN * f1 > 0.0:
                    pc1, vol1, f1 = pcNext, volN, fN
                    f2 *= 0.5  # Illinois damping
                else:
                    pc2, vol2, f2 = pcNext, volN, fN
                    f1 *= 0.5  # Illinois damping
                    
            clusterVol[k] = volN
            if volN>0.0:
                currPc = max(minPc, moles_k*RT/volN - imposedP)
            else:
                currPc = pcNext
                                
            if currPc > maxPc:
                currPc = maxPc
            new_conc = H * (imposedP + currPc)
            clusterPc[k] = currPc
            updateSatMolesConc_numba(mem, new_conc, satList, _cornArea, areaSPhase, gasConc, aqMoles, volarray)           
            
            
            
@njit(void(
    float64[:], float32[:], boolean[:], boolean[:], float32[:],
    float32[:], float32[:], float32[:], float32[:], int64, 
    float64, float32[:], float32[:], int32[:], float32[:], float32[:], float32[:],
    float32[:], float32[:], float64[:], float64[:], int32[:], float64[:], float32[:],
    float32[:], float32[:], float64[:], int32[:], int32[:], int32[:], int32[:], float64[:], float64[:],
    float64, float64, float64, float64, float64, float32[:], float64, float64, float64), cache=True)
def adjustClusterPcVolume_numba(cNetMolesAfterLastUpdate, halfAng, m_exists, m_inited, m_initOrMaxPcHist, 
    m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, totElements,
    sigma, thetaAdvAng, thetaRecAng, nCorners_arr, areaSPhase, _cornArea, _centerArea, 
    conAng_cur, apexDist_cur, volarray, satList, clusterID, clusterVol, clusterPc, 
    pcShrink, pcMax, clusterMoles, members, mem_offsets, sizes, tempID, gasConc, aqMoles,
    imposedP, molecular_length, _delta, H, RT, minPcArray, maxPc, moles_tol, pc_tol):
    
    nClusters = cNetMolesAfterLastUpdate.size
    i, j = 0, nClusters
    for k in range(nClusters):
        fr_mole_k = max(clusterMoles[k]*1e-10, 1e-14)
        netMole_k = cNetMolesAfterLastUpdate[k]
        if netMole_k > fr_mole_k:
            tempID[i] = k 
            i += 1
        elif netMole_k < -fr_mole_k:       
            j -= 1       
            tempID[j] = k
                        
    if i!=0:
        keysG = tempID[:i]
        returnClusterPcVolume_numba(keysG, halfAng, m_exists, m_inited, m_initOrMaxPcHist, 
            m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, True, totElements,
            sigma, thetaAdvAng, thetaRecAng, nCorners_arr, areaSPhase, _cornArea, _centerArea, 
            conAng_cur, apexDist_cur, volarray, satList, clusterID, clusterVol, clusterPc, 
            pcShrink, pcMax, clusterMoles, members, mem_offsets, sizes, gasConc, aqMoles,
            imposedP, molecular_length, _delta, H, RT, minPcArray, maxPc, moles_tol, pc_tol)
        cNetMolesAfterLastUpdate[keysG] = 0.0
        
    if j != nClusters-1:
        keysS = tempID[j:]
        returnClusterPcVolume_numba(keysS, halfAng, m_exists, m_inited, m_initOrMaxPcHist, 
            m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, False, totElements,
            sigma, thetaAdvAng, thetaRecAng, nCorners_arr, areaSPhase, _cornArea, _centerArea, 
            conAng_cur, apexDist_cur, volarray, satList, clusterID, clusterVol, clusterPc, 
            pcShrink, pcMax, clusterMoles, members, mem_offsets, sizes, gasConc, aqMoles,
            imposedP, molecular_length, _delta, H, RT, minPcArray, maxPc, moles_tol, pc_tol)
        cNetMolesAfterLastUpdate[keysS] = 0.0
        
            
            
        






    