import numpy as np
from joblib import dump, load, Parallel, delayed
from scipy import optimize
from numba import njit, prange, jit, types, typeof
from numba.experimental import jitclass
from numba.types import int32, float64, Array
from sortedcontainers import SortedList

from pnflowPy.cluster import Cluster as clust
import pnflowPy.utilities as do
from OstRipening.network_numba import ClusterJit, NetworkJit
from pnflowPy.cluster import ClusterObj 
import OstRipening.temp as temp
from OstRipening.utilities_numba import *


_temp = temp.TempArrays()

class Cluster(clust):
    def __init__(self, network, phase=1, nClusters=200): 
        super().__init__(network, phase, nClusters)
        temp.TempArrays.formTempNetworkArrays(
            network.nPores, network.nThroats, network.totElements, 
            network.connectivity_graph_flat.size, network.m_halfAngles)
        

    def loadCluster(self, jit, network, all=False):  

        self.size = jit.clustSize.view()
        self.members = jit.members.view()
        self.neighbours = jit.neighbours.view()
        self.trappedStatus = jit.trappedStatus.view()
        self.connected = jit.connected.view()
        self.pc = jit.pc.view()

        if self.phase == 0:
            self.trapped = network.trappedW = jit.trapped.view()
            self.conn = network.connW = jit.conn.view()
            self.hasFluid = network.hasWFluid = jit.hasFluid.view()
            self.clusterID = network.clusterW_ID = jit.clusterID.view()
        else:
            self.trapped = network.trappedNW = jit.trapped.view()
            self.conn = network.connNW = jit.conn.view()
            self.hasFluid = network.hasNWFluid = jit.hasFluid.view()
            self.clusterID = network.clusterNW_ID = jit.clusterID.view()
        
        if all:
            self.volume = jit.volume.view()
            self.valClust = jit.valClust.view()
            self.valClustD = jit.valClustD.view()
            self.n_exponents = jit.n_exponents.view()
            self.totalVolume = jit.totalVolume.view()
            self.criticalPc = jit.criticalPc.view()
            self.criticalVolume = jit.criticalVolume.view()
            self.criticalMoles = jit.criticalMoles.view()
            self.dVg_dPc = jit.dVg_dPc.view()
            self.toDrain = jit.toDrain.view()
            self.toImbibe = jit.toImbibe.view()
            self.pcMax = jit.pcMax.view()
            self.pcShrink = jit.pcShrink.view()
            self.moles = jit.moles.view()
            self.molesMax = jit.molesMax.view()
            self.molesShrink = jit.molesShrink.view()
            self.volMax = jit.volMax.view()
            self.volShrink = jit.volShrink.view()


    def resizeClusters(self, size, ostMode=False, update=False):
        super().resizeClusters(size)
        
        if ostMode:
            oldSize, newSize = self.nClusters-size, self.nClusters

            if hasattr(self, 'neighbours'):
                self.neighbours.extend([[] for _ in range(size)])
            else:
                self.neighbours = [[] for _ in range(newSize)]

            if self.phase==0: return

            attrList = ['pcShrink', 'pcMax', 'volume', 'volShrink', 'volMax',
                        'moles', 'molesShrink', 'molesMax',
                        'toDrain', 'toImbibe', 'valClust', 'valClustD']
            dtypes = [np.float32, np.float32, np.float64, np.float32, np.float32,
                np.float64, np.float32, np.float32,
                np.int32, np.int32, np.bool_, np.bool_]
            
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
        

    def __neighbours__(self, network, keys=None):
        if keys is None:
            keys = np.flatnonzero(self.size>0).astype('int32')
        self.updateClusterNeighbours(keys, network)
    

   

    def removeMembers(self, keys, toImbibe, network):
        ''' return the new pc after imbibition or drainage '''
        cond = (self.size[keys]==1)
        keys1, toImbibe1 = keys[cond], toImbibe[cond]
        network.aqueousMoles[toImbibe1] = self.moles[keys1]
        for k in keys:
            self.neighbours[k] = []

        notcond = ~cond
        keys2, toImbibe2 = keys[notcond], toImbibe[notcond]
        network.aqueousMoles[toImbibe2] = np.minimum(self.moles[keys2],
            #network.gasConc[toImbibe2,0]*network.volarray[toImbibe2],
            #network.gasConc[0,toImbibe2]*network.volarray[toImbibe2]
            network.minMoles[toImbibe2])

        #network.gasConc[toImbibe,1] = network.aqueousMoles[toImbibe]/network.volarray[toImbibe]
        network.gasConc[1, toImbibe] = network.aqueousMoles[toImbibe]/network.volarray[toImbibe]

        self.moles[keys] -= network.aqueousMoles[toImbibe]
        for i in range(toImbibe.size):
            self.unfill_phase(toImbibe[i], self.pc[keys[i]])


    def pcAfterShrinkage(self, keys, toImbibe, network):
        ''' return the new pc after imbibition or drainage '''
        term = (self.volume[keys] - network.volarray[toImbibe])/network.RT
        num = self.moles[keys] - network.imposedP*term
        den = term + network.H*network.volarray[toImbibe]
        return np.maximum(np.divide(num, den, where=(den!=0.0)), network.Pc)


    def addMembers(self, keys, toAdd, network):
        oldKeys = _temp.tempID
        oldKeys[:] = self.clusterID
        clusterW = network.clusterW
        mem = _temp.mem
        mem[mem] = False
        _toDrain = list(toAdd)
        _keys = list(keys)

        while len(_keys):
            ii = _toDrain.pop()
            k = _keys.pop()

            if self.hasFluid[ii]: continue

            if network.isCircle[ii]:
                clusterW_ID = clusterW.clusterID
                clusterW.unfill_phase(ii, clusterW.pc[clusterW_ID[ii]])

            if not network.isPore[ii]:
                conP = network.elem[ii].neighbours
                cond = (conP>0) & (network.PcI[conP] < network.PcI[ii]) & ~self.hasFluid[conP]
                if cond.any():
                    pp = conP[0] if cond[0] else conP[1]
                    _toDrain.append(pp)
                    _keys.append(k)
                    network.elemToUpdateW[pp] = False
                    network.elemToUpdateNW[pp] = True

            self.moles[k] += network.aqueousMoles[ii]
            network.aqueousMoles[ii] = 0.0
            self.fill_with_phase(ii, self.pc[k], network)
            newK = self.clusterID[ii]
            memID = self[newK].members
            mem[memID] = True
            memKeys = oldKeys[memID]
            memKeys = np.unique(memKeys[(memKeys>=0)])
            oldKeys[memID] = newK
            self.moles[newK] = self.moles[memKeys].sum()
            if memKeys.size>0:
                memKeys = memKeys[memKeys!=newK]
                self.moles[memKeys] = 0.0

        mem[[-1, 0]] = False
        _mem = np.flatnonzero(mem).astype(np.int32)
        try:
            thr = np.concatenate(network.PTConData[_mem[_mem<network.nPores+1]])
            mem[thr] = True
        except ValueError:
            pass

        for k in keys:
            self.neighbours[k] = []
       
        return _mem, mem[network.tList]


    def disappearCluster(self, network, pc_min, pc_max):   
        cond = _temp.resCond
        cond[:] = (self.size==0)
        cond[0] = False  
        keys = np.flatnonzero(cond).astype(np.int32)
        if not keys.any(): return
        #self.neighbours[keys] = False
        for k in keys:
            del self[k]
            self.neighbours[k] = []
            self.molesShrink[k], self.molesMax[k] = 0.0, 1e30
            self.volShrink[k] = self.volMax[k] = self.volume[k] = 0.0
            self.pcShrink[k], self.pcMax[k] = 0.0, 1e30
        updateToDrainImbibe_numba(keys, self.members, self.members_offset, self.size, self.neighbours, 
            self.toDrain, self.toImbibe, self.pcShrink, self.pcMax, network.PcI, network.PcD, 
            network.Rarray, network.totElements, _temp.mem, True, pc_min, pc_max)
        network.validToShrink[keys] = False
        network.validToGrow[keys] = False

        keys = np.setdiff1d(keys, self.availableID)
        keys = keys[keys!=0]
        #self.neighbours[keys] = False
        self.availableID.update(keys)


    def updateClusterNeighbours(self, keys, network):
        keys = list(keys)
        while len(keys):
            k = keys.pop()
            neigh = np.unique(np.concatenate(network.connectivity_graph[self[k].members]))
            nonMem = neigh[self.clusterID[neigh]!=k]
            while self.hasFluid[nonMem].any():
                # === coalescence occurs === #
                newKs = self.clusterID[nonMem]
                for j in np.unique(newKs[newKs>=0]):
                    sizeJ = self.size[j]
                    memJ = self[j].members
                    self.members_pos[memJ] = self.size[k]+np.arange(sizeJ)
                    self.size[k] += sizeJ
                    print('im on line 392 in updateClusterNeighbours!!!')
                    from IPython import embed; embed()
                    self.members[k].extend(memJ)
                    self.members[j] = []
                    self.size[j] = 0
                    self.trapped[k] &= self.trapped[j]
                    self.clusterID[memJ] = k

                keys.append(k)
    
            self.neighbours[k] = list(nonMem)

    
    def shrinkCluster(self, keys, network, moles_tol=1e-23, pc_tol=1e-4, 
                    max_iter=50, pc_min=1e-30, pc_max=1e30):
        ''' update toImbibe element(s) !!! '''
        # fill with the other phase
        network.clustersToShrink[keys] = False
        clust = network.clusterW if self.phase==1 else network.clusterNW
        for k in keys:
            clust.fill_with_phase(self.toImbibe[k], self.pc[k], network)

        # clusters to shrink
        toImbibe = self.toImbibe[keys]
        network.satList[toImbibe] = 1.0
        mem = _temp.mem
        mem[mem] = False
        for k in keys:
            mem[self[k].members] = True
        mem[toImbibe] = False
        oldMemKeys = self.clusterID[mem]

        self.removeMembers(keys, toImbibe, network)
        if not mem.any():
            self.disappearCluster(network, pc_min, pc_max)
            return

        _mem = np.flatnonzero(mem).astype(np.int32)
        self.updateClusterNeighbours(np.unique(self.clusterID[_mem]), network)
        newMemKeys = self.clusterID[_mem]
        self.disappearCluster(network, pc_min, pc_max)
        
        if np.all(oldMemKeys==newMemKeys):      #case1
            updateToDrainImbibe_numba(keys, self.members, self.members_offset, self.size, 
                self.neighbours, self.toDrain, self.toImbibe, self.pcShrink, self.pcMax, network.PcI, 
                network.PcD, network.Rarray, network.totElements, _temp.mem, True, pc_min, pc_max)
            self.valClustD[:] = self.valClust & (self.toDrain<=network.totElements)
            network.validToShrink[keys] = self.valClust[keys] & (self.pcShrink[keys]>network.oldaqAvgPres)
            network.validToGrow[keys] = self.valClustD[keys] & (self.pcMax[keys]<network.oldaqAvgPres)
            oldVol = self.volume[keys]
            oldPc = self.pc[keys]
            if 500 in keys and 22567 in toImbibe:
                print(f'keys={keys}; toImbibe={toImbibe}')
                from IPython import embed; embed()

            self.updateClusterPcVolume(
                keys, newMemKeys, _mem, self.moles, network, moles_tol, pc_tol, max_iter)
            # if (self.pc[keys] > oldPc).any():
            #     print('there is a strange new pc value @@')
            #     from IPython import embed; embed()
    
            cond = (toImbibe==self.toDrain[keys])
            if cond.any():
                spKeys, spElem = keys[cond], toImbibe[cond]
                self.adjustClusterPcShrink(spKeys, spElem, network)
                self.adjustClusterPcMax(spKeys, spElem, network)   
            #network.gasConc[_mem, 1] = network.H*self.pc[newMemKeys]         
            network.gasConc[1, _mem] = network.H*self.pc[newMemKeys]

        else:                                                     #case2
            _oldMemKeys, _newMemKeys = mapOldNewKeys(oldMemKeys, newMemKeys, _temp.resCond)
            oldToImbibe = self.toImbibe[_oldMemKeys]
            updateToDrainImbibe_numba(_newMemKeys, self.members, self.members_offset, self.size, 
                self.neighbours, self.toDrain, self.toImbibe, self.pcShrink, self.pcMax, network.PcI, 
                network.PcD, network.Rarray, network.totElements, _temp.mem, True, pc_min, pc_max)
            self.valClustD[:] = self.valClust & (self.toDrain<=network.totElements) #revise
            network.validToShrink[_newMemKeys] = self.valClust[_newMemKeys] & (self.pcShrink[_newMemKeys]>network.oldaqAvgPres)
            network.validToGrow[_newMemKeys] = self.valClustD[_newMemKeys] & (self.pcMax[_newMemKeys]<network.oldaqAvgPres)
            oldPc = self.pc[_oldMemKeys]
            self.distributeMoles(_newMemKeys, newMemKeys, _oldMemKeys, _mem, network)
            self.updateClusterPcVolume(
                _newMemKeys, newMemKeys, _mem, self.moles, network, moles_tol, pc_tol, max_iter)
            if (self.pc[_newMemKeys] > oldPc).any():
                print('there is a strange new pc value ££')
                from IPython import embed; embed()

            cond = (oldToImbibe==self.toDrain[_newMemKeys])
            if cond.any():
                spKeys, spElem = _newMemKeys[cond], oldToImbibe[cond]
                self.adjustClusterPcShrink(spKeys, spElem, network)
                self.adjustClusterPcMax(spKeys, spElem, network)

            #network.gasConc[_mem,1] = network.H*self.pc[newMemKeys] #
            network.gasConc[1, _mem] = network.H*self.pc[newMemKeys]


    def growCluster(self, keys, network, moles_tol=1e-23, pc_tol=1e-4, max_iter=50, 
                    pc_min=1.0e-30, pc_max=1.0e30):
        ''' update toDrain element(s) !!! '''
        network.clustersToGrow[keys] = False
        toDrain = self.toDrain[keys]
        _mem, neigh = self.addMembers(keys, toDrain, network)
        self.updateClusterNeighbours(np.unique(self.clusterID[_mem]), network)
        self.disappearCluster(network, pc_min, pc_max)
        
        memkeys = self.clusterID[_mem]
        toDrainKeys = np.unique(memkeys)
        updateToDrainImbibe_numba(toDrainKeys, self.members, self.members_offset, self.size, 
            self.neighbours, self.toDrain, self.toImbibe, self.pcShrink, self.pcMax, network.PcI, 
            network.PcD, network.Rarray, network.totElements, _temp.mem, True, pc_min, pc_max)
        self.valClustD[:] = self.valClust & (self.toDrain<=network.totElements)
        network.validToShrink[toDrainKeys] = self.valClust[toDrainKeys] & (self.pcShrink[toDrainKeys]>network.oldaqAvgPres)
        network.validToGrow[toDrainKeys] = self.valClustD[toDrainKeys] & (self.pcMax[toDrainKeys]<network.oldaqAvgPres)

        oldPc = self.pc[toDrainKeys]
        self.updateClusterPcVolume(
            toDrainKeys, memkeys, _mem, self.moles, network, moles_tol, pc_tol, max_iter)
        if (self.pc[keys] > oldPc).any():
            print('there is a strange new pc value $$')
            from IPython import embed; embed()
        newKeys = self.clusterID[toDrain]
        cond = (toDrain == self.toImbibe[newKeys])
        if cond.any():
            spElem = toDrain[cond]
            spKeys = self.clusterID[spElem]
            self.adjustClusterPcShrink(spKeys, spElem, network)
            self.adjustClusterPcMax(spKeys, spElem, network)

        #network.gasConc[_mem,1] = network.H*self.pc[memkeys] #
        network.gasConc[1, _mem] = network.H*self.pc[memkeys]
        self.trapped[_mem] = self.trappedStatus[memkeys]
        
        return toDrain


    def updateClusterPcVolume(self, keys, memkeys, memID, targetMoles, ntwk,
        moles_tol, pc_tol, max_iter):

        findPcVolume(keys, memkeys, memID, targetMoles, ntwk.contactAng, ntwk.m_halfAngles, ntwk.m_cornExists, 
            ntwk.m_initOrMaxPcHist, ntwk.m_initOrMinApexDistHist, ntwk.m_advPc, ntwk.m_recPc, 
            ntwk.m_initedApexDist, ntwk.trappedW, ntwk.trappedNW, ntwk.clusterW.pc, ntwk.clusterNW.pc, 
            ntwk.clusterW_ID, ntwk.clusterNW_ID, ntwk.clusterNW.volume, self. pcShrink, self.pcMax, 
            ntwk.areaSPhase, ntwk.maxCornerArea, ntwk.volarray, ntwk.sigma, ntwk.thetaAdvAng, ntwk.thetaRecAng, 
            ntwk.PcD, ntwk.satList, _temp.guessVol, _temp.volume, ntwk._delta, ntwk.isSquare, ntwk.isTriangle, 
            _temp.area, ntwk.muw, ntwk.MOLECULAR_LENGTH, ntwk.imposedP, ntwk.RT, ntwk.H, moles_tol, pc_tol, max_iter, 
            _temp.resCond, _temp.guessPcL, _temp.guessPcH, _temp.guessPc)


    def distributeMoles(self, keys, memkeys, oldkeys, memID, ntwk):
        _, index, cnt = np.unique(oldkeys, return_counts=True, return_inverse=True)
        shr_to_cleavage = _temp.resCond # shrinkage leading to cleavage

        condk = (cnt > 1)[index]
        if not condk.any(): return
        shr_to_cleavage[keys] = condk
        keys1, oldkeys1 = keys[condk], oldkeys[condk]
        
        condm = shr_to_cleavage[memkeys]
        memkeys1 = memkeys[condm]
        memID1 = memID[condm]

        _, moleShr = returnVolMoles(keys1, memkeys1, memID1, self.pcShrink, ntwk.contactAng, ntwk.m_halfAngles, 
            ntwk.m_cornExists, ntwk.m_initOrMaxPcHist, ntwk.m_initOrMinApexDistHist,
            ntwk.m_advPc, ntwk.m_recPc, ntwk.m_initedApexDist, ntwk.trappedW, ntwk.trappedNW, 
            ntwk.clusterW.pc, ntwk.clusterNW.pc, ntwk.clusterW_ID, ntwk.clusterNW_ID, 
            ntwk.areaSPhase, ntwk.maxCornerArea, ntwk.volarray, ntwk.sigma, ntwk.thetaAdvAng, ntwk.thetaRecAng, 
            ntwk.PcD, ntwk.satList, _temp.guessVol, _temp.volume, ntwk._delta, ntwk.isSquare, ntwk.isTriangle, 
            _temp.area, ntwk.muw, ntwk.MOLECULAR_LENGTH, ntwk.imposedP, ntwk.RT, False)

        _, moleMax = returnVolMoles(keys1, memkeys1, memID1, self.pcMax, ntwk.contactAng, ntwk.m_halfAngles, 
            ntwk.m_cornExists, ntwk.m_initOrMaxPcHist, ntwk.m_initOrMinApexDistHist,
            ntwk.m_advPc, ntwk.m_recPc, ntwk.m_initedApexDist, ntwk.trappedW, ntwk.trappedNW, 
            ntwk.clusterW.pc, ntwk.clusterNW.pc, ntwk.clusterW_ID, ntwk.clusterNW_ID, 
            ntwk.areaSPhase, ntwk.maxCornerArea, ntwk.volarray, ntwk.sigma, ntwk.thetaAdvAng, ntwk.thetaRecAng, 
            ntwk.PcD, ntwk.satList, _temp.guessVol, _temp.volume, ntwk._delta, ntwk.isSquare, ntwk.isTriangle, 
            _temp.area, ntwk.muw, ntwk.MOLECULAR_LENGTH, ntwk.imposedP, ntwk.RT, False)

        moleShrCum = np.bincount(oldkeys1, moleShr)[oldkeys1]
        np.logical_and.at(shr_to_cleavage, oldkeys1, moleMax>moleShr)
        valid = ((self.moles[oldkeys1] - moleShrCum) >= 0.0) & shr_to_cleavage[oldkeys1] # condition above shrinkage condition
        diff = (moleMax[valid] - moleShr[valid])
        oldkeys2, oldkeys3 = oldkeys1[valid], oldkeys1[~valid]

        self.moles[keys1[valid]] = moleShr[valid] + (
            diff/np.bincount(oldkeys2, diff)[oldkeys2]*(self.moles[oldkeys2]-moleShrCum[valid]))
        self.moles[keys1[~valid]] = moleShr[~valid]/moleShrCum[~valid]*self.moles[oldkeys3]
        self.moles[np.setdiff1d(oldkeys, keys)] = 0.0


    def adjustClusterPcMax(self, keys, elem, ntwk, otherKeys=None, shift_ind=None):
        _temp.mem[_temp.mem] = False
        for k in keys:
            _temp.mem[self[k].members] = True
            

        #_temp.mem[:] = self.members[keys].any(axis=0)
        _temp.mem[elem] = True
        extra = otherKeys is not None and otherKeys.size > 0
        if extra:
            for k in otherKeys:
                _temp.mem[self[k].members] = True
        
        memID = np.flatnonzero(_temp.mem).astype(np.int32)
        _temp.tempID[memID] = self.clusterID[memID]
        _temp.tempID[elem] = keys
        _temp.guessPc[keys] = np.max((self.pcShrink[keys], ntwk.PcI[elem]), axis=0)
        if extra:
            _temp.tempID[mem] = keys[shift_ind[ind]]
            np.maximum.at(_temp.guessPc, _temp.tempID[mem], ntwk.PcI[mem])

        keys = np.unique(keys)        
        volMax = computeVolume_numba(keys, _temp.tempID[memID], memID, _temp.guessPc, ntwk.contactAng, _temp.area, 
            ntwk.m_halfAngles, ntwk.m_cornExists, ntwk.m_initOrMaxPcHist, ntwk.m_initOrMinApexDistHist, 
            ntwk.m_advPc, ntwk.m_recPc, ntwk.m_initedApexDist, ntwk.trappedW, ntwk.trappedNW, 
            ntwk.clusterW.pc, ntwk.clusterNW.pc, ntwk.clusterW_ID, _temp.tempID, 
            ntwk.areaSPhase, ntwk.maxCornerArea, ntwk.volarray, ntwk.sigma, ntwk.thetaAdvAng, ntwk.thetaRecAng, 
            ntwk.PcD, _temp.guessVol, _temp.volume, ntwk.satList, ntwk._delta, ntwk.isSquare, ntwk.isTriangle, 
            ntwk.muw, ntwk.MOLECULAR_LENGTH, True, False)

        self.pcMax[keys] = np.maximum(self.pcMax[keys],
            (_temp.guessPc[keys]+ntwk.imposedP)*volMax/self.volume[keys] - ntwk.imposedP)


    def adjustClusterPcShrink(self, keys, elem, ntwk):
        _temp.guessPc[keys] = ntwk.PcD[elem]
        _temp.mem[_temp.mem] = False
        for k in keys:
            _temp.mem[self[k].members] = True
        #_temp.mem[:] = self.members[keys].any(axis=0)
        _temp.mem[elem] = False
        memID = np.flatnonzero(_temp.mem).astype(np.int32)
        _temp.tempID[memID] = self.clusterID[memID]
    
        volShr = np.bincount(_temp.tempID[memID], ntwk.volarray[memID])[keys]
        self.pcShrink[keys] = np.minimum(self.pcShrink[keys],
            self.moles[keys]/volShr - ntwk.imposedP)


    

        
        