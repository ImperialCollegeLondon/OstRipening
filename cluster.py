import numpy as np
from joblib import dump, load, Parallel, delayed
from scipy import optimize
from numba import njit, prange, jit, types, typeof
from numba.experimental import jitclass
from numba.types import int32, float64, Array
from sortedcontainers import SortedList
from numba.typed import List

from pnflowPy.cluster import Cluster as clust
#import pnflowPy.tPhaseImb as tPhaseImb
import pnflowPy.utilities as do
from network_numba import ClusterJit, NetworkJit
from pnflowPy.cluster import ClusterObj 
import OstRipening.temp as temp
from utilities_numba import *


_temp = temp.TempArrays()

class Cluster(clust):
    def __init__(self, network, phase=1, nClusters=200):
        jit = ClusterJit(nClusters, network.totElements)
        self.phase = phase
        self.network = network
        self.keys = [0]
        self.values = [ClusterObj(0, self, network)]
        self.nClusters = nClusters
        self.availableID = SortedList()
        self.availableID.update(np.arange(1,nClusters))
        temp.TempArrays.formTempNetworkArrays(
            network.nPores, network.nThroats, network.totElements, 
            network.connectivity_graph_flat.size, network.m_halfAngles)
    
        if phase == 0:
            jit.hasFluid[1:-1] = (
                network.hasWFluid[1:-1] if hasattr(network, 'hasWFluid') else True)
        else:
            jit.hasFluid[1:-1] = (
                network.hasNWFluid[1:-1] if hasattr(network, 'hasNWFluid') else False)

        self.loadCluster(jit, network, True)


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
        
        #self.fluid = network.fluid.view()
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


    def resizeClusters(self, size):
        totElements = self.network.totElements
        self.members = np.vstack(
            (self.members, np.zeros((size, totElements), dtype=np.bool_)))
        self.neighbours = np.vstack(
            (self.neighbours, np.zeros((size, totElements), dtype=np.bool_)))

        self.pc = np.concatenate((self.pc, np.zeros(size)))
        self.pcShrink = np.concatenate((self.pcShrink, np.zeros(size)))
        self.pcMax = np.concatenate((self.pcMax, np.zeros(size)))
        self.volume = np.concatenate((self.volume, np.zeros(size)))
        self.volShrink = np.concatenate((self.volShrink, np.zeros(size)))
        self.volMax = np.concatenate((self.volMax, np.zeros(size)))
        self.moles = np.concatenate((self.moles, np.zeros(size)))
        self.molesShrink = np.concatenate((self.molesShrink, np.zeros(size)))
        self.molesMax = np.concatenate((self.molesMax, np.zeros(size)))
        
        self.trappedStatus = np.concatenate((self.trappedStatus, np.zeros(size, dtype=np.bool_)))
        self.connected = np.concatenate((self.connected, np.zeros(size, dtype=np.bool_)))
        self.size = np.concatenate((self.size, np.zeros(size, dtype=np.bool_)))
        for c in np.arange(self.nClusters, self.nClusters+size):
            self[c] = {'key': c, 'parent':self.network}
        emptyKeys = np.flatnonzero(self.size==0)
        emptyKeys = emptyKeys[emptyKeys > 0]
        self.availableID.update(emptyKeys)
        self.nClusters += size        
        
        self.totalVolume = np.concatenate((self.totalVolume, np.zeros(size)))
        self.toDrain = np.concatenate((self.toDrain, np.full(size, totElements+1, dtype=np.int32)))
        self.toImbibe = np.concatenate((self.toImbibe, np.full(size, totElements+1, dtype=np.int32)))

        self.valClust = np.concatenate((self.valClust, np.zeros(size, dtype=np.bool_)))
        self.valClustD = np.concatenate((self.valClustD, np.zeros(size, dtype=np.bool_)))

        try:
            assert self.phase==1
            temp.TempArrays.updateTempClustArrays(self.nClusters)
        except (AssertionError, AttributeError):
            pass
        

    def __neighbours__(self, network):  
        self.updateNeighMatrix(network)
    

    def updateNeighMatrix(self, network, cond=None):
        '''This updates the neighMatrix!!! might be later revised!!!'''
        # make it numba
        if cond is None:
            cond = np.ones(network.nThroats, dtype=np.bool_)

        while True:
            condT = cond[network.tValid-1]
            TValid = network.TValid[condT]
            TPValid = network.TPValid[condT]
            clustP = self.clusterID[TPValid]
            clustT = self.clusterID[TValid]

            condP = (clustP != clustT)
            condP_P = condP & (clustP >= 0)
            condP_T = condP & (clustT >= 0)
            condP_P_T = condP_P & condP_T
            if condP_P_T.any():
                arr = np.unique(np.sort(np.column_stack(
                    (clustP[condP_P_T], clustT[condP_P_T])), axis=1), axis=0)
                arr_1D = np.unique(arr)
                neigh = self.neighbours[arr_1D].any(axis=0)
                self.coalesceClusters(arr, self.clusterID, network)
                neigh |= self.members[arr_1D].any(axis=0)
                cond |= neigh[network.tList]
            else:
                keysToUpdate = np.unique(clustT[clustT>=0])
                break

        clustID = np.concatenate((clustP[condP_P],  clustT[condP_T]))
        neigh = np.concatenate((TValid[condP_P],  TPValid[condP_T]))
        #self.neighbours[keysToUpdate] = False
        self.neighbours[clustID, neigh] = True
        return


    def coalesceClusters(self, arr, cluster_ID, network):
        ''' coalesce clusters together '''
        arr = [*map(np.array, arr)]
        values, counts = np.unique(arr, return_counts=True)
        
        def _f1(c, ar, network):
            # compute new moles, volume and pc
            _mem = self.members[ar].any(axis=0)
            mem = network.elementListS[_mem]
            if _mem[network.conTToIn].any() and _mem[network.conTToOut].any():
                c=0
                ar = ar[ar!=0]
                self.moles[c] += self.moles[ar].sum()
                self.volume[c] += self.volume[ar].sum()
            else:
                self.moles[c] = self.moles[ar].sum()
                self.volume[c] = self.volume[ar].sum()
            
            pc = np.append(self.pc[c], self.pc[ar])
            pc = pc[pc>network.Pc]
            if pc.size==0:
                self.pc[c] = network.Pc
            else:
                self.pc[c] = pc[pc>network.Pc].min()
        
            mem1 = mem[cluster_ID[mem]!=c]
            clustID = cluster_ID[mem1]
            cluster_ID[mem1] = c
            self.members[clustID, mem1] = Falsed
            self.members[c, mem1] = True
            return c
        
        while counts.size>0:
            c = values[np.argmax(counts)]
            arrC = [ar[ar!=c][0] for ar in arr if c in ar]
            self.neighbours[arrC] = False
            arrC.append(c)
            c1 = _f1(c, np.array(arrC), network)
            if c1 in arrC: arrC.remove(c1)
            for c2 in arrC: del self[c2]
            arr = [ar for ar in arr if c not in ar]
            values, counts = np.unique(arr, return_counts=True)
        
        return


    def removeMembers(self, keys, toImbibe, network):
        ''' return the new pc after imbibition or drainage '''
        cond = (self.size[keys]==1)
        keys1, toImbibe1 = keys[cond], toImbibe[cond]
        network.aqueousMoles[toImbibe1] = self.moles[keys1]

        notcond = ~cond
        keys2, toImbibe2 = keys[notcond], toImbibe[notcond]
        network.aqueousMoles[toImbibe2] = np.minimum(self.moles[keys2],
            network.gasConc[toImbibe2,0]*network.volarray[toImbibe2])

        network.gasConc[toImbibe,1] = network.aqueousMoles[toImbibe]/network.volarray[toImbibe]

        self.moles[keys] -= network.aqueousMoles[toImbibe]
        for i in range(toImbibe.size):
            self.neighbours[keys[i], self[keys[i]].neighbours] = False
            self.unfill_phase(toImbibe[i], self.pc[keys[i]])


    def pcAfterShrinkage(self, keys, toImbibe, network):
        ''' return the new pc after imbibition or drainage '''
        term = (self.volume[keys] - network.volarray[toImbibe])/network.RT
        num = self.moles[keys] - network.imposedP*term
        den = term + network.H*network.volarray[toImbibe]
        return np.maximum(np.divide(num, den, where=(den!=0.0)), network.Pc)


    def addMembers(self, keys, toDrain, network):
        clusterW = network.clusterW
        clusterW_ID = clusterW.clusterID
        mem, neigh = _temp.mem, _temp.memThroat
        mem[mem] = False
        #oldKeys = self.clusterID.copy()
        oldKeys = _temp.tempID
        oldKeys[:] = self.clusterID
        
        for i in range(toDrain.size):
            ii, k = toDrain[i], keys[i]
            if self.hasFluid[ii]: continue
            moles = self.moles[k] + network.aqueousMoles[ii]
            network.aqueousMoles[ii] = 0.0

            if network.isCircle[ii]:
                clusterW.unfill_phase(ii, clusterW.pc[clusterW_ID[ii]])

            if not network.isPore[ii]:
                conP = network.elem[ii].neighbours
                conP = conP[(conP>0) & (network.Rarray[conP] < network.Rarray[ii]) & ~self.hasFluid[conP]]
                if conP.any():
                    conP = int(conP[0])
                    network.fluid[conP] = self.phase
                    self.hasFluid[conP] = True
                    network.elemToUpdateW[conP] = False
                    network.elemToUpdateNW[conP] = True
                    moles += network.aqueousMoles[conP]
                    network.aqueousMoles[conP] = 0.0
                    if network.isCircle[conP]:
                        clusterW.unfill_phase(conP, clusterW.pc[clusterW_ID[conP]])

            self.fill_with_phase(ii, self.pc[k], network)
            newK = self.clusterID[ii]
            memID = self[newK].members
            mem[memID] = True
            memKeys = oldKeys[memID]
            memKeys = np.unique(memKeys[(memKeys>=0)&(memKeys!=k)])
            oldKeys[memID] = newK
            self.moles[newK] = moles + self.moles[memKeys].sum()
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
        neigh[:] = mem[network.tList]
        self.neighbours[:, _mem] = False

        return _mem, neigh


    # def updateVolumeNWCluster(self, elemID, valkeys, ntwk):
    #     ''' updating volume for NW cluster '''
    #     self.volume[valkeys]  = np.bincount(self.clusterID[elemID], ntwk.volarray[elemID])[valkeys]
    #     self.volMax[valkeys] = self.volume[valkeys]
    #     self.volShrink[valkeys] = self.volume[valkeys] - ntwk.volarray[self.toImbibe[valkeys]]


    # def updateMolesNWCluster(self, valkeys, ntwk):
    #     ''' updating moles for  NW cluster '''
    #     #print('Im in updateMolesNWCluster!!!')
    #     #from IPython import embed; embed()
    #     self.molesMax[valkeys] = np.minimum(
    #         (ntwk.imposedP + self.pcMax[valkeys])*self.volMax[valkeys]/ntwk.RT,
    #         ntwk.imposedP*(self.volMax[valkeys]+ntwk.volarray[self.toDrain[valkeys]])/ntwk.RT
    #         )
    #     #self.molesMax[valkeys] = ntwk.imposedP*(self.volMax[valkeys]+ntwk.volarray[self.toDrain[valkeys]])/ntwk.RT
        
    #     #from IPython import embed; embed()
    #     #self.molesMax[valkeys] = (ntwk.imposedP + self.pcMax[valkeys])*self.volMax[valkeys]/ntwk.RT
    #     self.molesShrink[valkeys] = np.maximum(
    #         (ntwk.imposedP + self.pcShrink[valkeys])*self.volShrink[valkeys]/ntwk.RT,
    #         (ntwk.imposedP + ntwk.oldaqAvgPres)*self.volShrink[valkeys]/ntwk.RT)


    def disappearCluster(self, network, pc_min, pc_max):   
        cond = _temp.resCond
        cond[:] = (self.size==0)
        cond[self.availableID] = False
        cond[0] = False  
        keys = np.flatnonzero(cond).astype(np.int32)
        if not keys.any(): return
        self.neighbours[keys] = False
        for k in keys:
            del self[k]
            self.molesShrink[k], self.molesMax[k] = 0.0, 1e30
            self.volShrink[k] = self.volMax[k] = 0.0
            self.pcShrink[k], self.pcMax[k] = 0.0, 1e30
        updateToDrainImbibe_numba(keys, self.members, self.neighbours, self.toDrain, 
            self.toImbibe, self.pcShrink, self.pcMax, network.PcI, network.PcD, 
            network.Rarray, network.totElements, _temp.mem, True, pc_min, pc_max)
        keys = np.setdiff1d(keys, self.availableID)
        keys = keys[keys!=0]
        self.availableID.update(keys)

    
    def shrinkCluster(self, keys, network, moles_tol=1e-23, pc_tol=1e-4, 
                    max_iter=50, pc_min=1e-30, pc_max=1e30):
        ''' update toImbibe element(s) !!! '''
        # fill with the other phase
        
        clust = network.clusterW if self.phase==1 else network.clusterNW
        for k in keys:
            clust.fill_with_phase(self.toImbibe[k], self.pc[k], network)
    
        # clusters to shrink
        toImbibe = self.toImbibe[keys]
        network.satList[toImbibe] = 1.0
        mem = self.members[keys].any(axis=0)
        neigh = mem|self.neighbours[keys].any(axis=0)
        mem[toImbibe] = False
        _mem = np.flatnonzero(mem).astype(np.int32)
        oldMemKeys = self.clusterID[_mem]

        self.removeMembers(keys, toImbibe, network)
        self.updateNeighMatrix(network, neigh[network.tList])
        newMemKeys = self.clusterID[_mem]

        if newMemKeys.size > 0:
            if np.all(oldMemKeys==newMemKeys):      #case1
                updateToDrainImbibe_numba(keys, self.members, self.neighbours, self.toDrain, 
                    self.toImbibe, self.pcShrink, self.pcMax, network.PcI, network.PcD, 
                    network.Rarray, network.totElements, _temp.mem, True, pc_min, pc_max)
                self.valClustD[:] = self.valClust & (self.toDrain<=network.totElements) #revise
                self.updateClusterPcVolume(
                    keys, newMemKeys, _mem, self.moles, network, moles_tol, pc_tol, max_iter)
                cond = (self.pcShrink[keys] < network.oldaqAvgPres)
                self.pcShrink[keys[cond]] = 1e-3

                cond = (toImbibe==self.toDrain[keys])
                if cond.any():
                    spKeys, spElem = keys[cond], toImbibe[cond]
                    arrr = (spElem > network.nPores)
                    if arrr.any():
                        spElemT, spKeysT = spElem[arrr], spKeys[arrr]
                        conP = network.TPConnections[spElemT-network.nPores]
                        condP = (conP>0) & ~self.hasFluid[conP]
                        if condP.any():
                            spKeys = np.append(spKeys, spKeysT[condP.any(axis=1)])
                            spElem = np.append(spElem, conP[condP])

                    otherKeys, ind  =  np.where(self.neighbours[:, spElem])
                    if otherKeys.any():
                        cond = (otherKeys != spKeys[ind]) & (
                            network.Rarray[self.toDrain[spKeys[ind]]]<network.Rarray[self.toImbibe[otherKeys]])
                        self.adjustClusterPcMax(spKeys, spElem, network, otherKeys[cond], ind[cond])
                    else:
                        self.adjustClusterPcMax(spKeys, spElem, network)
                
                network.gasConc[_mem, 1] = network.H*self.pc[newMemKeys]
            else:                                                     #case2
                _oldMemKeys, _newMemKeys = mapOldNewKeys(oldMemKeys, newMemKeys, _temp.resCond)
                oldToImbibe = self.toImbibe[_oldMemKeys]
                updateToDrainImbibe_numba(_newMemKeys, self.members, self.neighbours, self.toDrain, 
                    self.toImbibe, self.pcShrink, self.pcMax, network.PcI, network.PcD, 
                    network.Rarray, network.totElements, _temp.mem, True, pc_min, pc_max)
                self.valClustD[:] = self.valClust & (self.toDrain<=network.totElements) #revise
                self.distributeMoles(_newMemKeys, newMemKeys, _oldMemKeys, _mem, network)
                self.updateClusterPcVolume(
                    _newMemKeys, newMemKeys, _mem, self.moles, network, moles_tol, pc_tol, max_iter)
                cond = (self.pcShrink[_newMemKeys] < network.oldaqAvgPres)
                self.pcShrink[_newMemKeys[cond]] = 1e-3

                cond = (oldToImbibe==self.toDrain[_newMemKeys])
                if cond.any():
                    spKeys, spElem = _newMemKeys[cond], oldToImbibe[cond]
                    arrr = (spElem > network.nPores)
                    if arrr.any():
                        spElemT, spKeysT = spElem[arrr], spKeys[arrr]
                        conP = network.TPConnections[spElemT-network.nPores]
                        condP = (conP>0) & (network.Rarray[conP] < network.Rarray[self.toImbibe[spKeysT]][:,np.newaxis]) & ~self.hasFluid[conP]
                        if condP.any():
                            spKeys = np.append(spKeys, spKeysT[condP.any(axis=1)])
                            spElem = np.append(spElem, conP[condP])

                    otherKeys, ind  =  np.where(self.neighbours[:, spElem])
                    if otherKeys.any():
                        cond = (otherKeys != spKeys[ind]) & (
                            network.Rarray[self.toDrain[spKeys[ind]]]<network.Rarray[self.toImbibe[otherKeys]])
                        self.adjustClusterPcMax(spKeys, spElem, network, otherKeys[cond], ind[cond])
                    else:
                        self.adjustClusterPcMax(spKeys, spElem, network)

                network.gasConc[_mem,1] = network.H*self.pc[newMemKeys] #

        self.disappearCluster(network, pc_min, pc_max)
        if ((self.toImbibe>network.totElements) & (self.size>0)).any():
            print(' there is an issue with toimbibe and todrain, check!!!')
            from IPython import embed; embed()


    def growCluster(self, keys, network, moles_tol=1e-23, pc_tol=1e-4, max_iter=50, 
                    pc_min=1.0e-30, pc_max=1.0e30):
        ''' update toDrain element(s) !!! '''
        toDrain = self.toDrain[keys]
        _mem, neigh = self.addMembers(keys, toDrain, network)
        self.disappearCluster(network, pc_min, pc_max)
        self.updateNeighMatrix(network, neigh)
        
        memkeys = self.clusterID[_mem]
        toDrainKeys = np.unique(memkeys)
        updateToDrainImbibe_numba(toDrainKeys, self.members, self.neighbours, self.toDrain, 
            self.toImbibe, self.pcShrink, self.pcMax, network.PcI, network.PcD, 
            network.Rarray, network.totElements, _temp.mem, True, pc_min, pc_max)
        self.valClustD[:] = self.valClust & (self.toDrain<=network.totElements)
        self.updateClusterPcVolume(
            toDrainKeys, memkeys, _mem, self.moles, network, moles_tol, pc_tol, max_iter)

        cond = (self.pcShrink[toDrainKeys] < network.oldaqAvgPres)
        self.pcShrink[toDrainKeys[cond]] = 1e-3

        # elem just drained and are candidate of next shrinkage
        spElem = np.intersect1d(toDrain, self.toImbibe[toDrainKeys])
        if spElem.any():
            spKeys = self.clusterID[spElem]
            self.adjustClusterPcShrink(spKeys, spElem, network)

            # otherKeys, ind  =  np.where(self.neighbours[:, self.toDrain[spKeys]])
            # if otherKeys.any():
            #     cond = (otherKeys != spKeys[ind]) & (
            #         network.Rarray[self.toDrain[spKeys[ind]]]<network.Rarray[self.toImbibe[otherKeys]])
            #     if cond.any():
            #         self.adjustClusterPcMax(spKeys, spElem, network, otherKeys[cond], ind[cond])

        network.gasConc[_mem,1] = network.H*self.pc[memkeys] #
        self.trapped[_mem] = self.trappedStatus[memkeys]
        network.satList[toDrain] = 0.0
        
        return toDrain


    def updateClusterPcVolume(self, keys, memkeys, memID, targetMoles, ntwk,
        moles_tol, pc_tol, max_iter):

        findPcVolume(keys, memkeys, memID, targetMoles, ntwk.m_halfAngles, ntwk.m_cornExists, 
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

        _, moleShr = returnVolMoles(keys1, memkeys1, memID1, self.pcShrink, ntwk.m_halfAngles, 
            ntwk.m_cornExists, ntwk.m_initOrMaxPcHist, ntwk.m_initOrMinApexDistHist,
            ntwk.m_advPc, ntwk.m_recPc, ntwk.m_initedApexDist, ntwk.trappedW, ntwk.trappedNW, 
            ntwk.clusterW.pc, ntwk.clusterNW.pc, ntwk.clusterW_ID, ntwk.clusterNW_ID, 
            ntwk.areaSPhase, ntwk.maxCornerArea, ntwk.volarray, ntwk.sigma, ntwk.thetaAdvAng, ntwk.thetaRecAng, 
            ntwk.PcD, ntwk.satList, _temp.guessVol, _temp.volume, ntwk._delta, ntwk.isSquare, ntwk.isTriangle, 
            _temp.area, ntwk.muw, ntwk.MOLECULAR_LENGTH, ntwk.imposedP, ntwk.RT)

        _, moleMax = returnVolMoles(keys1, memkeys1, memID1, self.pcMax, ntwk.m_halfAngles, 
            ntwk.m_cornExists, ntwk.m_initOrMaxPcHist, ntwk.m_initOrMinApexDistHist,
            ntwk.m_advPc, ntwk.m_recPc, ntwk.m_initedApexDist, ntwk.trappedW, ntwk.trappedNW, 
            ntwk.clusterW.pc, ntwk.clusterNW.pc, ntwk.clusterW_ID, ntwk.clusterNW_ID, 
            ntwk.areaSPhase, ntwk.maxCornerArea, ntwk.volarray, ntwk.sigma, ntwk.thetaAdvAng, ntwk.thetaRecAng, 
            ntwk.PcD, ntwk.satList, _temp.guessVol, _temp.volume, ntwk._delta, ntwk.isSquare, ntwk.isTriangle, 
            _temp.area, ntwk.muw, ntwk.MOLECULAR_LENGTH, ntwk.imposedP, ntwk.RT)

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
        _temp.mem[:] = self.members[keys].any(axis=0)
        _temp.mem[elem] = True
        extra = otherKeys is not None and otherKeys.size > 0
        if extra:
            ind, mem = np.where(self.members[otherKeys])
            _temp.mem[mem] = True
        memID = np.flatnonzero(_temp.mem).astype(np.int32)
        _temp.tempID[memID] = self.clusterID[memID]
        _temp.tempID[elem] = keys
        if extra:
            _temp.tempID[mem] = keys[shift_ind[ind]]

        keys = np.unique(keys)
        _temp.guessPc[keys] = 1e-3
        np.maximum.at(_temp.guessPc, _temp.tempID[memID], ntwk.PcI[memID])
        volMax = computeVolume_numba(keys, _temp.tempID[memID], memID, _temp.guessPc, _temp.area, 
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
        _temp.mem[:] = self.members[keys].any(axis=0)
        _temp.mem[elem] = False
        memID = np.flatnonzero(_temp.mem).astype(np.int32)
        _temp.tempID[memID] = self.clusterID[memID]
    
        volShr = computeVolume_numba(keys, _temp.tempID[memID], memID, _temp.guessPc, _temp.area, 
            ntwk.m_halfAngles, ntwk.m_cornExists, ntwk.m_initOrMaxPcHist, ntwk.m_initOrMinApexDistHist, 
            ntwk.m_advPc, ntwk.m_recPc, ntwk.m_initedApexDist, ntwk.trappedW, ntwk.trappedNW, 
            ntwk.clusterW.pc, ntwk.clusterNW.pc, ntwk.clusterW_ID, _temp.tempID, 
            ntwk.areaSPhase, ntwk.maxCornerArea, ntwk.volarray, ntwk.sigma, ntwk.thetaAdvAng, ntwk.thetaRecAng, 
            ntwk.PcD, _temp.guessVol, _temp.volume, ntwk.satList, ntwk._delta, ntwk.isSquare, ntwk.isTriangle, 
            ntwk.muw, ntwk.MOLECULAR_LENGTH, True, False)
        
        self.pcShrink[keys] = np.minimum(self.pcShrink[keys],
            (_temp.guessPc[keys]+ntwk.imposedP)*volShr/self.volume[keys] - ntwk.imposedP)

 

        

        
        