import numpy as np
from time import time
import math
import os
import dill

from . import temp
from .utilities_numba import *
from .cluster import *




_temp = temp.TempArrays()


def initialize(self, pc_min=1.0e-30, pc_max=1.0e30):
    print('---------------------------------------------------------------------------')
    print('-------------------------Equilibrium Model---------------------------------')
    
    cWP = self.cWP
    cNWP = self.cNWP
    
    # re-size the clusters
    cNWP.resizeClusters(0, ostMode=True)
    cWP.resizeClusters(0, ostMode=True)
        
    
    nClust = cNWP.nClusters
    keys = np.arange(nClust).astype(np.int32)
    self.entryPress_freezed = np.zeros(self.totElements, dtype=np.bool_)
    cNWP.updateClusterProperties(keys, self, 1e-3, 1e30, False)
    self.satList = np.zeros(self.totElements)
    self.satList[1:-1] = self.areaWPhase[1:-1]/self.areaSPhase[1:-1]
    updateClusterVolume(cNWP.hasFluid, cNWP.clusterID, cNWP.volume, 
        self.satList, self.volarray, self.totElements)

    print('done with initialization !!!')
        


def equilibrate(self, pc_min=1.0e-30, pc_max=1.0e30):
    cNWP = self.cNWP
    cWP = self.cWP
    
    Pc = cNWP.pc[cNWP.clusterID].astype(np.float64)
    arrr = np.ones(self.totElements, dtype=bool)
    arrr[[-1,0]] = False
    do.update_areas_conductances(self, arrr, Pc, False, True, True)
    hasFluid = cNWP.hasFluid
    self.satList[hasFluid] = self._cornArea[hasFluid]/self.areaSPhase[hasFluid]
    cNWP.volume[:] = np.bincount(cNWP.clusterID[hasFluid], (1-self.satList[hasFluid])*self.volarray[hasFluid], cNWP.nClusters)
    
    cond = np.flatnonzero(cNWP.valClustD)
    aqAvgPres0 = (cNWP.pc*cNWP.volume).sum()/cNWP.volume.sum()
    cPc = cNWP.pc.copy()
    cVol = cNWP.volume.copy()
    satList = self.satList.copy()
    
    cNWP.pc[cond] = alpha*cNWP.pcMax[cond]+(1-alpha)*cNWP.pc[cond]
    arrr = np.ones(self.totElements, dtype=bool)
    arrr[[-1,0]] = False
    adjustVolume = True
    
    if adjustVolume:
        Pc = cNWP.pc[cNWP.clusterID].astype(np.float64)
        do.update_areas_conductances(self, arrr, Pc, False, True, True)
        self.satList[hasFluid] = self._cornArea[hasFluid]/self.areaSPhase[hasFluid]
        cNWP.volume[:] = np.bincount(cNWP.clusterID[hasFluid], (1-self.satList[hasFluid])*self.volarray[hasFluid], cNWP.nClusters)
       
    
    self.aqAvgPres = (cNWP.pc*cNWP.volume).sum()/cNWP.volume.sum()
    keysToUpdate = _temp.resCond
    cNWP.drainEvents, cNWP.imbEvents = 0, 0
    self.satW = initialSw = (self.satList*self.volarray*self.isinsideBox).sum()/self.volarray[self.isinsideBox].sum()
    self.validToShrink = cNWP.valClust & (cNWP.pcShrink>self.aqAvgPres)
    self.validToGrow = cNWP.valClustD & (cNWP.pcMax<self.aqAvgPres)
    print(f'initial Sw = {self.satW} avgPres = {self.aqAvgPres}')
    writeData(self, cNWP)
    
    cnt = 0
    while True:
        cnt += 1
        condG = (cNWP.valClustD & (cNWP.pc <= self.aqAvgPres) & (cNWP.pcMax <= self.aqAvgPres))
        if condG.any():
            keys = np.flatnonzero(condG).astype(np.int32)
            toDrain = cNWP.toDrain[keys]
            print('$$:  ', keys, toDrain)
            cNWP.drainEvents += keys.size
            #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#')
            #from IPython import embed; embed()
            keys_to_update, _mem = addMembers(self, keys, toDrain, cNWP)
            
            cNWP.updateClusterProperties(keys_to_update, self, 1e-3, 1e30, False)
            
            Pc = cNWP.pc[cNWP.clusterID].astype(np.float64)
            self.is_oil_inj = True
            mem = np.zeros(self.totElements, dtype=np.bool_)
            mem[_mem] = True
            do.update_areas_conductances(self, mem, Pc, False, True, True)
            self.satList[_mem] = self._cornArea[_mem]/self.areaSPhase[_mem]
            
            

        condS = cNWP.valClust & (((cNWP.pc > self.aqAvgPres) & (cNWP.pcShrink > self.aqAvgPres)))
        if condS.any():
            keys = np.flatnonzero(condS).astype(np.int32)
            print('@@:  ', keys, cNWP.toImbibe[keys])
            cond = np.zeros(cNWP.nClusters, dtype=np.bool_)
            cond[keys] = True
            
            for k in keys:
                toImb = cNWP.toImbibe[k]
                cWP.fill_with_phase(toImb, cNWP.pcShrink[k], self)
                cNWP.unfill_phase(toImb, self.PcI[toImb])
                cNWP.imbEvents += 1
                self.satList[toImb] = 1.0
                neigh = self.connectivity_graph[toImb]
                cids = cNWP.clusterID[neigh]
                cids = cids[cids>=0]
                cond[cids] = True
                
            keys_to_update = np.flatnonzero(cond).astype(np.int32)
            mem = np.zeros(self.totElements, dtype=np.bool_)
            for k in keys_to_update:
                mem[cNWP[k].members] = True
            mem &= cNWP.hasFluid
            _mem = np.flatnonzero(mem).astype(np.int32)
            cNWP.updateClusterProperties(keys_to_update, self, 1e-3, 1e30, True)
            Pc = cNWP.pc[cNWP.clusterID].astype(np.float64)
            self.is_oil_inj = False
            do.update_areas_conductances(self, mem, Pc, False, True, True)
            self.satList[mem] = self._cornArea[mem]/self.areaSPhase[mem]
            
            
        
        if not (condG.any() or condS.any()):
            break
            
       
        self.satW = ((self.satList[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()/
                        self.totVoidVolume)
        valkeys = np.flatnonzero(cNWP.valClust).astype(np.int32)
        cNWP.volume[valkeys] = np.bincount(
            cNWP.clusterID[cNWP.hasFluid], (1-self.satList[cNWP.hasFluid])*self.volarray[cNWP.hasFluid])[valkeys]

        writeData(self, cNWP)
        
        print('\n\n')
        
    print(f'{cnt}s @alpha={alpha}, initial_sat={initialSw}, final_sat={self.satW}')

    print('Im done !!!')


def addMembers(self, keys, toDrain, cNWP):
    cWP = self.cWP    
    keys_to_update = np.zeros(cNWP.nClusters, dtype=np.bool_)
    mem = np.zeros(self.totElements, dtype=np.bool_)
    keys_to_update[keys] = True
    
    for i in range(toDrain.size):
        ii, k = toDrain[i], keys[i]
        
        if cNWP.hasFluid[ii]:
            continue
        
        if self.isCircle[ii]:
            cWP.unfill_phase(ii, self.PcD[ii])

        cNWP.fill_with_phase(ii, self.PcD[ii], self)
        cid = cNWP.clusterID[ii]
        keys_to_update[cid] = True
        mem[cNWP[cid].members] = True
    
    return np.flatnonzero(keys_to_update).astype(np.int32), np.flatnonzero(mem)
        


def removeMembers(self, keys, toImbibe, cNWP):
    for i in range(toImbibe.size):
        k, toImb = keys[i], toImbibe[i]
        cNWP.unfill_phase(toImb, self.PcI[toImb])


def writeData(self, clust):
    with open(os.path.join(MEMORY_DIR, 'clustPcOstRipening_bent.dat'), 'a') as f1,\
            open(os.path.join(MEMORY_DIR, 'clustVolOstRipening_bent.dat'), 'a') as f2,\
            open(os.path.join(MEMORY_DIR, 'clustIDOstRipening_bent.dat'), 'a') as f4,\
            open(os.path.join(MEMORY_DIR, 'numberEventsOstRipening_bent.dat'), 'a') as f7,\
            open(os.path.join(MEMORY_DIR, 'saturationOstRipening_bent.dat'), 'a') as f8,\
            open(os.path.join(MEMORY_DIR, 'clustSizeOstRipening_bent.dat'), 'a') as f9,\
            open(os.path.join(MEMORY_DIR, 'saturation_by_element_OstRipening_bent.dat'), 'a') as f10:
       
        np.savetxt(f1, [clust.pc], delimiter=',', fmt='%g')
        np.savetxt(f2, [clust.volume], delimiter=',', fmt='%g')
        np.savetxt(f4, [clust.clusterID], delimiter=',', fmt='%g')
        np.savetxt(f7, [[clust.imbEvents, clust.drainEvents]], delimiter=',', fmt='%g')
        f8.write(str(self.satW)+',')
        np.savetxt(f9, [clust.sizes], delimiter=',', fmt='%g')
        np.savetxt(f10, [self.satList], delimiter=',', fmt='%g')

       



