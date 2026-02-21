import numpy as np
from time import time
import math
import os
import dill

import OstRipening.temp as temp
from OstRipening.mutilities_numba import *
from OstRipening.mcluster import *
import pnflowPy.tPhaseD as tPhaseD
import pnflowPy.tPhaseImb as tPhaseImb

from IPython import embed
a = 5070

_temp = temp.TempArrays()
MEMORY_DIR = f"equilibrium_results/test/bentSepi600_aqAvgPresAvgClustPc_alpha0dot56"
os.makedirs(MEMORY_DIR, exist_ok=True)
_a, alpha = 0, 0.56

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
    #from IPython import embed; embed()
    


def equilibrate(self, pc_min=1.0e-30, pc_max=1.0e30):
    cNWP = self.cNWP
    cWP = self.cWP
    cond = np.flatnonzero(cNWP.valClustD)
    
    
    
    cNWP.pc[cond] = alpha*cNWP.pcMax[cond]+(1-alpha)*cNWP.pc[cond]
    #cNWP.pc[cond] = np.maximum(alpha*cNWP.pcMax[cond]+(1-alpha)*cNWP.pcShrink[cond], cNWP.pc[cond])
    
    #cond = (cNWP.sizes>0)
    #condG = (cond &(cNWP.pcMax<a))
    #condS = (cond &(cNWP.pcShrink>a))
    #cNWP.pc[condG] = alpha*cNWP.pcMax[condG]+(1-alpha)*cNWP.pc[condG]
    #cNWP.pc[condS] = alpha*cNWP.pcShrink[condS]+(1-alpha)*cNWP.pc[condS]
    
    
    self.aqAvgPres = (cNWP.pc*cNWP.volume).sum()/cNWP.volume.sum() + _a
    #self.aqAvgPres = a
    keysToUpdate = _temp.resCond
    cNWP.drainEvents, cNWP.imbEvents = 0, 0
    initialSw = self.satW
    self.validToShrink = cNWP.valClust & (cNWP.pcShrink>self.aqAvgPres)
    self.validToGrow = cNWP.valClustD & (cNWP.pcMax<self.aqAvgPres)
    print(f'initial Sw = {self.satW} avgPres = {self.aqAvgPres}')
    writeData(self, cNWP)
    
    #from IPython import embed; embed()
    cnt = 0
    while True:
        cnt += 1
        condG = (cNWP.valClustD & (cNWP.pc <= self.aqAvgPres) & (cNWP.pcMax <= self.aqAvgPres))
        if condG.any():
            keys = np.flatnonzero(condG).astype(np.int32)
            toDrain = cNWP.toDrain[keys]
            print('$$:  ', keys, toDrain)
            cNWP.drainEvents += keys.size
            #from IPython import embed; embed()
            keys_to_update, _mem = addMembers(self, keys, toDrain, cNWP)
            cNWP.updateClusterProperties(keys_to_update, self, 1e-3, 1e30, False)
            
            tPhaseD.__CondTP_Drainage__(self, cNWP.pc[cNWP.clusterID].astype(np.float64))
            self.satList[_mem] = self._cornArea[_mem]/self.areaSPhase[_mem]

        condS = (cNWP.valClust & (cNWP.pc > self.aqAvgPres) & (cNWP.pcShrink > self.aqAvgPres))
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
                
            _mem = np.flatnonzero(mem).astype(np.int32)
            cNWP.updateClusterProperties(keys_to_update, self, 1e-3, 1e30, True)
            tPhaseImb.__CondTPImbibition__(self, mem, cNWP.pc[cNWP.clusterID].astype(np.float64), True, True)
            self.satList[_mem] = self._areaWP[_mem]/self.areaSPhase[_mem]
            
            # print('????????????????????/')
            # from IPython import embed; embed()
                
            
                
        
            # clusters to shrink
            # toImbibe = cNWP.toImbibe[keys]
            # print('@@:  ', keys, toImbibe)
            # cNWP.imbEvents += keys.size
            # self.satList[toImbibe] = 1.0

            # mem, neigh = _temp.mem, _temp.done
            # mem[mem] = False
            # neigh[neigh] = False
            # for k in keys:
                # mem[cNWP[k].members] = True
                # neigh[cNWP[k].neighbours] = True

            # #mem = cNWP.members[keys].any(axis=0)
            # #neigh = mem|cNWP.neighbours[keys].any(axis=0)
            # neigh |= mem
            # mem[toImbibe] = False
            # _mem = np.flatnonzero(mem).astype(np.int32)
            # oldMemKeys = cNWP.clusterID[_mem]

            # removeMembers(self, keys, toImbibe, cNWP)
            # #cNWP.updateNeighMatrix(self, neigh[self.tList])
            # cNWP.updateClusterNeighbours(np.unique(cNWP.clusterID[_mem]), self)
            # newMemKeys = np.unique(cNWP.clusterID[_mem])
            # updateToDrainImbibe_numba(newMemKeys, cNWP.members, cNWP.members_offset, cNWP.size,
                # cNWP.neighbours, cNWP.toDrain, cNWP.toImbibe, cNWP.pcShrink, cNWP.pcMax, 
                # self.PcI, self.PcD, self.Rarray, self.totElements, _temp.mem, True, pc_min, pc_max)
            # cNWP.valClustD[:] = cNWP.valClust & (cNWP.toDrain<=self.totElements) #revise
            # cNWP.disappearCluster(self, pc_min, pc_max)
            # cNWP.valClust[:] = (cNWP.size>0)
            # cNWP.valClustD[:] = cNWP.valClust & (cNWP.toDrain<=self.totElements)

            # tPhaseImb.__CondTPImbibition__(self, mem, cNWP.pc[cNWP.clusterID], True, True)
            # self.satList[_mem] = self._areaWP[_mem]/self.areaSPhase[_mem]
        
        if not (condG.any() or condS.any()):
            break

        self.satW = ((self.satList[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()/
                        self.totVoidVolume)
        valkeys = np.flatnonzero(cNWP.valClust).astype(np.int32)
        cNWP.volume[valkeys] = np.bincount(
            cNWP.clusterID[cNWP.hasFluid], (1-self.satList[cNWP.hasFluid])*self.volarray[cNWP.hasFluid])[valkeys]

        #input('waittt!!!!')
        writeData(self, cNWP)

        #for k in range(cNWP.nClusters):
        #    if (cNWP.sizes[k]==0):
        #       if (cNWP[k].neighbours.size!=0):
        #            print(k, cNWP[k].neighbours)
        #    else:
        #        #neigh = np.unique(np.concatenate(self.connectivity_graph[cNWP[k].members]))
        #        #from IPython import embed; embed()
        #        neigh = np.unique(np.concatenate(self.connectivity_graph[cNWP[k].members]))
        #        nonMem = neigh[cNWP.clusterID[neigh]!=k]
        #        if cNWP.hasFluid[nonMem].any() or (nonMem.size != cNWP[k].neighbours.size):
        #            print(k, cNWP[k].neighbours, neigh)
        #            input('waiittttttttttttttt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
        print('\n\n')
        
    #print('~~~~~~~########~~~~~~~~~~~~')
    #from IPython import embed; embed()
    # satListI = self.satList.copy()
    # self.capPresMin = self.aqAvgPres
    # tPhaseImb.__CondTPImbibition__(self, overrideTrapping=True)
    # satListI[1:-1] = self._areaWP[1:-1]/self.areaSPhase[1:-1]
    # satW_I = ((satListI[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()/self.totVoidVolume)

    # satListD = self.satList.copy()
    # self.capPresMax = self.aqAvgPres
    # tPhaseD.__CondTP_Drainage__(self)
    # satListD[1:-1] = self._cornArea[1:-1]/self.areaSPhase[1:-1]
    # satListD[self.fluid==0] = 1.0
    # satW_D = ((satListD[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()/self.totVoidVolume)


    #print(f'@alpha={alpha}, satW_I={satW_I}, satW_D={satW_D}, previous_satW={self.satW}')
    print(f'{cnt}s @alpha={alpha}, initial_sat={initialSw}, final_sat={self.satW}')

    print('Im done !!!')
    from IPython import embed; embed()


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

       



