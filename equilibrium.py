import numpy as np
from time import time
import math
import os
import dill

import OstRipening.temp as temp
from OstRipening.utilities_numba import *
from OstRipening.cluster import *
import pnflowPy.tPhaseD as tPhaseD
import pnflowPy.tPhaseImb as tPhaseImb

_temp = temp.TempArrays()
MEMORY_DIR = f"equilibrium_results/1/bent_aqAvgPresAvgClustPc_alpha1dot0"
os.makedirs(MEMORY_DIR, exist_ok=True)
_a, alpha = 0, 1.0

def initialize(self, pc_min=1.0e-30, pc_max=1.0e30):
    print('---------------------------------------------------------------------------')
    print('-------------------------Equilibrium Model---------------------------------')
    
    clustW = self.clusterW
    clustNW = self.clusterNW

    # re-size the clusters
    clustNW.resizeClusters(0, ostMode=True)
    clustW.resizeClusters(0, ostMode=True)
    
    # re-initialize the clusters   
    clustW.__neighbours__(self)
    clustNW.__neighbours__(self)

    # set up some arrays
    clustNW.valClust[:] = (clustNW.size>0)
    valkeys = np.flatnonzero(clustNW.valClust).astype(np.int32)
    updateToDrainImbibe_numba(valkeys, clustNW.members, clustNW.members_offset, clustNW.size, clustNW.neighbours, 
        clustNW.toDrain, clustNW.toImbibe, clustNW.pcShrink, clustNW.pcMax, self.PcI, self.PcD, self.Rarray,
        self.totElements, _temp.mem, True, pc_min, pc_max)
    clustNW.valClustD[:] = clustNW.valClust & (clustNW.toDrain<=self.totElements)
    
    self.satList = np.zeros(self.totElements)
    self.satList[1:-1] = self.areaWPhase[1:-1]/self.areaSPhase[1:-1]
    clustNW.volume[valkeys] = np.bincount(
        clustNW.clusterID[self.hasNWFluid], (1-self.satList[self.hasNWFluid])*self.volarray[self.hasNWFluid])[valkeys]

    print('done with initialization !!!')
    #from IPython import embed; embed()
    


def equilibrate(self, pc_min=1.0e-30, pc_max=1.0e30):
    clust = self.clusterNW
    clustW = self.clusterW
    cond = np.flatnonzero(clust.valClustD)
    
    clust.pc[cond] = alpha*clust.pcMax[cond]+(1-alpha)*clust.pc[cond]
    #clust.pc[cond] = np.maximum(alpha*clust.pcMax[cond]+(1-alpha)*clust.pcShrink[cond], clust.pc[cond])
    self.aqAvgPres = (clust.pc*clust.volume).sum()/clust.volume.sum() + _a
    keysToUpdate = _temp.resCond
    clust.drainEvents, clust.imbEvents = 0, 0
    initialSw = self.satW
    self.validToShrink = clust.valClust & (clust.pcShrink>self.aqAvgPres)
    self.validToGrow = clust.valClustD & (clust.pcMax<self.aqAvgPres)
    print(f'initial Sw = {self.satW}')
    writeData(self, clust)

    while True:
        condG = (clust.valClustD & (clust.pc <= self.aqAvgPres) & (clust.pcMax <= self.aqAvgPres))
        if condG.any():
            keys = np.flatnonzero(condG).astype(np.int32)
            toDrain = clust.toDrain[keys]
            print('$$:  ', keys, toDrain)
            clust.drainEvents += keys.size
            _mem, neigh = addMembers(self, keys, toDrain, clust)
            clust.updateClusterNeighbours(np.unique(clust.clusterID[_mem]), self)
            #clust.updateNeighMatrix(self, neigh)
            
            memkeys = clust.clusterID[_mem]
            keysToUpdate[memkeys] = True
            toDrainKeys = np.unique(memkeys)
            updateToDrainImbibe_numba(toDrainKeys, clust.members, clust.members_offset, clust.size, 
                clust.neighbours, clust.toDrain, clust.toImbibe, clust.pcShrink, clust.pcMax, 
                self.PcI, self.PcD, self.Rarray, self.totElements, _temp.mem, True, pc_min, pc_max)
            clust.disappearCluster(self, pc_min, pc_max)
            clust.valClust[:] = (clust.size>0)
            clust.valClustD[:] = clust.valClust & (clust.toDrain<=self.totElements)
            
            print('~~~~~~~~~~~~~~')
            try:
                tPhaseD.__CondTP_Drainage__(self, clust.pc[clust.clusterID])
            except:
                from IPython import embed; embed()
            self.satList[_mem] = self._cornArea[_mem]/self.areaSPhase[_mem]

        
        condS = (clust.valClust & (clust.pc > self.aqAvgPres) & (clust.pcShrink > self.aqAvgPres))
        if condS.any():
            keys = np.flatnonzero(condS).astype(np.int32)
            #print('????????????????????/')
            #from IPython import embed; embed()
            for k in keys:
                clustW.fill_with_phase(clust.toImbibe[k], clust.pcShrink[k], self)
        
            # clusters to shrink
            toImbibe = clust.toImbibe[keys]
            print('@@:  ', keys, toImbibe)
            clust.imbEvents += keys.size
            self.satList[toImbibe] = 1.0

            mem, neigh = _temp.mem, _temp.done
            mem[mem] = False
            neigh[neigh] = False
            for k in keys:
                mem[clust[k].members] = True
                neigh[clust.neighbours[k]] = True

            #mem = clust.members[keys].any(axis=0)
            #neigh = mem|clust.neighbours[keys].any(axis=0)
            neigh |= mem
            mem[toImbibe] = False
            _mem = np.flatnonzero(mem).astype(np.int32)
            oldMemKeys = clust.clusterID[_mem]

            removeMembers(self, keys, toImbibe, clust)
            #clust.updateNeighMatrix(self, neigh[self.tList])
            clust.updateClusterNeighbours(np.unique(clust.clusterID[_mem]), self)
            newMemKeys = np.unique(clust.clusterID[_mem])
            updateToDrainImbibe_numba(newMemKeys, clust.members, clust.members_offset, clust.size,
                clust.neighbours, clust.toDrain, clust.toImbibe, clust.pcShrink, clust.pcMax, 
                self.PcI, self.PcD, self.Rarray, self.totElements, _temp.mem, True, pc_min, pc_max)
            clust.valClustD[:] = clust.valClust & (clust.toDrain<=self.totElements) #revise
            clust.disappearCluster(self, pc_min, pc_max)
            clust.valClust[:] = (clust.size>0)
            clust.valClustD[:] = clust.valClust & (clust.toDrain<=self.totElements)

            tPhaseImb.__CondTPImbibition__(self, mem, clust.pc[clust.clusterID], True, True)
            self.satList[_mem] = self._areaWP[_mem]/self.areaSPhase[_mem]
        
        if not (condG.any() or condS.any()):
            break

        self.satW = ((self.satList[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()/
                        self.totVoidVolume)
        valkeys = np.flatnonzero(clust.valClust).astype(np.int32)
        clust.volume[valkeys] = np.bincount(
            clust.clusterID[clust.hasFluid], (1-self.satList[clust.hasFluid])*self.volarray[clust.hasFluid])[valkeys]

        #input('waittt!!!!')
        writeData(self, clust)

        for k in range(clust.nClusters):
            if (clust.size[k]==0):
                if (clust[k].neighbours.size!=0):
                    print(k, clust[k].neighbours)
            else:
                #neigh = np.unique(np.concatenate(self.connectivity_graph[clust[k].members]))
                #from IPython import embed; embed()
                neigh = np.unique(np.concatenate(self.connectivity_graph[clust[k].members]))
                nonMem = neigh[clust.clusterID[neigh]!=k]
                if clust.hasFluid[nonMem].any() or (nonMem.size != clust[k].neighbours.size):
                    print(k, clust[k].neighbours, neigh)
                    input('waiittttttttttttttt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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
    print(f'@alpha={alpha}, initial_sat={initialSw}, final_sat={self.satW}')

    print('Im done !!!')
    from IPython import embed; embed()


def addMembers(self, keys, toDrain, clust):
    clusterW = self.clusterW
    mem = _temp.mem
    mem[mem] = False
    
    for i in range(toDrain.size):
        ii, k = toDrain[i], keys[i]
        
        if clust.hasFluid[ii]:
            continue
        
        if self.isCircle[ii]:
            clusterW.unfill_phase(ii, self.PcD[ii])

        clust.fill_with_phase(ii, self.PcD[ii], self)
        newK = clust.clusterID[ii]
        memID = clust[newK].members
        mem[memID] = True       

    mem[[-1, 0]] = False
    _mem = np.flatnonzero(mem).astype(np.int32)
    try:
        print(_mem)
        thr = np.concatenate(self.PTConData[_mem[self.isPore[_mem]]])
        mem[thr] = True
    except ValueError:
        pass
    
    for k in keys:
        clust.neighbours[k] = []
    
    return _mem, mem[self.tList]


def removeMembers(self, keys, toImbibe, clust):
    for i in range(toImbibe.size):
        k, toImb = keys[i], toImbibe[i]
        #print(':::::@@@@@@@@@@@@@@')
        #from IPython import embed; embed()
        #clust.neighbours[k, clust[k].neighbours] = False
        clust.neighbours[k] = []
        clust.unfill_phase(toImb, self.PcI[toImb])


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
        np.savetxt(f9, [clust.size], delimiter=',', fmt='%g')
        np.savetxt(f10, [self.satList], delimiter=',', fmt='%g')

       



