import numpy as np
from pnflowPy.clustering import Cluster as clust
    

class Cluster(clust):
    def __init__(self, obj, fluid=1, numClusters=200):
        super().__init__(obj, fluid, numClusters)

    def updateToDrainImbibe(self, k):
        ''' updates the toDrain and toImbibe arrays'''
        arrI = self[k].members
        arrD = self[k].neighbours
        try:
            toImbibe = arrI[np.argmax(self.PcI[arrI])]
            self.toImbibe[k] = toImbibe
            self.volLow[k] = self.volume[k] - self.volarray[toImbibe]
            self.pcLow[k] = self.PcI[self.toImbibe[k]]

            toDrain = arrD[np.argmin(self.PcD[arrD])]
            self.toDrain[k] = toDrain
            self.volHigh[k] = self.volume[k] + self.volarray[toDrain]
            self.pcHigh[k] = self.PcD[self.toDrain[k]]
        except ValueError:
            pass

    def shrinkCluster(self, keys, other):
        ''' update toImbibe element(s) !!! '''
        toImbibe = self.toImbibe[keys]
        mem = self.members[keys].any(axis=0)
        neigh = mem|self.neighbours[keys].any(axis=0)
        oldkeys = self.clusterNW_ID[mem]
        [*map(other.fillWithWater, toImbibe)]
        [other.unfillWithOil(self.toImbibe[k], self.pc[k], True, False, False, False) for k in keys]
        newkeys = self.clusterNW_ID[mem]
        oldkeys, newkeys = mapOldNewKeys(oldkeys, newkeys)
        self.obj.satList[toImbibe] = 1.0
        
        ''' recluster the clusters!!! '''
        initialMolesTotal = self.molesTotal[oldkeys]
        initialMoles = self.moles[oldkeys]
        initialVolume = self.volLow[oldkeys]
        self.molesLow[oldkeys] = 1e-30
        self.moles[oldkeys] = 0.0
        other.betaLow[oldkeys] = other.factLow[oldkeys] = 0.0
        other.betaHigh[oldkeys] = other.factHigh[oldkeys] = 0.0
            
        ''' update the clusters'''
        self.updateNeighMatrix(neigh[self.tList])
        try:
            assert newkeys.size>0
            self.volume[newkeys] = (self.volarray*self.members[newkeys]).sum(axis=1)
            fr = self.volume[newkeys]/initialVolume
            other.initialMoles[newkeys] = self.moles[newkeys] = fr*initialMoles
            self.molesTotal[newkeys] = fr*initialMolesTotal
            other.initialPc[newkeys] = self.pc[newkeys]
            [self.updateToDrainImbibe(k) for k in newkeys]
            updateHighLow(self, newkeys, other)
        except AssertionError:
            pass

    def growCluster(self, keys, other):
        ''' update toDrain element(s) !!! '''
        print('Im in grow clusters!!!')
        toDrain = self.toDrain[keys]
        mem = self.members[keys].any(axis=0)
        neigh = mem|self.neighbours[keys].any(axis=0)

        # unfill circular elements with water 
        toDrain1 = toDrain[self.isCircle[toDrain]]
        self.obj.hasWFluid[toDrain1] = False
        self.clusterW.members[self.clusterW_ID[toDrain1], toDrain1] = False
        self.clusterW_ID[toDrain1] = -5

        # fill with oil
        self.obj.fluid[toDrain] = 1
        self.obj.hasNWFluid[toDrain] = True
        self.clusterNW_ID[toDrain] = keys
        self.members[keys, toDrain] = True
        self.updateNeighMatrix(neigh[self.tList])

        self.volume[keys] = (self.volarray*self.members[keys]).sum(axis=1)
        self.molesTotal[keys] = (self.moles[keys]+
                                (other.dissolvedMoles*self.members[keys]).sum(axis=1))
        [self.updateToDrainImbibe(k) for k in keys]
        other.initialMoles[keys] = self.moles[keys]
        updateHighLow(self, keys, other)
        print('Im on line 93!!!')
        


def mapOldNewKeys(oldkeys, newkeys):
    ''' returns unique pairs of old and new keys '''
    i=0
    oldkeys = oldkeys[newkeys>=0]
    newkeys = newkeys[newkeys>=0]
    while True:
        try:
            cond = (newkeys!=newkeys[i])
            cond[i] = True
            newkeys = newkeys[cond]
            oldkeys = oldkeys[cond]
            i += 1
        except IndexError:
            break
    return oldkeys, newkeys

def updateHighLow(self, newkeys, other):
    self.molesHigh[newkeys] = (
        (self.imposedP+self.pcHigh[newkeys])*self.volHigh[newkeys]/(self.R*self.T))
    self.molesLow[newkeys] = np.clip(
        (self.imposedP+self.pcLow[newkeys])*
        self.volLow[newkeys]/(self.R*self.T), 1e-30, None)
    
    other.betaLow[newkeys] = (other.initialPc[newkeys]-self.pcLow[newkeys])/(
        other.initialMoles[newkeys]-self.molesLow[newkeys])
    other.betaHigh[newkeys] = (self.pcHigh[newkeys]-other.initialPc[newkeys])/(
        self.molesHigh[newkeys]-other.initialMoles[newkeys])
    other.factLow[newkeys] = 1/(1+(
        other.betaLow[newkeys]*self.H*self.volume[newkeys]))
    other.factHigh[newkeys] = 1/(1+(
        other.betaHigh[newkeys]*self.H*self.volume[newkeys]))






    



