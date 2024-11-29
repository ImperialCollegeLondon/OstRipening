import numpy as np
from pnflowPy.clustering import Cluster as clust
    

class Cluster(clust):
    def __init__(self, obj, fluid=1, numClusters=200):
        super().__init__(obj, fluid, numClusters)

    def updateToDrainImbibe(self, k, other):
        ''' updates the toDrain and toImbibe arrays'''
        mem = self[k].members
        neigh = self[k].neighbours
        try:
            toImbibe = mem[np.argmax(self.PcI[mem])]
            self.toImbibe[k] = toImbibe
            #self.volLow[k] = self.volume[k] - self.obj.volarrayNW[toImbibe]
            self.pcShrink[k] = self.PcI[self.toImbibe[k]]
            
            self.volMax[k] = other.maxNWVolarray[mem].sum()
            toDrain = neigh[np.argmin(self.PcD[neigh])]
            self.toDrain[k] = toDrain
            self.pcMax[k] = self.PcD[self.toDrain[k]]
        except ValueError:
            pass

    def shrinkCluster(self, keys, other):
        ''' update toImbibe element(s) !!! '''
        toImbibe = self.toImbibe[keys]
        other.gasConc[toImbibe,1] = self.H*self.imposedP
        other.dissolvedMoles[toImbibe] = other.gasConc[toImbibe,1]*self.volarray[toImbibe]
        self.moles[keys] -= other.dissolvedMoles[toImbibe]

        mem = self.members[keys].any(axis=0)
        neigh = mem|self.neighbours[keys].any(axis=0)
        self.neighbours[keys] = False
        oldkeys = self.clusterNW_ID[mem]
        [*map(other.fillWithWater, toImbibe)]
        [other.unfillWithOil(self.toImbibe[k], self.pc[k], True, False, False, False) for k in keys]
            
        _oldkeys = oldkeys[(self.clusterNW_ID[mem]>=0)]
        mem = self.elementListS[mem & (self.clusterNW_ID>=0)]
        _newkeys = self.clusterNW_ID[mem]
        try:
            assert _newkeys.max()<self.moles.size
        except AssertionError:
            self.resizeArrays(other)
        self.updateNeighMatrix(neigh[self.tList])

        initialVolume = self.volume[_oldkeys]
        initialMoles = self.moles[_oldkeys]
        oldVolMax = self.volMax[_oldkeys]
        oldkeys, newkeys = mapOldNewKeys(_oldkeys, _newkeys)
        other.satList[toImbibe] = 1.0
        
        ''' recluster the clusters!!! '''
        self.molesShrink[oldkeys] = self.molesMin[oldkeys] = self.moles[oldkeys] = 0.0
        self.molesMax[oldkeys] = 1e30
        other.betaLow[oldkeys] = other.betaHigh[oldkeys] = 0.0
        other.totalVolClusters[oldkeys] = 0.0
           
        ''' update the clusters'''
        try:
            assert newkeys.size>0
            [self.updateToDrainImbibe(k, other) for k in newkeys]           
            
            other.initialVolume[newkeys] = self.volume[newkeys] = np.bincount(_newkeys, 
                other.maxNWVolarray[mem]/oldVolMax*initialVolume,
                self.volume.size)[newkeys]
            volMax=np.bincount(oldkeys, self.volMax[newkeys])
            other.initialMoles[newkeys] = self.moles[newkeys] = np.bincount(_newkeys, 
                other.maxNWVolarray[mem]/volMax[_oldkeys]*initialMoles,
                self.volume.size)[newkeys]
            
            other.initialPc[newkeys] = self.pc[newkeys]
            other.totalVolClusters[newkeys] = np.bincount(
                _newkeys, self.volarray[mem])[newkeys]
            self.updateHighLow(newkeys, other)
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
        print('Im in line 80!!!')
        from IPython import embed; embed()
        self.volume[keys] = (self.volarrayNW*self.members[keys]).sum(axis=1)
        self.molesTotal[keys] = (self.moles[keys]+
                                (other.dissolvedMoles*self.members[keys]).sum(axis=1))
        [self.updateToDrainImbibe(k, other) for k in keys]
        other.initialMoles[keys] = self.moles[keys]
        self.updateHighLow(keys, other)
        print('Im on line 93!!!')

    def updateHighLow(self, newkeys, other):
        self.molesMax[newkeys] = (
            (other.imposedP+self.pcMax[newkeys])*self.volMax[newkeys]/(other.R*other.T)+
            other.H*other.imposedP*other.totalVolClusters[newkeys])
        self.molesMin[newkeys] = (other.H*other.imposedP*
            other.totalVolClusters[newkeys])
        
        self.molesShrink[newkeys] = self.molesMin[newkeys].copy()
        valkeysI = newkeys[(self.size[newkeys]>1)&(self.pcShrink[newkeys]>=0.0)]
        members = self.members[valkeysI]
        members[np.arange(valkeysI.size), self.toImbibe[valkeysI]] = False
        newVolMax = (other.maxNWVolarray*members).sum(axis=1)
        volShrink = np.minimum(newVolMax,
            (self.pcShrink[valkeysI]/other.initialPc[valkeysI])*
            other.initialVolume[valkeysI])
        self.molesShrink[valkeysI] += (
            (other.imposedP+self.pcShrink[valkeysI])*volShrink/(other.R*other.T))

        other.betaLow[valkeysI] = other.initialPc[valkeysI]/(
            other.initialMoles[valkeysI]-self.molesMin[valkeysI])
        other.betaHigh[newkeys] = (self.pcMax[newkeys]-other.initialPc[newkeys])/(
            self.molesMax[newkeys]-other.initialMoles[newkeys])
        

    def resizeArrays(self, other):
        ct = self.pc.size - self.neighbours.shape[0]
        self.neighbours = np.vstack(
            (self.neighbours, np.zeros([ct,self.totElements], dtype=bool)))
        self.toDrain = np.concatenate((self.toDrain, np.zeros(ct, dtype=int)))
        self.toImbibe = np.concatenate((self.toImbibe, np.zeros(ct, dtype=int)))
        self.pcMax = np.concatenate((self.pcMax, np.full(ct, 1e30, dtype=float)))
        self.pcShrink = np.concatenate((self.pcShrink, np.full(ct, -1e30, dtype=float)))
        self.moles = np.concatenate((self.moles, np.zeros(ct, dtype=float)))
        self.molesMax = np.concatenate((self.molesMax, np.full(ct, 1e30, dtype=float)))
        self.molesMin = np.concatenate((self.molesMin, np.zeros(ct, dtype=float)))
        self.molesShrink = np.concatenate((self.molesShrink, np.zeros(ct, dtype=float)))
        self.volume = np.concatenate((self.volume, np.zeros(ct, dtype=float)))
        self.volMax = np.concatenate((self.volMax, np.zeros(ct, dtype=float)))
        other.initialMoles = np.concatenate((other.initialMoles, np.zeros(ct, dtype=float)))
        other.initialPc = np.concatenate((other.initialPc, np.zeros(ct, dtype=float)))
        other.initialVolume = np.concatenate((other.initialVolume, np.zeros(ct, dtype=float)))
        other.totalVolClusters = np.concatenate((other.totalVolClusters, np.zeros(ct, dtype=float)))
        other.betaLow = np.concatenate((other.betaLow, np.zeros(ct, dtype=float)))
        other.betaHigh = np.concatenate((other.betaHigh, np.zeros(ct, dtype=float)))
        for c in np.arange(len(self.keys), self.pc.size):
            self[c] = {'keys': c}

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








    



