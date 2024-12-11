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
            assert mem.any()
            toImbibe = mem[np.argmax(self.PcI[mem])]
            self.toImbibe[k] = toImbibe
            self.pcShrink[k] = max(0.0, self.PcI[self.toImbibe[k]])
            
            self.volMax[k] = other.maxNWVolarray[mem].sum()
            toDrain = neigh[np.argmin(self.PcD[neigh])]
            self.toDrain[k] = toDrain
            self.pcMax[k] = self.PcD[self.toDrain[k]]
        except ValueError:
            self.toDrain[k] = self.totElements+1
            self.pcMax[k] = 1e30
        except AssertionError:
            self.toImbibe[k] = self.toDrain[k] = self.totElements+1
            self.pcShrink[k], self.pcMax[k] = -1e30, 1e30

    def shrinkCluster(self, keys, other):
        ''' update toImbibe element(s) !!! '''
        toImbibe = self.toImbibe[keys]
        other.dissolvedMoles[toImbibe] = other.gasConc[toImbibe,1]*self.volarray[toImbibe]
        self.moles[keys] -= other.dissolvedMoles[toImbibe]

        mem = self.members[keys].any(axis=0)
        neigh = mem|self.neighbours[keys].any(axis=0)
        self.neighbours[keys] = False
        oldkeys = self.clusterNW_ID[mem]

        [*map(other.fillWithWater, toImbibe)]
        [other.unfillWithOil(self.toImbibe[k], self.pc[k], True, False, False, False) for k in keys]
            
        other.satList[toImbibe] = 1.0
        _oldkeys = oldkeys[(self.clusterNW_ID[mem]>=0)]
        try:
            assert _oldkeys.size>0
        except AssertionError:
            return
        
        mem = self.elementListS[mem & (self.clusterNW_ID>=0)]
        _newkeys = self.clusterNW_ID[mem]
        try:
            assert (_oldkeys==_newkeys).all()
            self.updateNeighMatrix(neigh[self.tList])
            keys = np.unique(_newkeys)
            other.initialMoles[keys] = self.moles[keys]
            other.initialVolume[keys] = self.volume[keys]
            other.initialPc[keys] = self.pc[keys]
            other.totalVolClusters[keys] = np.bincount(_newkeys, self.volarray[mem])[keys]
            [self.updateToDrainImbibe(k, other) for k in _newkeys]
            self.updateHighLow(keys, other)
        except AssertionError:
            try:
                assert _newkeys.max()<self.moles.size
            except AssertionError:
                self.resizeArrays(other)
            self.updateNeighMatrix(neigh[self.tList])
            
            initialVolume = self.volume[_oldkeys]
            initialMoles = self.moles[_oldkeys]
            oldkeys, newkeys = mapOldNewKeys(_oldkeys, _newkeys)
        
            ''' update the clusters'''
            [self.updateToDrainImbibe(k, other) for k in newkeys]
            volMax=np.bincount(oldkeys, self.volMax[newkeys])
            fr = other.maxNWVolarray[mem]/volMax[_oldkeys]
            other.initialVolume[newkeys] = self.volume[newkeys] = np.bincount(
                _newkeys, fr*initialVolume)[newkeys]
            other.initialMoles[newkeys] = self.moles[newkeys] = np.bincount(
                _newkeys, fr*initialMoles)[newkeys]
            other.initialPc[newkeys] = self.pc[newkeys]
            other.totalVolClusters[newkeys] = np.bincount(
                _newkeys, self.volarray[mem])[newkeys]
            self.updateHighLow(newkeys, other, False)

            if any(self.molesShrink[newkeys]>self.moles[newkeys]):
               print('a cluster may need further shrinking, check!!!')
        except:
            print('there is an error on line 89!!!')
            from IPython import embed; embed()


    
    def growCluster(self, keys, other):
        ''' update toDrain element(s) !!! '''
        print('Im in grow clusters!!!')
        toDrain = self.toDrain[keys]
        self.moles[keys] += other.dissolvedMoles[toDrain]
        mem = self.members[keys].any(axis=0)
        neigh = mem|self.neighbours[keys].any(axis=0)
        self.neighbours[keys] = False

        other.initialVolume[keys] = self.volume[keys]
        other.initialMoles[keys] = self.moles[keys]
        other.initialPc[keys] = self.pc[keys]

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
        
        mem = self.members[keys].any(axis=0)
        newkeys = self.clusterNW_ID[mem]
        try:
            other.totalVolClusters[keys] = np.bincount(newkeys, self.volarray[mem])[keys]
        except:
            print('there is an error on line 123!!!')
            from IPython import embed; embed()
        [self.updateToDrainImbibe(k, other) for k in keys]
        self.updateHighLow(keys, other, shrink=False)
        

    def updateHighLow(self, newkeys, other, update=True, shrink=True):
        def _updateShrink():
            self.molesShrink[newkeys] = self.molesMin[newkeys].copy()
            valkeysI = newkeys[(self.size[newkeys]>1)&(self.pcShrink[newkeys]>=0.0)]
            members = self.members[valkeysI]
            members[np.arange(valkeysI.size), self.toImbibe[valkeysI]] = False
            
            volShrink = ((1-self.avgSatList)*self.volarray*members).sum(axis=1)
            self.molesShrink[valkeysI] = (
                (other.imposedP+self.pcShrink[valkeysI])*volShrink/(other.R*other.T)+
                other.H*self.pcShrink[valkeysI]*(other.totalVolClusters[valkeysI]-volShrink))
            
        def _updateMax():
            self.molesMax[newkeys] = (
                (other.imposedP+self.pcMax[newkeys])*self.volMax[newkeys]/(other.R*other.T)+
                other.H*self.pcMax[newkeys]*(other.totalVolClusters[newkeys]-self.volMax[newkeys]))
            
        try:
            self.molesMin[newkeys] = (
                other.H*self.pcShrink[newkeys]*other.totalVolClusters[newkeys])
            assert update
            try:
                assert shrink
                self.molesMax[newkeys] = self.molesShrink[newkeys]
                _updateShrink()
            except AssertionError:
                self.molesShrink[newkeys] = self.molesMax[newkeys]
                _updateMax()
        except AssertionError:
            _updateMax()
            _updateShrink()
        
        other.betaLow[newkeys] = (other.initialPc[newkeys]-self.pcShrink[newkeys])/(
            other.initialMoles[newkeys]-self.molesShrink[newkeys])
        other.betaHigh[newkeys] = (self.pcMax[newkeys]-other.initialPc[newkeys])/(
            self.molesMax[newkeys]-other.initialMoles[newkeys])
        # if any(other.betaHigh<0.0):
        #     print('betaHigh is low!!! check')
        #     from IPython import embed; embed()

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








    



