import numpy as np
from pnflowPy.clustering import Cluster as clust
    

class Cluster(clust):
    def __init__(self, obj, fluid=1, numClusters=200):
        super().__init__(obj, fluid, numClusters)


    def updateToDrainImbibe(self, k, other):
        ''' updates the toDrain and toImbibe arrays'''
        mem = self[k].members
        self.elemCornerArea[mem] = True
        try:
            assert mem.any()
            if mem.size==1:
                self.pcShrink[k] = other.PcI[mem]
                self.volShrink[k] = 0.0
                self.toImbibe[k] = mem
            else:
                toImbibe = mem[np.argmax(other.PcI[mem])]
                self.toImbibe[k] = toImbibe
                self.pcShrink[k] = other.PcI[toImbibe]
                memI = self.members[k].copy()
                memI[toImbibe] = False
               
            neigh = self[k].neighbours
            toDrain = neigh[np.argmin(other.PcD[neigh])]
            self.toDrain[k] = toDrain
            self.pcMax[k] = other.PcD[self.toDrain[k]]
            
        except ValueError:
            self.toDrain[k] = other.totElements+1
            self.pcMax[k] = 1e30
        except AssertionError:
            self.toImbibe[k] = self.toDrain[k] = other.totElements+1
            self.pcShrink[k], self.pcMax[k] = -1e30, 1e30


    def computeCornerArea(self, key, arr, newPc, other):
        # store the trapped status of the elements, 
        # compute the corner areas at the new pc and restore back the trapped status
        oldWStatus, oldNWStatus = other.trappedW[arr], other.trappedNW[arr]
        other.trappedW[arr], other.trappedNW[arr] = False, True
        oldPc = self.pc[key]
        self.pc[key] = newPc[key]
        pc = newPc[other.clusterNW_ID]
        pc[other.clusterNW_ID<0] = 0.0
        other.__CondTPImbibition__(arr, pc, False)
        other.trappedW[arr], other.trappedNW[arr] = oldWStatus, oldNWStatus
        self.pc[key] = oldPc


    def updateMolesShrinkGrowth(self, keys, other):
        self.molesMax[keys] = (
            (other.imposedP+self.pcMax[keys])*self.volMax[keys]/(other.R*other.T)
            + other.H*self.pcMax[keys]*(self.totalVolume[keys]-self.volMax[keys]))
        self.molesShrink[keys] = (
            (other.imposedP+self.pcShrink[keys])*self.volShrink[keys]/(other.R*other.T)
            + other.H*self.pcShrink[keys]*(self.totalVolume[keys]-self.volShrink[keys]))

    def shrinkCluster(self, keys, other):
        ''' update toImbibe element(s) !!! '''

        toImbibe = self.toImbibe[keys]
        other.dissolvedMoles[toImbibe] = other.gasConc[toImbibe,1]*other.volarray[toImbibe]
        self.moles[keys] -= other.dissolvedMoles[toImbibe]

        mem = self.members[keys].any(axis=0)
        neigh = mem|self.neighbours[keys].any(axis=0)
        self.neighbours[keys] = False
        oldkeys = other.clusterNW_ID.copy()

        [*map(other.fillWithWater, toImbibe)]
        [other.unfillWithOil(self.toImbibe[k], self.pc[k], True, False, False, False) for k in keys]
            
        other.satList[toImbibe] = 1.0
        newMem = mem & (other.clusterNW_ID>=0)
        # try:
        #     assert mem.any()
        #     initialVolume = self.volume.copy()
        #     initialMoles = self.moles.copy()
        # except AssertionError:
        #     return
        _oldkeys = oldkeys[newMem]
        _newkeys = other.clusterNW_ID[newMem]
        try:
            assert (_oldkeys==_newkeys).all()
            self.updateNeighMatrix(neigh[other.tList])
            newkeys = np.unique(_newkeys)
            other.initialMoles[newkeys] = self.moles[newkeys]
            _ = self.computeVolume(newkeys, newMem, oldkeys, other)
            other.initialVolume[newkeys] = self.volume[newkeys]
            other.initialPc[newkeys] = self.pc[newkeys]
            other.initialCornerArea[newMem] = other.cornerArea[newMem]
            self.totalVolume[newkeys] = np.bincount(_newkeys, other.volarray[newMem])[newkeys]
            other.updateVolumeShrinkMax(self, newkeys)
            #[self.updateToDrainImbibe(k, other) for k in _newkeys]
            self.updateMolesShrinkGrowth(newkeys, other)
            #self.updateHighLow(keys, other)
        except AssertionError:
            _keys, newkeys = mapOldNewKeys(_oldkeys, _newkeys)
            try:
                assert newkeys.max()<self.moles.size
            except AssertionError:
                self.resizeArrays(other)
            self.updateNeighMatrix(neigh[other.tList])
            
            ''' update the clusters'''
            
            # _newMem = newMem.copy()
            # _newMem[newMem] = (_oldkeys!=_newkeys)
            # cond = np.zeros(other.totElements, dtype=bool)
            # cond[mem[_oldkeys != _newkeys]] = True
            # cond = (_keys!=newkeys)
            # _keys, newkeys = _keys[cond], newkeys[cond]
            initialMoles = self.moles[_keys]
            initialVolume = self.computeVolume(newkeys, newMem, oldkeys, other)
            other.initialVolume[newkeys] = self.volume[newkeys]
            fr =  self.volume[newkeys]/initialVolume[_keys]
            other.initialMoles[newkeys] = self.moles[newkeys] = fr*initialMoles
            other.initialPc[newkeys] = self.pc[newkeys]
            other.initialCornerArea[newMem] = other.cornerArea[newMem]
            self.totalVolume[newkeys] = np.bincount(_newkeys, other.volarray[newMem])[newkeys]
            other.updateVolumeShrinkMax(self, newkeys)

            #[self.updateToDrainImbibe(k, other) for k in newkeys]
            #volMax=np.bincount(oldkeys, self.volMax[newkeys])
            # print("::::::::::@@@@@@@@@@@@@@@@@@@")
            # from IPython import embed; embed()
            #fr = other.maxNWVolarray[mem]/volMax[_oldkeys]
            # fr_bin = np.bincount(_oldkeys, fr) 
            # if not np.isclose(fr_bin[fr_bin!=0.0],1.0).all():
            #     print("::::::::::@@@@@@@@@@@@@@@@@@@")
            #     from IPython import embed; embed()

            # other.initialVolume[newkeys] = self.volume[newkeys] = np.bincount(
            #     _newkeys, fr*initialVolume)[newkeys]
            # other.initialMoles[newkeys] = self.moles[newkeys] = np.bincount(
            #     _newkeys, fr*initialMoles)[newkeys]
            # self.initialPc[newkeys] = self.pc[newkeys]
            # self.totalVolume[newkeys] = np.bincount(
            #     _newkeys, other.volarray[mem])[newkeys]
            #self.updateHighLow(newkeys, other, False)
            self.updateMolesShrinkGrowth(newkeys, other)

            if any(self.molesShrink[newkeys]>self.moles[newkeys]):
               print('a cluster may need further shrinking, check!!!')
               #from IPython import embed; embed()
        except ValueError:
            pass
        except:
            print('there is an error on line 89!!!')
            from IPython import embed; embed()

        keysEmpty = keys[self.size[keys]==0]
        self.availableID.update(np.setdiff1d(keysEmpty,self.availableID))
        for k in keysEmpty:
            del self[k]
            self.molesShrink[k], self.molesMax[k] = 0.0, 1e30
            self.volShrink[k] = self.volMax[k] = 0.0
            self.pcShrink[k], self.pcMax[k] = 0.0, 1e30


    def computeVolume(self, newkeys, newMem, oldkeys, other):
        self.computeCornerArea(newkeys, newMem, self.pc, other)
        newMem = other.elementListS[newMem]
        vol = (1-other.cornerArea[newMem]/other.areaSPhase[newMem])*other.volarray[newMem]
        
        self.volume[newkeys] = np.bincount(other.clusterNW_ID[newMem], vol)[newkeys]
        initialVolume = np.bincount(oldkeys[newMem], vol)
        return initialVolume

       
    def growCluster(self, keys, other):
        ''' update toDrain element(s) !!! '''
        print('Im in grow clusters!!!')
        
        toDrain = self.toDrain[keys]
        self.moles[keys] += other.dissolvedMoles[toDrain]
        mem = self.members[keys].any(axis=0)
        neigh = mem|self.neighbours[keys].any(axis=0)
        for e in toDrain:
            neigh[other.elem[e].neighbours] = True
        neigh[[-1,0]] = False

        # unfill circular elements with water 
        toDrain1 = toDrain[other.isCircle[toDrain]]
        other.hasWFluid[toDrain1] = False
        other.clusterW.members[other.clusterW_ID[toDrain1], toDrain1] = False
        other.clusterW_ID[toDrain1] = -5

        # fill with oil
        other.fluid[toDrain] = 1
        other.hasNWFluid[toDrain] = True
        other.clusterNW_ID[toDrain] = keys
        self.members[keys, toDrain] = True
        # if 797 in keys:
        #     from IPython import embed; embed()
        self.updateNeighMatrix(neigh[other.tList])

        newkeys = np.unique(other.clusterNW_ID[mem])
        for k in newkeys:
            other.gasConc[other.clusterNW_ID==k,1] = other.H*self.pc[k]

        mem = self.members[newkeys].any(axis=0)
        _ = self.computeVolume(newkeys, mem, other.clusterNW_ID, other)
        other.initialVolume[newkeys] = self.volume[newkeys]
        other.initialMoles[newkeys] = self.moles[newkeys]
        other.initialPc[newkeys] = self.pc[newkeys]
        other.initialCornerArea[mem] = other.cornerArea[mem]

        self.totalVolume[newkeys] = np.bincount(
            other.clusterNW_ID[mem], other.volarray[mem])[newkeys]
        other.updateVolumeShrinkMax(self, newkeys)
        #[self.updateToDrainImbibe(k, other) for k in newkeys]
        #self.updateHighLow(newkeys, other, shrink=False)
        self.updateMolesShrinkGrowth(newkeys, other)
    

    def updateHighLow(self, newkeys, other, update=True, shrink=True):
        def _updateShrink():
            self.molesShrink[newkeys] = self.molesMin[newkeys]
            valkeysI = newkeys[(self.size[newkeys]>1)&(self.pcShrink[newkeys]>=0.0)]
            members = self.members[valkeysI]
            members[np.arange(valkeysI.size), self.toImbibe[valkeysI]] = False
            
            volShrink = ((1-self.avgSatList)*other.volarray*members).sum(axis=1)
            self.molesShrink[valkeysI] = (
                (other.imposedP+self.pcShrink[valkeysI])*volShrink/(other.R*other.T)+
                other.H*self.pcShrink[valkeysI]*(self.totalVolume[valkeysI]-volShrink))
            
        def _updateMax():
            self.molesMax[newkeys] = (
                (other.imposedP+self.pcMax[newkeys])*self.volMax[newkeys]/(other.R*other.T)+
                other.H*self.pcMax[newkeys]*(self.totalVolume[newkeys]-self.volMax[newkeys]))
            
        try:
            self.molesMin[newkeys] = (
                other.H*self.pcShrink[newkeys]*self.totalVolume[newkeys])
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
        

    def resizeArrays(self, other):
        ct = self.pc.size - self.neighbours.shape[0]
        self.neighbours = np.vstack(
            (self.neighbours, np.zeros([ct,other.totElements], dtype=bool)))
        self.toDrain = np.concatenate((self.toDrain, np.zeros(ct, dtype=int)))
        self.toImbibe = np.concatenate((self.toImbibe, np.zeros(ct, dtype=int)))
        self.pcMax = np.concatenate((self.pcMax, np.full(ct, 1e30, dtype=float)))
        self.pcShrink = np.concatenate((self.pcShrink, np.full(ct, -1e30, dtype=float)))
        self.moles = np.concatenate((self.moles, np.zeros(ct, dtype=float)))
        self.molesMax = np.concatenate((self.molesMax, np.full(ct, 1e30, dtype=float)))
        #self.molesMin = np.concatenate((self.molesMin, np.zeros(ct, dtype=float)))
        self.molesShrink = np.concatenate((self.molesShrink, np.zeros(ct, dtype=float)))
        self.volume = np.concatenate((self.volume, np.zeros(ct, dtype=float)))
        self.volMax = np.concatenate((self.volMax, np.zeros(ct, dtype=float)))
        self.volShrink = np.concatenate((self.volShrink, np.zeros(ct, dtype=float)))
        other.initialMoles = np.concatenate((other.initialMoles, np.zeros(ct, dtype=float)))
        other.initialPc = np.concatenate((other.initialPc, np.zeros(ct, dtype=float)))
        other.initialVolume = np.concatenate((other.initialVolume, np.zeros(ct, dtype=float)))
        self.totalVolume = np.concatenate((self.totalVolume, np.zeros(ct, dtype=float)))
        

def mapOldNewKeys(oldkeys, newkeys):
    ''' returns unique pairs of old and new keys '''
    i=0
    # oldkeys = oldkeys[newkeys>=0]
    # newkeys = newkeys[newkeys>=0]
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








    



