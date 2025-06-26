import numpy as np
from joblib import dump, load, Parallel, delayed
from scipy import optimize

from pnflowPy.clustering import Cluster as clust
import timeDependency as tD
import pnflowPy.tPhaseImb as tPhaseImb


class Cluster(clust):
    def __init__(self, obj, fluid=1, numClusters=200):
        super().__init__(obj, fluid, numClusters)


    def updateToDrainImbibe(self, k, other, update=False):
        ''' updates the toDrain and toImbibe arrays'''
        mem = self[k].members
        try:
            assert mem.any()
            self.elemCornerArea[mem] = True
            try:
                assert not update
                if mem.size==1:
                    self.pcShrink[k] = other.PcI[mem]
                    self.toImbibe[k] = mem
                else:
                    toImbibe = mem[np.argmax(other.PcI[mem])]
                    self.toImbibe[k] = toImbibe
                    self.pcShrink[k] = other.PcI[toImbibe]
            except AssertionError:
                self.toImbibe[k] = mem[np.argmax(other.PcI[mem])]
                self.pcShrink[k] = other.PcI[self.toImbibe[k]]
            
            try:
                assert self.pcShrink[k]>0.0
            except AssertionError:
                toImbibe = self.toImbibe[k]
                other.PcI[toImbibe] = other.PistonPcAdv[toImbibe]
                self.pcShrink[k] = other.PistonPcAdv[toImbibe]
                           
            neigh = self[k].neighbours
            toDrain = neigh[np.argmin(other.PcD[neigh])]
            self.toDrain[k] = toDrain
            #self.pcMax[k] = max(other.PcD[self.toDrain[k]], self.pcShrink[k])
            self.pcMax[k] = other.PcD[self.toDrain[k]]
           
        except ValueError:
            self.toDrain[k] = other.totElements+1
            self.pcMax[k] = 1e30
        except AssertionError:
            self.toImbibe[k] = self.toDrain[k] = other.totElements+1
            self.pcShrink[k], self.pcMax[k] = 1e-30, 1e30


    def computeCornerArea1(self, arr, newPc, other):
        # store the trapped status of the elements, 
        # compute the corner areas at the new pc and restore back the trapped status
        oldWStatus, oldNWStatus = other.trappedW[arr], other.trappedNW[arr]
        other.trappedW[arr], other.trappedNW[arr] = False, True
        cornerArea = other._cornArea.copy()
        print('Im in computeCornerArea!!!')
        from IPython import embed; embed()
        tPhaseImb.__CondTPImbibition__(other, arr, newPc, True, True)
        other.trappedW[arr], other.trappedNW[arr] = oldWStatus, oldNWStatus
        cond = (arr & (other._cornArea!=cornerArea))
        other.elementPc_area[cond] = newPc[cond]


    def computeCornerArea(self, arrr, newPc, other):
        arrrS = arrr[other.isSquare]
        arrrT = arrr[other.isTriangle]
        arrrC = arrr[other.isCircle]&(other._areaWP[other.isCircle]>0.0)

        try:
            assert arrrS.any()
            apexDist = np.empty_like(other.hingAngSq.T)
            tD.cornerApex(other, other.elemSquare, arrrS, other.halfAnglesSq[:, np.newaxis], 
                          newPc[other.elemSquare], other.cornExistsSq.T,
                          apexDist, other.initedApexDistSq.T, 4)
        except AssertionError:
            pass
        except:
            print('@@@@????????')
            from IPython import embed; embed()
        try:
            assert arrrT.any()
            apexDist = np.empty_like(other.hingAngTr.T)
            # print('Im in computeCornerArea!!!')
            # from IPython import embed; embed()
            tD.cornerApex(other, other.elemTriangle, arrrT, other.halfAnglesTr.T, 
                          newPc[other.elemTriangle], other.cornExistsTr.T,
                          apexDist, other.initedApexDistTr.T, 3)
        except AssertionError:
            pass
        except:
            print(':::>>><<<<<<')
            from IPython import embed; embed()
        try:
            assert arrrC.any()
            arrC = other.elemCircle[arrrC]
            other._areaWP[arrC] = 0.0
            other._areaNWP[arrC] = other.areaSPhase[arrC]
        except AssertionError:
            pass

    
    def computeVolume(self, memkeys, mem, newPc, newkeys, oldkeys, other):
        _mem = mem & (newPc!=other.elementPc_area)
        self.computeCornerArea(_mem, newPc, other)
        mem = other.elementListS[mem]
        vol = (1-other.areaWPhase[mem]/other.areaSPhase[mem])*other.volarray[mem]    
        self.volume[memkeys] = np.bincount(newkeys[mem], vol)[memkeys]
        initialVolume = np.bincount(oldkeys[mem], vol)
        return initialVolume


    def updateMolesShrinkGrowth(self, keys, other):
        self.molesMax[keys] = (
            (other.imposedP+self.pcMax[keys])*self.volMax[keys]/(other.R*other.T)
            + other.H*self.pcMax[keys]*(self.totalVolume[keys]-self.volMax[keys]))
        self.molesShrink[keys] = np.clip(
            (other.imposedP+self.pcShrink[keys])*self.volShrink[keys]/(other.R*other.T)
            + other.H*self.pcShrink[keys]*(self.totalVolume[keys]-self.volShrink[keys]), 
            1e-22, None)
        

    def shrinkCluster(self, keys, other):
        ''' update toImbibe element(s) !!! '''
        #print('Im in shrinkCluster!!!')
        toImbibe = self.toImbibe[keys]
        
        cond = (self.size[keys]==1)
        toImbibe1,keys1 = toImbibe[cond], keys[cond] # clusters to disappear
        other.aqueousMoles[toImbibe1] = self.moles[keys1]
        other.gasConc[toImbibe1,1] = other.aqueousMoles[toImbibe1]/other.volarray[toImbibe1]
        toImbibe2,keys2 = toImbibe[~cond], keys[~cond] #clusters to shrink
        moles2 = other.gasConc[toImbibe2,1]*other.volarray[toImbibe2]
        newMoles = self.moles.copy()
        newMoles[keys2] -= moles2
        other.aqueousMoles[toImbibe2] = moles2
        
    
        mem = self.members[keys].any(axis=0)
        neigh = mem|self.neighbours[keys].any(axis=0)
        self.neighbours[keys] = False
        oldkeys = other.clusterNW_ID.copy()

        # if 0 in keys:
        #     print('Im in shrinkCluster with keys 0!!!')
        #     from IPython import embed; embed()
        
        try:
            assert (other.fluid[toImbibe]==1).all()
            [tPhaseImb.fillWithWater(other, i) for i in toImbibe]
            [tPhaseImb.unfillWithOil(
                other, self.toImbibe[k], self.pc[k], True, False, False, False) for k in keys]
            assert self.members.shape[0] != self.neighbours.shape[0]
            self.resizeArrays(other)
            other.valClust = (self.members.any(axis=1))
            newMoles = self.moles.copy()
        except AssertionError:
            pass
          
        newkeys = other.clusterNW_ID
        other.satList[toImbibe] = 1.0
        other._areaWP[toImbibe] = other.areaSPhase[toImbibe]
        other._areaNWP[toImbibe] = 0.0
        newMem = mem & (newkeys>=0)
        oldMemKeys = oldkeys[newMem]
        newMemKeys = newkeys[newMem]

        try:
            assert (oldMemKeys==newMemKeys).all()
            self.updateNeighMatrix(other,neigh[other.tList])
            _newMemKeys = np.unique(newMemKeys)
            self.totalVolume[_newMemKeys] = np.bincount(
                newMemKeys, other.volarray[newMem])[_newMemKeys]
            oldToImbibe = self.toImbibe[_newMemKeys]
            tD.updateVolumeShrinkMax(other, self, _newMemKeys)
            
            kk = _newMemKeys[(oldToImbibe==self.toDrain[_newMemKeys])]
            oldMolesShrink = self.molesShrink[kk]
            self.updateMolesShrinkGrowth(_newMemKeys, other)
            self.dVg_dPc[_newMemKeys] = (self.volMax[_newMemKeys]-self.volShrink[_newMemKeys])/(
                self.pcMax[_newMemKeys]-self.pcShrink[_newMemKeys])
            self.molesMax[kk] = np.maximum(self.molesMax[kk], oldMolesShrink)
            self.molesShrink[_newMemKeys] = np.clip(np.minimum(
                self.molesShrink[_newMemKeys], self.moles[_newMemKeys]-1e-22), 1e-22, None)
            
            keysVol = np.zeros(self.pc.size, dtype=bool)
            keysVol[_newMemKeys] = True

            tD.updateClusterPcVolume(other, newMem, keysVol, self, newkeys, newMoles, True)
            self.moles[_newMemKeys] = newMoles[_newMemKeys]

            other.gasConc[newMem,1] = other.H*self.pc[newkeys[newMem]]
            
        except AssertionError:
            _oldMemKeys, _newMemKeys = mapOldNewKeys(oldMemKeys.copy(), newMemKeys.copy())

            self.updateNeighMatrix(other,neigh[other.tList])
            
            ''' update the clusters'''
            self.totalVolume[_newMemKeys] = np.bincount(
                newMemKeys, other.volarray[newMem])[_newMemKeys]

            oldToImbibe = self.toImbibe[_oldMemKeys]
            tD.updateVolumeShrinkMax(other, self, _newMemKeys)
            kk = _newMemKeys[(oldToImbibe==self.toDrain[_newMemKeys])]
            oldMolesShrink = self.molesShrink[kk]
            self.updateMolesShrinkGrowth(_newMemKeys, other)
            self.dVg_dPc[_newMemKeys] = (self.volMax[_newMemKeys]-self.volShrink[_newMemKeys])/(
                self.pcMax[_newMemKeys]-self.pcShrink[_newMemKeys])
            self.molesMax[kk] = np.maximum(self.molesMax[kk], oldMolesShrink)
            
            self.moles[_oldMemKeys] = 0.0
            if _newMemKeys.size==1:
                newMoles[_newMemKeys] = newMoles[_oldMemKeys]
            else:
                frS= self.molesShrink[_newMemKeys]/np.bincount(
                    _oldMemKeys,self.molesShrink[_newMemKeys])[_oldMemKeys]
                frM=self.molesMax[_newMemKeys]/np.bincount(
                    _oldMemKeys,self.molesMax[_newMemKeys])[_oldMemKeys]
                newMoles[_newMemKeys] = (frM+frS)/2*newMoles[_oldMemKeys]

            self.molesShrink[_newMemKeys] = np.clip(np.minimum(
                self.molesShrink[_newMemKeys], self.moles[_newMemKeys]-1e-22), 1e-22, None)

            keysVol = np.zeros(self.pc.size, dtype=bool)
            keysVol[_newMemKeys] = True
            tD.updateClusterPcVolume(other, newMem, keysVol, self, newkeys, newMoles, True)
            self.moles[_newMemKeys] = newMoles[_newMemKeys]
            other.gasConc[newMem,1] = other.H*self.pc[newkeys[newMem]]
        except ValueError:
            pass
        except:
            print('there is an error on line 89!!!')
            from IPython import embed; embed()

        keysEmpty = np.where((self.size==0) & (self.toImbibe<other.totElements))[0]
        self.availableID.update(np.setdiff1d(keysEmpty,self.availableID))
        for k in keysEmpty:
            del self[k]
            self.molesShrink[k], self.molesMax[k] = 0.0, 1e30
            self.volShrink[k] = self.volMax[k] = 0.0
            self.pcShrink[k], self.pcMax[k] = 0.0, 1e30
            self.updateToDrainImbibe(k, other)

        for k in _newMemKeys[~self.trappedStatus[_newMemKeys]]:
            if not self.members[k][other.conTToExit].any():
                self.trappedStatus[k] = True
                other.trappedNW[self[k].members] = True

       
    def growCluster(self, keys, other):
        ''' update toDrain element(s) !!! '''
        mem = self.members[keys].any(axis=0)
        neigh = mem|self.neighbours[keys].any(axis=0)

        # to prevent duplication of elements that are toDrain for more than one cluster
        newkeys = other.clusterNW_ID
        toDrain = self.toDrain[keys]
        newkeys[toDrain] = keys
        keys = np.unique(newkeys[toDrain])
        toDrain = self.toDrain[keys]
        self.moles[keys] += other.aqueousMoles[toDrain]
    
        try:
            assert (toDrain>=other.nPores).any()
            toDrainT = toDrain[(toDrain>other.nPores)] # throats
            conP = other.TPConnections[toDrainT-other.nPores]
            toDrainT = toDrainT[((conP>0)&(newkeys[conP]<0)).any(axis=1)]
            assert toDrainT.any()
            conP = conP[(newkeys[conP]<0)&(conP>0)]
            print(f'pores {conP} has now automatically being filled from throats {toDrainT}')

            newkeys[conP] = newkeys[toDrainT]
            self.moles[newkeys[conP]] += other.aqueousMoles[conP]
            toDrain = np.append(toDrain, conP)
            keys = np.append(keys, newkeys[conP])
        except AssertionError:
            pass

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
        self.members[keys, toDrain] = True
        mem[toDrain] = True

        self.updateNeighMatrix(other,neigh[other.tList])
        
        memkeys = np.unique(newkeys[toDrain])
        mem = self.members[memkeys].any(axis=0)
        self.totalVolume[memkeys] = np.bincount(newkeys[mem], other.volarray[mem])[memkeys]
        tD.updateVolumeShrinkMax(other, self, memkeys)
        
        # update moles needed at shrinkage and growth
        cond = (toDrain==self.toImbibe[newkeys[toDrain]])
        oldkk, newkk = keys[cond], newkeys[toDrain[cond]]
        oldMolesMax = self.molesMax[oldkk]
        self.updateMolesShrinkGrowth(memkeys, other)
        self.dVg_dPc[memkeys] = (self.volMax[memkeys]-self.volShrink[memkeys])/(
            self.pcMax[memkeys]-self.pcShrink[memkeys])
        self.molesShrink[newkk] = np.minimum(self.molesShrink[newkk], oldMolesMax)
        self.molesShrink[memkeys] = np.clip(np.minimum(
                self.molesShrink[memkeys], self.moles[memkeys]-1e-22), 1e-22, None)

        # determine the new volume and pc of the clusters
        keysVol = np.zeros(self.pc.size, dtype=bool)
        keysVol[memkeys] = True        
        newMoles = self.moles.copy()
        tD.updateClusterPcVolume(other, mem, keysVol, self, newkeys, newMoles, True)
        self.moles[keysVol] = newMoles[keysVol]
    
        other.gasConc[mem,1] = other.H*self.pc[newkeys[mem]]
        [self.updateToDrainImbibe(k, other) for k in np.setdiff1d(keys, memkeys)]
        for k in memkeys:
            if self.members[k][other.conTToExit].any():
                self.trappedStatus[k] = False
                other.trappedNW[self[k].members] = False
        
        return toDrain


    def _func1(self, mem, arrPc, other):
        self.computeCornerArea(mem, arrPc, other)
        vol = other.areaWPhase[mem]/other.areaSPhase[mem]*other.volarray[mem]
        return vol.sum()
            
    def _func2(self, x, mem, totalVol, moles, elemPc, other):
        elemPc[mem] = x
        vol = self._func1(mem, elemPc, other)
        molesS = ((other.imposedP+x)*vol/(other.R*other.T) + other.H*x*(totalVol-vol))
        return moles-molesS

    def _func3(self, x0, mem, totalVol, moles, elemPc, other, step=-1000):
        """Expand the interval until f(x) changes sign."""
        x00 = x0
        while step<-1:
            x0 = x00
            x1 = x0 + step
            fx0 = self._func2(x0, mem, totalVol, moles, elemPc, other)
            fx1 = self._func2(x1, mem, totalVol, moles, elemPc, other)
            print(x0, fx0, x1, fx1, step)
            
            while (x1>=0.0):
                if fx0 * fx1 < 0:  # Found sign change
                    return x0, x1
                
                x0, x1 = x1, x1 + step
                fx0, fx1 = fx1, self._func2(x1, mem, totalVol, moles, elemPc, other)

            step/=10
           
        print("Failed to find a valid interval.")
        return x0

    def resizeArrays(self, other):
        ct = self.pc.size - self.neighbours.shape[0]
        self.neighbours = np.vstack(
            (self.neighbours, np.zeros([ct,other.totElements], dtype=bool)))
        self.toDrain = np.concatenate((self.toDrain, np.full(ct, other.totElements+1, dtype=int)))
        self.toImbibe = np.concatenate((self.toImbibe, np.full(ct, other.totElements+1, dtype=int)))
        self.pcMax = np.concatenate((self.pcMax, np.full(ct, 1e30, dtype=float)))
        self.pcShrink = np.concatenate((self.pcShrink, np.full(ct, -1e30, dtype=float)))
        self.moles = np.concatenate((self.moles, np.zeros(ct, dtype=float)))
        self.molesMax = np.concatenate((self.molesMax, np.full(ct, 1e30, dtype=float)))
        self.molesShrink = np.concatenate((self.molesShrink, np.zeros(ct, dtype=float)))
        self.volume = np.concatenate((self.volume, np.zeros(ct, dtype=float)))
        self.volMax = np.concatenate((self.volMax, np.zeros(ct, dtype=float)))
        self.volShrink = np.concatenate((self.volShrink, np.zeros(ct, dtype=float)))
        other.initialMoles = np.concatenate((other.initialMoles, np.zeros(ct, dtype=float)))
        other.initialPc = np.concatenate((other.initialPc, np.zeros(ct, dtype=float)))
        other.initialVolume = np.concatenate((other.initialVolume, np.zeros(ct, dtype=float)))
        self.totalVolume = np.concatenate((self.totalVolume, np.zeros(ct, dtype=float)))
        self.dVg_dPc = np.concatenate((self.dVg_dPc, np.zeros(ct, dtype=float)))
        

def mapOldNewKeys(oldkeys, newkeys):
    ''' returns unique pairs of old and new keys '''
    i=0
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








    



