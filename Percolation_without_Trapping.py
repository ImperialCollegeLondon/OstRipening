import os
import numpy as np
from sortedcontainers import SortedList
from time import time

from pnflowPy.tPhaseD import TwoPhaseDrainage
from pnflowPy.tPhaseImb import TwoPhaseImbibition


class PDrainage(TwoPhaseDrainage):
    cycle = 0
    def __new__(cls, obj, writeData=False, writeTrappedData=True):
        obj.__class__ = PDrainage
        return obj
    
    def __init__(self, obj, writeData=False, writeTrappedData=True):
        super().__init__(obj, writeData=writeData)
        if self.writeData: self.__fileName__()
        self.prevFilled = (self.fluid==1)
        self.writeTrappedData = writeTrappedData
    
    def popUpdateOilInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcD[k]
        self.capPresMax = np.max([self.capPresMax, capPres])

        self.fluid[k] = 1
        self.hasNWFluid[k] = True
        arr = self.elem[k].neighbours[self.elem[k].neighbours>0]
        arr0 = arr[(self.fluid[arr]==0) & ~(self.trappedW[arr])]
        arr1 = arr[self.hasNWFluid[arr]]
        self.PistonPcRec[k] = self.centreEPOilInj[k]

        ''' check whether the just filled element is a circle '''
        try:
            assert self.isCircle[k]
            self.hasWFluid[k] = False
            kk = self.clusterW_ID[k]
            self.clusterW.members[kk,k] = False
            self.clusterW_ID[k] = -5
            try:
                assert not self.clusterW.members[kk].any()
                self.clusterW.availableID.update([kk])
            except AssertionError:
                assert arr0.size>0
                self.do.check_Trapping_Clustering(
                   arr0.copy(), self.hasWFluid.copy(), 0, self.capPresMax, True)
        except AssertionError:
            pass

        ''' update the clustering details of the just filled element '''
        self.quickClustering(k, arr1, capPres)
        self.__update_PcD_ToFill__(arr0)
        self.cnt += 1
        self.invInsideBox += self.isinsideBox[k]

    
    def quickTrapping(self, arr0, capPres):
        ''' write code here'''
        pass


    def quickClustering(self, k, arr1, capPres):
        try:
            ids = self.clusterNW_ID[arr1]
            kk = ids.min()
            ''' newly filled takes the properties of already filled neighbour '''
            self.connNW[k] = self.clusterNW[kk].connected
            self.clusterNW_ID[k] = kk
            self.clusterNW.members[kk, k] = True
            arr1 = arr1[ids!=kk]
            assert arr1.size>0
            ''' need to coalesce '''
            ids = ids[ids!=kk]
            mem = self.elementListS[self.clusterNW.members[ids].any(axis=0)]
            self.clusterNW_ID[mem] = kk
            self.clusterNW.members[ids] = False
            self.clusterNW.members[kk, mem] = True
            self.connNW[self.clusterNW[kk].members] = (
                self.clusterNW.connected[kk] or self.clusterNW.connected[ids].any())
            self.clusterNW.availableID.update(ids)
        except ValueError:
            ''' create a new cluster for k'''
            mem = np.zeros(self.totElements, dtype=bool)
            mem[k] = True
            arrDict = {}
            trappedStatus = ~(self.elem[k].neighbours<=0).any()
            arrDict[1] = {'members': mem, 'connStatus': False, 'trappedStatus': trappedStatus}
            self.clusterNW.clustering(
                arrDict, capPres, self.clusterNW_ID, self.clusterNW, self.trappedNW)
        except AssertionError:
            pass
    
    def __fileName__(self):
        result_dir = "./results_csv/"
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        if not hasattr(self, '_num'):
            self._num = 1
            while True:         
                file_name = os.path.join(
                    result_dir, "Flowmodel_"+self.title+"_Drainage_cycle"+str(self.cycle)+\
                                "_ostRip_"+str(self._num)+".csv")
                if os.path.isfile(file_name): self._num += 1
                else:
                    break
            self.file_name = file_name
        else:
            self.file_name = os.path.join(
                result_dir, "Flowmodel_"+self.title+"_Drainage_cycle"+str(self.cycle)+\
                    "_ostRip_"+str(self._num)+".csv")
            

    def __writeTrappedData__(self):
        filename = os.path.join(
            "./results_csv/Flowmodel_{}_Drainage_cycle{}_ostRip_trappedDist_{}.csv".format(
                self.title, self.cycle, self._num))
        data = [*zip(self.Rarray, self.volarray, self.fluid, self.trappedW, self.trappedNW)]
        np.savetxt(filename, data, delimiter=',', header='rad, volume, fluid, trappedW, trappedNW')


class SecDrainage(PDrainage):
    def __new__(cls, obj, writeData=False, writeTrappedData=True):
        obj.__class__ = SecDrainage
        return obj
    
    def __init__(self, obj, writeData=False, writeTrappedData=True):
        self.fluid[[-1, 0]] = 1, 0        
        self.capPresMax = self.capPresMin
        self.is_oil_inj = True
        self.contactAng, self.thetaRecAng, self.thetaAdvAng = self.prop_drainage.values()
        
        self.do.__initCornerApex__()
        self.Fd_Tr = self.do.__computeFd__(self.elemTriangle, self.halfAnglesTr)
        self.Fd_Sq = self.do.__computeFd__(self.elemSquare, self.halfAnglesSq)
        self.__computePistonPc__()
        self.centreEPOilInj[self.elementLists] = 2*self.sigma*np.cos(
           self.thetaRecAng[self.elementLists])/self.Rarray[self.elementLists]
        self.PcD[:] = self.PistonPcRec
        self.PistonPcRec[self.fluid==1] = self.centreEPOilInj[self.fluid==1]
        self.ElemToFill = SortedList(key=lambda i: self.LookupList(i))
        self.NinElemList[:] = True
        self.initializeToFill()  

        self._cornArea = self._areaWP.copy()
        self._centerArea = self._areaNWP.copy()
        self._cornCond = self._condWP.copy()
        self._centerCond = self._condNWP.copy()

        self.cycle += 1
        self.writeData = writeData
        if self.writeData: self.__fileName__()
        self.primary = False
        self.prevFilled = (self.fluid==1)
        self.writeTrappedData = writeTrappedData
        self.totNumFill = 0

    def initializeToFill(self):
        condlist1 = np.zeros(self.totElements-2, dtype='bool')
        def _f(i):
            return self.hasNWFluid[self.elem[i].neighbours].any()
        condlist1[(self.fluid[1:-1]==0)] = np.array(
            [*map(lambda i: _f(i), self.elementLists[(self.fluid[1:-1]==0)])])
        ElemToFill = self.elementLists[condlist1]

        self.__update_PcD_ToFill__(ElemToFill)

            
class PImbibition(TwoPhaseImbibition):
    def __new__(cls, obj, writeData=False, writeTrappedData=True):
        obj.__class__ = PImbibition
        return obj
    
    def __init__(self, obj, writeData=False, writeTrappedData=True):
        super().__init__(obj, writeData=writeData, trapping=False)
        self.writeData = writeData
        self.writeTrappedData = writeTrappedData


    def popUpdateWaterInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcI[k]
        self.capPresMin = np.min([self.capPresMin, capPres])

        self.fluid[k] = 0
        self.hasNWFluid[k] = False
        kk = self.clusterNW_ID[k]
        self.clusterNW_ID[k] = -5
        self.clusterNW.members[kk,k] = False
        self.trappedNW[k] = False

        ''' update the trapping status of the cluster kk'''
        try:
            mem = self.clusterNW.members[kk]
            assert mem.any()
            trappedStatus = (~mem[self.conTToExit].any())
            self.trappedNW[self.clusterNW[kk].members] = trappedStatus
            self.clusterNW.trappedStatus[kk] = trappedStatus
        except AssertionError:
            self.clusterNW.availableID.update([kk])
        
        neigh = self.elem[k].neighbours[self.elem[k].neighbours>0]
        try:
            assert not self.hasWFluid[k]
            self.hasWFluid[k] = True
            neighW = neigh[self.hasWFluid[neigh]]
            ids = self.clusterW_ID[neighW]
            ii = ids.min()

            ''' newly filled takes the properties of already filled neighbour '''
            self.clusterW_ID[k] = ii
            self.clusterW.members[ii,k] = True
            self.connW[k] = self.clusterW[ii].connected
            ids = ids[ids!=ii]    
            assert ids.size>0

            ''' need to coalesce '''
            mem = self.elementListS[self.clusterW.members[ids].any(axis=0)]
            self.clusterW.members[ii][mem] = True
            self.clusterW.members[ids] = False
            self.clusterW_ID[mem] = ii
        except AssertionError:
            pass

        self.fillmech[k] = 1*(self.PistonPcAdv[k]==capPres)+2*(
            self.porebodyPc[k]==capPres)+3*(self.snapoffPc[k]==capPres)
        self.cnt += 1
        self.invInsideBox += self.isinsideBox[k]
        neighb = neigh[self.hasNWFluid[neigh]]
        
        self.do.check_Trapping_Clustering(
            neighb.copy(), self.hasNWFluid.copy(), 1, self.capPresMin,True)
        prev = len(self.ElemToFill)
        self.__computePc__(self.capPresMin, neighb, trapping=False)            


    def __PImbibition__(self):
        self.totNumFill = 0
        while (self.PcTarget-1.0e-32 < self.capPresMin) & (
                self.satW <= self.SwTarget):
            self.oldSatW = self.satW
            self.invInsideBox = 0
            self.cnt = 0
            try:
                while (self.invInsideBox < self.fillTarget) & (
                    len(self.ElemToFill) != 0) & (
                        self.PcI[self.ElemToFill[0]] >= self.PcTarget):
                    try:
                        assert (self.clusterNW.members[0][self.conTToInletBdr].any() and 
                                    self.clusterNW.members[0][self.conTToOutletBdr].any())
                        self.popUpdateWaterInj()
                    except AssertionError:
                        self.filling = False
                        self.PcTarget = self.capPresMin
                        break
                        
                assert (self.PcI[self.ElemToFill[0]] < self.PcTarget) & (
                        self.capPresMin > self.PcTarget)
                self.capPresMin = self.PcTarget
            except IndexError:
                self.capPresMin = min(self.capPresMin, self.PcTarget)
            except AssertionError:
                pass

            self.__CondTPImbibition__()
            self.satW = self.do.Saturation(self.areaWPhase, self.areaSPhase)
            self.totNumFill += self.cnt

            try:
                assert self.PcI[self.ElemToFill[0]] >= self.PcTarget
                assert self.filling
            except (AssertionError, IndexError):
                break
            
        try:
            assert (self.PcI[self.ElemToFill[0]] < self.PcTarget) & (
                self.capPresMin > self.PcTarget)
            self.capPresMin = self.PcTarget
        except AssertionError:
            self.PcTarget = self.capPresMin
        except IndexError:
            pass
        
        self.__CondTPImbibition__()
        self.satW = self.do.Saturation(self.areaWPhase, self.areaSPhase)
        self.do.computePerm(self.capPresMin)
        self.resultI_str = self.do.writeResult(self.resultI_str, self.capPresMin)
        
    def __fileName__(self):
        result_dir = "./results_csv/"
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        if not hasattr(self, '_num'):
            self._num = 1
            while True:         
                file_name = os.path.join(
                    result_dir, "Flowmodel_"+self.title+"_Imbibition_cycle"+str(self.cycle)+\
                        "_ostRip_"+str(self._num)+".csv")
                if os.path.isfile(file_name): self._num += 1
                else:
                    break
            self.file_name = file_name
        else:
            self.file_name = os.path.join(
                result_dir, "Flowmodel_"+self.title+"_Imbibition_cycle"+str(self.cycle)+\
                    "_ostRip_"+str(self._num)+".csv") 
            

    def __writeTrappedData__(self):
        filename = os.path.join(
            "./results_csv/Flowmodel_{}_Imbibition_cycle{}_ostRip_trappedDist_{}.csv".format(
                self.title, self.cycle, self._num))
        data = [*zip(self.Rarray, self.volarray, self.fluid, self.trappedW, self.trappedNW)]
        np.savetxt(filename, data, delimiter=',', header='rad, volume, fluid, trappedW, trappedNW')


class SecImbibition(PImbibition):
    def __new__(cls, obj, writeData=False, writeTrappedData=True):
        obj.__class__ = SecImbibition
        return obj
    
    def __init__(self, obj, writeData=False, writeTrappedData=True):    
        self.fluid[[-1, 0]] = 0, 1  
        self.ElemToFill = SortedList(key=lambda i: self.LookupList(i))
        self.capPresMin = self.maxPc
        
        self.contactAng, self.thetaRecAng, self.thetaAdvAng = self.prop_imbibition.values()
        self.is_oil_inj = False

        self.do.__initCornerApex__()
        self.__computePistonPc__()
        self.__computePc__(self.maxPc, self.elementLists, trapping=False)

        self._areaWP = self._cornArea.copy()
        self._areaNWP = self._centerArea.copy()
        self._condWP = self._cornCond.copy()
        self._condNWP = self._centerCond.copy()

        self.writeData = writeData
        if self.writeData: self.__fileName__()        
        self.writeTrappedData = writeTrappedData

        from IPython import embed; embed()


