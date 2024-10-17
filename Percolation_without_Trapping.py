import os
import numpy as np
from sortedcontainers import SortedList
import warnings
from itertools import chain
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


    def drainage(self):
        start = time()
        print('---------------------------------------------------------------------------')
        print('-------------------------Two Phase Drainage Cycle {}------------------------'.format(self.cycle))

        if self.writeData:
            self.__fileName__()
            self.__writeHeadersD__()
        else: self.resultD_str = ""

        self.SwTarget = max(self.finalSat, self.satW-self.dSw*0.5)
        self.PcTarget = min(self.maxPc, self.capPresMax+(
            self.minDeltaPc+abs(
             self.capPresMax)*self.deltaPcFraction)*0.1)
        self.oldPcTarget = 0
        self.resultD_str = self.do.writeResult(self.resultD_str, self.capPresMin)

        while self.filling:
            self.oldSatW = self.satW
            self.__PDrainage__()
            
            if (self.PcTarget > self.maxPc-0.001) or (
                 self.satW < self.finalSat+0.00001):
                self.filling = False
                break
            
            self.oldPcTarget = self.capPresMax
            self.PcTarget = min(self.maxPc+1e-7, self.PcTarget+(
                self.minDeltaPc+abs(self.PcTarget)*self.deltaPcFraction))
            self.SwTarget = max(self.finalSat-1e-15, round((
                self.satW-self.dSw*0.75)/self.dSw)*self.dSw)

            if len(self.ElemToFill) == 0:
                self.filling = False
                self.cnt, self.totNumFill = 0, 0

                while (self.PcTarget < self.maxPc-1e-8) and (self.satW>self.finalSat):
                    self.__CondTP_Drainage__()
                    self.satW = self.do.Saturation(self.areaWPhase, self.areaSPhase)
                    self.do.computePerm(self.capPresMax)
                    self.resultD_str = self.do.writeResult(self.resultD_str, self.capPresMax)

                    self.PcTarget = min(self.maxPc-1e-7, self.PcTarget+(
                        self.minDeltaPc+abs(self.PcTarget)*self.deltaPcFraction))
                    if self.capPresMax == self.PcTarget: break
                    else: self.capPresMax = self.PcTarget
                   
                break

        if self.writeData:
            with open(self.file_name, 'a') as fQ:
                fQ.write(self.resultD_str)
            if self.writeTrappedData:
                self.__writeTrappedData__()

        self.maxPc = self.capPresMax
        self.rpd = self.sigma/self.maxPc
        print("Number of trapped elements: W: {}  NW:{}".format(
            self.trappedW.sum(), self.trappedNW.sum()))
        print('No of W clusters: {}, No of NW clusters: {}'.format(
            np.count_nonzero(self.clusterW.size), 
            np.count_nonzero(self.clusterNW.size)))
        self.is_oil_inj = False
        self.do.__finitCornerApex__(self.capPresMax)
        print('Time spent for the drainage process: ', time() - start)        
        print('==========================================================\n\n')
        #from IPython import embed; embed()
    

    def popUpdateOilInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcD[k]
        self.capPresMax = np.max([self.capPresMax, capPres])
        try:
            self.fluid[k] = 1
            self.hasNWFluid[k] = True
            self.connNW[k] = True
            self.clusterNW_ID[k] = 0
            self.clusterNW.members[0, k] = True
            self.PistonPcRec[k] = self.centreEPOilInj[k]
            arr = self.elem[k].neighbours[self.elem[k].neighbours>0]
            self.cnt += 1
            self.invInsideBox += self.isinsideBox[k]
            arr2 = arr[self.hasWFluid[arr]]
            self.__update_PcD_ToFill__(arr2)            
        except AssertionError:
            pass
        
    
    def __PDrainage__(self):
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        self.totNumFill = 0
        self.fillTarget = max(self.m_minNumFillings, int(
            self.m_initStepSize*(self.totElements)*(
             self.SwTarget-self.satW)))
        self.invInsideBox = 0
        
        while (self.PcTarget+1.0e-32 > self.capPresMax) & (
                self.satW > self.SwTarget):
            self.oldSatW = self.satW
            self.invInsideBox = 0
            self.cnt = 0
            try:
                while (self.invInsideBox < self.fillTarget) & (
                    len(self.ElemToFill) != 0) & (
                        self.PcD[self.ElemToFill[0]] <= self.PcTarget):
                    self.popUpdateOilInj()
            except IndexError:
                self.totNumFill += self.cnt
                break

            try:
                assert (self.PcD[self.ElemToFill[0]] > self.PcTarget) & (
                        self.capPresMax < self.PcTarget)
                self.capPresMax = self.PcTarget
            except AssertionError:
                pass
            
            self.__CondTP_Drainage__()
            self.satW = self.do.Saturation(self.areaWPhase, self.areaSPhase)
            self.totNumFill += self.cnt
            try:
                self.fillTarget = max(self.m_minNumFillings, int(min(
                    self.fillTarget*self.m_maxFillIncrease,
                    self.m_extrapCutBack*(self.invInsideBox / (
                        self.satW-self.oldSatW))*(self.SwTarget-self.satW))))
            except OverflowError:
                pass
                
            try:
                assert self.PcD[self.ElemToFill[0]] <= self.PcTarget
            except AssertionError:
                break

        try:
            assert (self.PcD[self.ElemToFill[0]] > self.PcTarget)
            self.capPresMax = self.PcTarget
        except (AssertionError, IndexError):
            self.PcTarget = self.capPresMax
        
        self.__CondTP_Drainage__()
        self.satW = self.do.Saturation(self.areaWPhase, self.areaSPhase)
        self.do.computePerm(self.capPresMax)
        self.resultD_str = self.do.writeResult(self.resultD_str, self.capPresMax)
        

    
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
        
        self.trapped = self.trappedW
        self.trappedData = self.clusterW
        self.trapClust = self.clusterW_ID
    
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


    def popUpdateOilInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcD[k]
        self.capPresMax = np.max([self.capPresMax, capPres])

        try:
            self.fluid[k] = 1
            self.hasNWFluid[k] = True
            self.connNW[k] = True
            self.clusterNW_ID[k] = 0
            self.clusterNW.members[0, k] = True
            self.PistonPcRec[k] = self.centreEPOilInj[k]
            arr = self.elem[k].neighbours[self.elem[k].neighbours>0]
            arr0 = arr[self.hasWFluid[arr]]
            arr1 = arr[self.hasNWFluid[arr]&(self.clusterNW_ID[arr]>0)]
            try:
                assert arr1.size>0
                ids = self.clusterNW_ID[arr1]
                mem = self.elementListS[self.clusterNW.members[ids].any(axis=0)]
                self.connNW[mem] = True
                self.clusterNW_ID[mem] = 0
                self.clusterNW.members[0, mem] = True
                self.populateToFill(mem)
            except:
                self.__update_PcD_ToFill__(arr0)

            self.cnt += 1
            self.invInsideBox += self.isinsideBox[k]
        except (AssertionError, IndexError):
            pass


    def untrapNWElements(self, ind):
        idx = self.trapCluster_NW[ind]
        arrr = np.zeros(self.totElements, dtype='bool')
        arrr[ind[idx==0]] = True
        idx = idx[idx>0]
        while True:
            try:
                i = idx[0]
                arrr[(self.trapCluster_NW==i)] = True
                idx = idx[idx!=i]
            except IndexError:
                break
        try:
            cond = arrr & self.trappedNW
            assert np.any(cond)
            self.trapCluster_NW[arrr] = 0
        except AssertionError:
            pass
        
        self.connNW[arrr] = True
        self.trappedNW[arrr] = False
        self.prevFilled[arrr] = False
        self.populateToFill(self.elementLists[arrr[1:-1]])

    
    def identifyTrappedElements(self):
        Notdone = (self.fluid==1)
        tin = list(self.conTToIn[Notdone[self.conTToIn+self.nPores]])
        tout = self.conTToOut[Notdone[self.conTToOut+self.nPores]]
        self.trappedNW[:] = True
        conn = np.zeros(self.totElements, dtype='bool')

        while True:
            try:
                conn[:] = False
                tt = tin.pop(0)
                Notdone[tt+self.nPores] = False
                conn[tt+self.nPores] = True
                while True:
                    try:
                        pp = np.array([self.P1array[tt-1], self.P2array[tt-1]])
                        pp = pp[Notdone[pp]]
                        Notdone[pp] = False
                        conn[pp] = True

                        tt = np.array([*chain(*self.PTConData[pp])])
                        tt = tt[Notdone[tt+self.nPores]]
                        Notdone[tt+self.nPores] = False
                        conn[tt+self.nPores] = True
                    except IndexError:
                        try:
                            tin = np.array(tin)
                            tin = list(tin[Notdone[tin]])
                        except IndexError:
                            tin=[]
                        break
                if any(conn[tout]):
                    self.trappedNW[conn] = False
                
            except IndexError:
                break


    def initializeToFill(self):
        condlist1 = np.zeros(self.totElements-2, dtype='bool')
        def _f(i):
            return self.fluid[self.elem[i].neighbours].sum()>0
        condlist1[(self.fluid[1:-1]==0)] = np.array(
            [*map(lambda i: _f(i), self.elementLists[(self.fluid[1:-1]==0)])])
        ElemToFill = self.elementLists[condlist1]

        self.__update_PcD_ToFill__(ElemToFill)


    def populateToFill(self, arr):
        done = np.zeros(self.totElements, dtype='bool')
        fluid0 = (self.fluid==0)
        fluid1 = (self.fluid==1)
        arr = arr[fluid0[arr]]
        done[arr] = True
        ElemToFill = done.copy()
        done[[-1,0]] = True
        
        temp = np.zeros(self.totElements, dtype='bool')
        while True:
            temp[self.PTConnections[arr[arr<=self.nPores]]] = True
            temp[self.TPConnections[arr[arr>self.nPores]-self.nPores]] = True
            temp[done] = False
            ElemToFill[np.where(temp & fluid0)] = True
            arr = np.where(temp & fluid1)[0]
            if not any(arr):
                break            
            done[temp] = True

        ElemToFill= np.where(ElemToFill)[0]
        self.__update_PcD_ToFill__(ElemToFill)


    def renumCluster(self):
        numOld = np.unique(self.trapCluster_NW)
        for ind, v in enumerate(numOld):
            self.trapCluster_NW[self.trapCluster_NW==v] = ind
         
            
class PImbibition(TwoPhaseImbibition):
    def __new__(cls, obj, writeData=False, writeTrappedData=True):
        obj.__class__ = PImbibition
        return obj
    
    def __init__(self, obj, writeData=False, writeTrappedData=True):
        super().__init__(obj, writeData=writeData)
        self.writeData = writeData
        self.trapped = self.trappedNW
        self.trappedData = self.clusterNW
        self.trapClust = self.clusterNW_ID
        self.writeTrappedData = writeTrappedData
        print(self.contactAng)
        print(self.contactAng.sum())


    def imbibition(self):
        start = time()
        print('----------------------------------------------------------------------------------')
        print('---------------------------------Two Phase Imbibition Cycle {}---------------------'.format(self.cycle))

        if self.writeData:
            self.__fileName__()
            self.__writeHeadersI__()
        else:
            self.resultI_str = ""
            self.totNumFill = 0
            self.resultI_str = self.do.writeResult(self.resultI_str, self.capPresMax)
            
        self.SwTarget = min(self.finalSat, self.satW+self.dSw*0.5)
        self.PcTarget = max(self.minPc, self.capPresMin-(
            self.minDeltaPc+abs(self.capPresMin)*self.deltaPcFraction)*0.1)
        self.fillTarget = max(self.m_minNumFillings, int(
            self.m_initStepSize*(self.totElements)*(
                self.satW-self.SwTarget)))

        while self.filling:
            self.__PImbibition__()

            if (self.PcTarget < self.minPc+0.001) or (
                 self.satW > self.finalSat-0.00001):
                self.filling = False
                break

            if (len(self.ElemToFill)==0):
                self.filling = False
                self.cnt, self.totNumFill = 0, 0
                _pclist = np.array([-1e-7, self.minPc])
                _pclist = np.sort(_pclist[_pclist<self.capPresMin])[::-1]
                for Pc in _pclist:
                    self.capPresMin = Pc
                    self.__CondTPImbibition__()
                    self.satW = self.do.Saturation(self.areaWPhase, self.areaSPhase)
                    self.do.computePerm(self.capPresMin)
                    self.resultI_str = self.do.writeResult(self.resultI_str, self.capPresMin)
                    
                break

            self.PcTarget = max(self.minPc+1e-7, self.PcTarget-(
                self.minDeltaPc+abs(
                    self.PcTarget)*self.deltaPcFraction+1e-16))
            self.SwTarget = min(self.finalSat+1e-15, round((
                self.satW+self.dSw*0.75)/self.dSw)*self.dSw)

        if self.writeData:
            with open(self.file_name, 'a') as fQ:
                fQ.write(self.resultI_str)
            if self.writeTrappedData:
                self.__writeTrappedData__()

        
        print("Number of trapped elements: W: {}  NW:{}".format(
            self.trappedW.sum(), self.trappedNW.sum()))
        print('No of W clusters: {}, No of NW clusters: {}'.format(
            np.count_nonzero(self.clusterW.size),
            np.count_nonzero(self.clusterNW.size)))
        self.is_oil_inj = True
        self.do.__finitCornerApex__(self.capPresMin)
        print('Time spent for the imbibition process: ', time() - start)
        print('===========================================================\n\n')
        #from IPython import embed; embed()


    def popUpdateWaterInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcI[k]
        self.capPresMin = np.min([self.capPresMin, capPres])

        try:
            self.fluid[k] = 0
            self.hasWFluid[k] = True
            self.hasNWFluid[k] = False
            kk = self.clusterNW_ID[k]
            self.clusterNW_ID[k] = -5
            self.clusterNW.members[kk,k] = False

            neigh = self.elem[k].neighbours[self.elem[k].neighbours>0]
            neighW = neigh[self.hasWFluid[neigh]]
            ids = self.clusterW_ID[neighW]
            try:
                idmin = ids.min()
                ids = ids[ids!=idmin]
                assert ids.size>0
                mem = self.elementListS[self.clusterW.members[ids].any(axis=0)]
                self.clusterW.members[idmin][mem] = True
                self.clusterW.members[ids] = False
                self.clusterW_ID[mem] = idmin
            except (AssertionError, ValueError):
                pass

            self.fillmech[k] = 1*(self.PistonPcAdv[k]==capPres)+2*(
                self.porebodyPc[k]==capPres)+3*(self.snapoffPc[k]==capPres)
            self.cnt += 1
            self.invInsideBox += self.isinsideBox[k]
            
            neighb = neigh[self.hasNWFluid[neigh]]
            self.do.check_Trapping_Clustering(
                neighb, self.hasNWFluid.copy(), 1, self.capPresMin,True)
            self.__computePc__(self.capPresMin, neighb)
        except AssertionError:
            pass


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
                        assert self.connNW.sum()>0
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
        

    def isNWConnected(self):
        Notdone = (self.fluid==1)&(self.isinsideBox)&(~self.trappedNW)
        conTToInlet = self.conTToInlet+self.nPores
        conTToOutlet = self.conTToOutlet+self.nPores
        try:
            assert Notdone[conTToInlet].sum()>0
            assert Notdone[conTToOutlet].sum()>0
            conTToOutlet = conTToOutlet[Notdone[conTToOutlet]]
        except AssertionError:
            return False
        
        arrlist = SortedList(key=lambda i: self.distToExit[i])
        arrlist.update(conTToInlet[Notdone[conTToInlet]])

        while True:
            try:
                i = arrlist.pop(0)
                Notdone[i] = False
                arr = self.elem[i].neighbours
                arr = arr[Notdone[arr]]
                Notdone[arr] = False
                assert (~Notdone[conTToOutlet]).sum()==0
                arrlist.update(arr)
            except AssertionError:
                return True
            except IndexError:
                return False


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
        self.trapped = self.trappedNW
        self.trappedData = self.clusterNW
        self.trapClust = self.clusterNW_ID

        self.do.__initCornerApex__()
        self.__computePistonPc__()
        self.__computePc__(self.maxPc, self.elementLists, False)

        self._areaWP = self._cornArea.copy()
        self._areaNWP = self._centerArea.copy()
        self._condWP = self._cornCond.copy()
        self._condNWP = self._centerCond.copy()

        self.writeData = writeData
        if self.writeData: self.__fileName__()        
        self.writeTrappedData = writeTrappedData




    




