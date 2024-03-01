import os
import numpy as np
from sortedcontainers import SortedList
import warnings
from itertools import chain

from Flowmodel.tPhaseD import TwoPhaseDrainage
from Flowmodel.tPhaseImb import TwoPhaseImbibition


class PDrainage(TwoPhaseDrainage):
    cycle = 0
    def __new__(cls, obj, writeData=False, includeTrapping=True, writeTrappedData=True):
        obj.__class__ = PDrainage
        return obj
    
    def __init__(self, obj, writeData=False, includeTrapping=True, writeTrappedData=True):
        super().__init__(obj, writeData=writeData)
        self.includeTrapping = includeTrapping
        self.cycle += 1
        if self.writeData: self.__fileName__()
        self.prevFilled = (self.fluid==1)
        self.writeTrappedData = writeTrappedData

    
    def popUpdateOilInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcD[k]
        self.capPresMax = np.max([self.capPresMax, capPres])

        try:
            self.do.isTrapped(k, 0, self.capPresMax)
            self.fluid[k] = 1
            self.PistonPcRec[k] = self.centreEPOilInj[k]
            arr = self.elem[k].neighbours
            arr = arr[(self.fluid[arr] == 0) & ~(self.trappedW[arr]) & (arr>0)]
            [*map(lambda i: self.do.isTrapped(i, 0, self.capPresMax), arr)]

            self.cnt += 1
            self.invInsideBox += self.isinsideBox[k]
            self.__update_PcD_ToFill__(arr)
        except AssertionError:
            pass

        
    def __writeHeadersD__(self):
        self.resultD_str="======================================================================\n"
        self.resultD_str+="# Fluid properties:\nsigma (mN/m)  \tmu_w (cP)  \tmu_nw (cP)\n"
        self.resultD_str+="# \t%.6g\t\t%.6g\t\t%.6g" % (
            self.sigma*1000, self.muw*1000, self.munw*1000, )
        self.resultD_str+="\n# calcBox: \t %.6g \t %.6g" % (
            self.calcBox[0], self.calcBox[1], )
        self.resultD_str+="\n# Wettability:"
        self.resultD_str+="\n# model \tmintheta \tmaxtheta \tdelta \teta \tdistmodel"
        self.resultD_str+="\n# %.6g\t\t%.6g\t\t%.6g\t\t%.6g\t\t%.6g" % (
            self.wettClass, round(self.minthetai*180/np.pi,3), round(self.maxthetai*180/np.pi,3), self.delta, self.eta,) 
        self.resultD_str+=self.distModel
        self.resultD_str+="\nmintheta \tmaxtheta \tmean  \tstd"
        self.resultD_str+="\n# %3.6g\t\t%3.6g\t\t%3.6g\t\t%3.6g" % (
            round(self.contactAng.min()*180/np.pi,3), round(self.contactAng.max()*180/np.pi,3), 
            round(self.contactAng.mean()*180/np.pi,3), round(self.contactAng.std()*180/np.pi,3))
        
        self.resultD_str+="\nPorosity:  %3.6g" % (self.porosity)
        self.resultD_str+="\nMaximum pore connection:  %3.6g" % (self.maxPoreCon)
        self.resultD_str+="\nAverage pore-to-pore distance:  %3.6g" % (self.avgP2Pdist)
        self.resultD_str+="\nMean pore radius:  %3.6g" % (self.Rarray[self.poreList].mean())
        self.resultD_str+="\nAbsolute permeability:  %3.6g" % (self.absPerm)
        
        self.resultD_str+="\n======================================================================"
        self.resultD_str+="\n# Sw\t qW(m3/s)\t krw\t qNW(m3/s)\t krnw\t Pc\t Invasions"


    def __fileName__(self):
        result_dir = "./results_csv/"
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        self.label = 'wt' if self.includeTrapping else 'nt'
        if not hasattr(self, '_num'):
            self._num = 1
            while True:         
                file_name = os.path.join(
                    result_dir, "Flowmodel_"+self.title+"_Drainage_cycle"+str(self.cycle)+\
                                "_"+self.label+"_"+str(self._num)+".csv")
                if os.path.isfile(file_name): self._num += 1
                else:
                    break
            self.file_name = file_name
        else:
            self.file_name = os.path.join(
                result_dir, "Flowmodel_"+self.title+"_Drainage_cycle"+str(self.cycle)+\
                    "_"+self.label+"_"+str(self._num)+".csv")


class SecDrainage(PDrainage):
    def __new__(cls, obj, writeData=False, includeTrapping=True, writeTrappedData=True):
        obj.__class__ = SecDrainage
        return obj
    
    def __init__(self, obj, writeData=False, includeTrapping=True, writeTrappedData=True):
        self.fluid[[-1, 0]] = 1, 0  
        self.includeTrapping = includeTrapping
        
        self.capPresMax = self.capPresMin
        self.is_oil_inj = True
        self.contactAng, self.thetaRecAng, self.thetaAdvAng = self.prop_drainage.values()
        self.trapped = self.trappedW
        self.trappedPc = self.trappedW_Pc
        self.trapClust = self.trapCluster_W
        self.trapdata = (self.trappedNW, self.trappedNW_Pc, self.trapCluster_NW)
        
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

    
    def popUpdateOilInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcD[k]
        self.capPresMax = np.max([self.capPresMax, capPres])
    
        try:
            self.trapped[k] = False
            self.trappedPc[k] = 0.0
            self.trapClust[k] = 0

            self.fluid[k] = 1
            self.PistonPcRec[k] = self.centreEPOilInj[k]
            arr = self.elem[k].neighbours
            ar1 = arr[(self.fluid[arr]==0)&(arr>0)]
            [*map(lambda i: self.do.isTrapped(i, 0, self.capPresMax), ar1)]

            self.do.isTrapped(k, 1, self.capPresMax, self.trapdata)
            ar2 = arr[(self.fluid[arr]==1)&(arr>0)&(self.prevFilled[arr])]

            try:
                assert ar2.size > 0
                try:
                    assert (not self.trappedNW[k])
                    self.untrapNWElements(ar2)
                except AssertionError:
                    self.populateToFill(arr)
            except AssertionError:
                pass
            

            self.cnt += 1
            self.invInsideBox += self.isinsideBox[k]
            self.__update_PcD_ToFill__(ar1)
    
        except (IndexError, AssertionError):
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
        
        self.trappedNW[arrr] = False
        self.connNW[arrr] = True
        self.trapCluster_NW[arrr] = 0
        self.trappedNW_Pc[arrr] = 0.0
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


    def func1(self, i):
        return self.fluid[self.elem[i].neighbours].sum()>0
        
    
    def initializeToFill(self):
        condlist1 = np.zeros(self.totElements-2, dtype='bool')
        condlist1[(self.fluid[1:-1]==0)] = np.array(
            [*map(lambda i: self.func1(i), self.elementLists[(self.fluid[1:-1]==0)])])
        ElemToFill = self.elementLists[condlist1]

        self.__update_PcD_ToFill__(ElemToFill)


    def populateToFill(self, arr):
        condlist = np.zeros(self.totElements, dtype='bool')
        condlist[arr[self.fluid[arr]==0]] = True
        nPores = self.nPores

        Notdone = np.ones(self.totElements, dtype='bool')
        Notdone[condlist] = False
        Notdone[[-1,0]] = False

        arr = arr[self.fluid[arr]==1]
        arrP = arr[arr<=nPores]   #pores
        arr = list(arr[arr>nPores])  #throats
        try:
            assert arrP.size>0
            Notdone[arrP] = False
            self.prevFilled[arrP] = False
            arrT = self.PTConnections[arrP]
            condlist[arrT[(self.fluid[arrT]==0)&(arrT>0)]] = True
            arr.extend(arrT[(self.fluid[arrT]==1)&(arrT>0)])
        except AssertionError:
            pass

        while True:
            try:
                tt = arr.pop(0)
                Notdone[tt] = False
                self.prevFilled[tt] = False
                while True:
                    try:
                        tt = tt-nPores
                        arrr = np.zeros(nPores+2, dtype='bool')
                        arrr[self.TPConnections[tt]] = True
                        pp = self.poreList[(arrr&Notdone[self.poreListS])[1:-1]]
                        Notdone[pp] = False
                        self.prevFilled[pp] = False
                        condlist[pp[self.fluid[pp]==0]] = True
                        pp = pp[self.fluid[pp]==1]

                        arrr = np.zeros(self.totElements, dtype='bool')
                        arrr[self.PTConnections[pp]] = True
                        tt = self.elementLists[(arrr&Notdone)[1:-1]]
                        Notdone[tt] = False
                        self.prevFilled[tt] = False
                        condlist[tt[self.fluid[tt]==0]] = True
                        tt = tt[self.fluid[tt]==1]
                        
                        assert tt.size > 0

                    except (AssertionError, IndexError):
                        try:
                            arr = np.array(arr)
                            arr = list(arr[Notdone[arr]])
                        except IndexError:
                            arr=[]
                        break
            except IndexError:
                break
        ElemToFill = self.elementLists[condlist[1:-1]]
        self.__update_PcD_ToFill__(ElemToFill)
    

    def __writeHeadersD__(self):
        self.resultD_str="======================================================================\n"
        self.resultD_str+="# Fluid properties:\nsigma (mN/m)  \tmu_w (cP)  \tmu_nw (cP)\n"
        self.resultD_str+="# \t%.6g\t\t%.6g\t\t%.6g" % (
            self.sigma*1000, self.muw*1000, self.munw*1000, )
        self.resultD_str+="\n# calcBox: \t %.6g \t %.6g" % (
            self.calcBox[0], self.calcBox[1], )
        self.resultD_str+="\n# Wettability:"
        self.resultD_str+="\n# model \tmintheta \tmaxtheta \tdelta \teta \tdistmodel"
        self.resultD_str+="\n# %.6g\t\t%.6g\t\t%.6g\t\t%.6g\t\t%.6g" % (
            self.wettClass, round(self.minthetai*180/np.pi,3), round(self.maxthetai*180/np.pi,3), self.delta, self.eta,) 
        self.resultD_str+=self.distModel
        self.resultD_str+="\nmintheta \tmaxtheta \tmean  \tstd"
        self.resultD_str+="\n# %3.6g\t\t%3.6g\t\t%3.6g\t\t%3.6g" % (
            round(self.contactAng.min()*180/np.pi,3), round(self.contactAng.max()*180/np.pi,3), 
            round(self.contactAng.mean()*180/np.pi,3), round(self.contactAng.std()*180/np.pi,3))
        
        self.resultD_str+="\nPorosity:  %3.6g" % (self.porosity)
        self.resultD_str+="\nMaximum pore connection:  %3.6g" % (self.maxPoreCon)
        self.resultD_str+="\nAverage pore-to-pore distance:  %3.6g" % (self.avgP2Pdist)
        self.resultD_str+="\nMean pore radius:  %3.6g" % (self.Rarray[self.poreList].mean())
        self.resultD_str+="\nAbsolute permeability:  %3.6g" % (self.absPerm)
        
        self.resultD_str+="\n======================================================================"
        self.resultD_str+="\n# Sw\t qW(m3/s)\t krw\t qNW(m3/s)\t krnw\t Pc\t Invasions"

        self.totNumFill = 0
        self.resultD_str = self.do.writeResult(self.resultD_str, self.capPresMin)


    def __writeTrappedData__(self):
        filename = os.path.join(
            "./results_csv/Flowmodel_{}_Drainage_cycle{}_{}_{}_trappedDist.csv".format(
                self.title, self.cycle, self.label, self._num))
        data = [*zip(self.Rarray, self.volarray, self.fluid, self.trappedW, self.trappedNW)]
        np.savetxt(filename, data, delimiter=',', header='rad, volume, fluid, trappedW, trappedNW')

        
            
class PImbibition(TwoPhaseImbibition):
    def __new__(cls, obj, writeData=False, includeTrapping=True, writeTrappedData=True):
        obj.__class__ = PImbibition
        return obj
    
    def __init__(self, obj, writeData=False, includeTrapping=True, writeTrappedData=True):
        super().__init__(obj, writeData=writeData)
        self.writeData = writeData
        self.includeTrapping = includeTrapping
        if self.writeData: self.__fileName__()
        self.trapdata = (self.trappedW, self.trappedW_Pc, self.trapCluster_W)
        self.writeTrappedData = writeTrappedData
        print(self.contactAng)
        print(self.contactAng.sum())
    
    def popUpdateWaterInj(self):
        #from IPython import embed; embed()
        k = self.ElemToFill.pop(0)
        capPres = self.PcI[k]
        self.capPresMin = np.min([self.capPresMin, capPres])

        try:
            #assert not self.do.isTrapped(k, 1, self.capPresMin)
            self.trapped[k] = False
            self.trappedPc[k] = 0.0
            self.trapClust[k] = 0

            self.fluid[k] = 0
            self.fillmech[k] = 1*(self.PistonPcAdv[k]==capPres)+2*(
                self.porebodyPc[k]==capPres)+3*(self.snapoffPc[k]==capPres)
            self.cnt += 1
            self.invInsideBox += self.isinsideBox[k]

            arr = self.elem[k].neighbours
            arr = arr[(self.fluid[arr] == 1)&(arr>0)]
            [*map(lambda i: self.do.isTrapped(i, 1, self.capPresMin), arr)]
            self.__computePc__(self.capPresMin, arr)

            self.do.isTrapped(k, 0, self.capPresMin, self.trapdata)
    
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
                        self.fluid[self.ElemToFill[0]] = 0
                        assert self.isNWConnected()
                        self.popUpdateWaterInj()
                    except AssertionError:
                        self.fluid[self.ElemToFill[0]] = 1
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
            self.satW = self.do.Saturation(self.AreaWPhase, self.AreaSPhase)
            self.totNumFill += self.cnt

            try:
                assert self.PcI[self.ElemToFill[0]] >= self.PcTarget
            except (AssertionError, IndexError):
                break

            if not self.filling:
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
        self.satW = self.do.Saturation(self.AreaWPhase, self.AreaSPhase)
        self.do.computePerm()
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

    
    def __writeHeadersI__(self):
        self.resultI_str="======================================================================\n"
        self.resultI_str+="# Fluid properties:\nsigma (mN/m)  \tmu_w (cP)  \tmu_nw (cP)\n"
        self.resultI_str+="# \t%.6g\t\t%.6g\t\t%.6g" % (
            self.sigma*1000, self.muw*1000, self.munw*1000, )
        self.resultI_str+="\n# calcBox: \t %.6g \t %.6g" % (
            self.calcBox[0], self.calcBox[1], )
        self.resultI_str+="\n# Wettability:"
        self.resultI_str+="\n# model \tmintheta \tmaxtheta \tdelta \teta \tdistmodel"
        self.resultI_str+="\n# %.6g\t\t%.6g\t\t%.6g\t\t%.6g\t\t%.6g" % (
            self.wettClass, round(self.minthetai*180/np.pi,3), round(self.maxthetai*180/np.pi,3), self.delta, self.eta,) 
        self.resultI_str+=self.distModel
        self.resultI_str+="\nmintheta \tmaxtheta \tmean  \tstd"
        self.resultI_str+="\n# %3.6g\t\t%3.6g\t\t%3.6g\t\t%3.6g" % (
            round(self.contactAng.min()*180/np.pi,3), round(self.contactAng.max()*180/np.pi,3), 
            round(self.contactAng.mean()*180/np.pi,3), round(self.contactAng.std()*180/np.pi,3))
        
        self.resultI_str+="\nPorosity:  %3.6g" % (self.porosity)
        self.resultI_str+="\nMaximum pore connection:  %3.6g" % (self.maxPoreCon)
        self.resultI_str+="\nAverage pore-to-pore distance:  %3.6g" % (self.avgP2Pdist)
        self.resultI_str+="\nMean pore radius:  %3.6g" % (self.Rarray[self.poreList].mean())
        self.resultI_str+="\nAbsolute permeability:  %3.6g" % (self.absPerm)
        
        self.resultI_str+="\n======================================================================"
        self.resultI_str+="\n# Sw\t qW(m3/s)\t krw\t qNW(m3/s)\t krnw\t Pc\t Invasions"

        self.totNumFill = 0
        self.resultI_str = self.do.writeResult(self.resultI_str, self.capPresMax)


    def __fileName__(self):
        result_dir = "./results_csv/"
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        self.label = 'wt' if self.includeTrapping else 'nt'
        if not hasattr(self, '_num'):
            self._num = 1
            while True:         
                file_name = os.path.join(
                    result_dir, "Flowmodel_"+self.title+"_Imbibition_cycle"+str(self.cycle)+\
                        "_"+self.label+"_"+str(self._num)+".csv")
                if os.path.isfile(file_name): self._num += 1
                else:
                    break
            self.file_name = file_name
        else:
            self.file_name = os.path.join(
                result_dir, "Flowmodel_"+self.title+"_Imbibition_cycle"+str(self.cycle)+\
                    "_"+self.label+"_"+str(self._num)+".csv") 
            

    def __writeTrappedData__(self):
        filename = os.path.join(
            "./results_csv/Flowmodel_{}_Imbibition_cycle{}_{}_{}_trappedDist.csv".format(
                self.title, self.cycle, self.label, self._num))
        data = [*zip(self.Rarray, self.volarray, self.fluid, self.trappedW, self.trappedNW)]
        np.savetxt(filename, data, delimiter=',', header='rad, volume, fluid, trappedW, trappedNW')



class SecImbibition(PImbibition):
    def __new__(cls, obj, writeData=False, includeTrapping=True, writeTrappedData=True):
        obj.__class__ = SecImbibition
        return obj
    
    def __init__(self, obj, writeData=False, includeTrapping=True, writeTrappedData=True):    
        self.fluid[[-1, 0]] = 0, 1  
        self.ElemToFill = SortedList(key=lambda i: self.LookupList(i))
        self.capPresMin = self.maxPc
        
        self.contactAng, self.thetaRecAng, self.thetaAdvAng = self.prop_imbibition.values()
        self.is_oil_inj = False
        self.trapped = self.trappedNW
        self.trappedPc = self.trappedNW_Pc
        self.trapClust = self.trapCluster_NW
        self.do.__initCornerApex__()
        self.__computePistonPc__()
        self.__computePc__(self.maxPc, self.elementLists, False)

        self._areaWP = self._cornArea.copy()
        self._areaNWP = self._centerArea.copy()
        self._condWP = self._cornCond.copy()
        self._condNWP = self._centerCond.copy()

        self.writeData = writeData
        if self.writeData: self.__fileName__()
        self.trapdata = (self.trappedW, self.trappedW_Pc, self.trapCluster_W)
        self.writeTrappedData = writeTrappedData



