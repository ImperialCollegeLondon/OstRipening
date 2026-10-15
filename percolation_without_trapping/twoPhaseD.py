import os
import numpy as np
from sortedcontainers import SortedList
from time import time

import pnflowPy.tPhaseD as tPhaseD
from pnflowPy.tPhaseD import TwoPhaseDrainage
import pnflowPy.utilities as do
import pnflowPy.SecondaryProcesses as secProcess



class PDrainage(TwoPhaseDrainage):
    def __init__(self, obj, writeData=False):
        super().__init__(obj, writeData=writeData)
        tPhaseD.popUpdateOilInj = popUpdateOilInj
        obj.results_dir = "percolation_without_trapping_results/"
        obj.results_str = ""
        if obj.writeData: do.__fileName__(obj)


class SecDrainage(PDrainage):  
    def __init__(self, obj, writeData=False):

        tPhaseD.popUpdateOilInj = popUpdateOilInj
        obj.fluid[[-1, 0]] = 1, 0        
        obj.capPresMax = obj.capPresMin
        obj.is_oil_inj = True
        obj.contactAng, obj.thetaRecAng, obj.thetaAdvAng = obj.prop_drainage.values()
        
        do.__initCornerApex__(obj)
        obj.Fd_Tr = do.__computeFd__(obj, obj.elemTriangle, obj.halfAnglesTr)
        obj.Fd_Sq = do.__computeFd__(obj, obj.elemSquare, obj.halfAnglesSq)
        tPhaseD.computePistonPc(obj)
        obj.centreEPOilInj[obj.elementLists] = 2*obj.sigma*np.cos(
           obj.thetaRecAng[obj.elementLists])/obj.Rarray[obj.elementLists]
        obj.PcD[:] = obj.PistonPcRec
        hasFluid1 = (obj.fluid==1)
        obj.PistonPcRec[hasFluid1] = obj.centreEPOilInj[hasFluid1]
        obj.ElemToFill = SortedList(key=lambda i: tPhaseD.LookupList(obj, i))
        obj.NinElemList[:] = True
        obj.prevFilled = hasFluid1
        initializeToFill(obj)  
        
        obj._cornArea = obj._areaWP.copy()
        obj._centerArea = obj._areaNWP.copy()
        obj._cornCond = obj._condWP.copy()
        obj._centerCond = obj._condNWP.copy()

        obj.areaWPhase = obj._cornArea.view()
        obj.areaNWPhase = obj._centerArea.view()
        obj.gWPhase = obj._cornCond.view()
        obj.gNWPhase = obj._centerCond.view()  

        obj.cycle += 1
        obj.writeData = writeData
        obj.results_dir = "percolation_without_trapping_results/"
        obj.results_str = ""
        if obj.writeData: do.__fileName__(obj)
        obj.primary = False
        obj.totNumFill = 0


def popUpdateOilInj(self):
    self.pop += 1
    k = self.ElemToFill.pop(0)
    self.NinElemList[k] = True
    capPres = self.PcD[k]
    self.capPresMax = max(self.capPresMax, capPres)
    
    if not self.cWP.trapped[k]:
        self.cNWP.fill_with_phase(k, self.capPresMax, self)
        arr = self.connectivity_graph[k]
        cID = np.unique(self.cNWP.clusterID[arr[self.cNWP.hasFluid[arr]]])
        self.PistonPcRec[k] = self.centreEPOilInj[k]
        if self.isCircle[k]:
            self.cWP.unfill_phase(k, self.capPresMax)
        
        self.cnt += 1
        self.invInsideBox += self.isinsideBox[k]
        if cID.size==1:
            arr = arr[(self.fluid[arr]==0) & ~(self.cWP.trapped[arr])]
            tPhaseD.update_PcD_ToFill(self, arr)
        else:
            kk = self.cNWP.clusterID[k]
            secProcess.populateToFill(self, self.cNWP[kk].members)
    


def initializeToFill(self):
    arr = np.union1d(self.TPValid[self.cNWP.hasFluid[self.TValid]], 
                    self.TValid[self.cNWP.hasFluid[self.TPValid]])
    arr = np.union1d(self.conTToIn, arr)
    arr0 = arr[self.fluid[arr]==0]
    tPhaseD.update_PcD_ToFill(self, arr0)


