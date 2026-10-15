import os
import numpy as np
from sortedcontainers import SortedList
from time import time

from pnflowPy.tPhaseImb import TwoPhaseImbibition
import pnflowPy.tPhaseImb as tPhaseImb
import pnflowPy.utilities as do


class PImbibition(TwoPhaseImbibition):
    def __init__(self, obj, writeData=False, includeTrapping=False):
        super().__init__(obj, writeData=writeData, includeTrapping=False)
        tPhaseImb.popUpdateWaterInj = popUpdateWaterInj
        obj.writeData = writeData
        obj.results_dir = "percolation_without_trapping_results/"
        obj.results_str = ""


class SecImbibition(PImbibition):    
    def __init__(self, obj, writeData=False):
        tPhaseImb.popUpdateWaterInj = popUpdateWaterInj  
        obj.fluid[[-1, 0]] = 0, 1  
        obj.ElemToFill = SortedList(key=lambda i: tPhaseImb.LookupList(i, obj.PcI, obj.nPores))
        obj.capPresMin = obj.maxPc
        
        obj.contactAng, obj.thetaRecAng, obj.thetaAdvAng = obj.prop_imbibition.values()
        obj.is_oil_inj = False

        do.__initCornerApex__(obj)
        tPhaseImb.__computePistonPc__(obj)
        obj.NWElemNotInToFill = obj.cNWP.hasFluid.copy()
        tPhaseImb.__computePc__(obj, obj.maxPc, obj.elementLists, update=False, trapping=False)

        obj._areaWP = obj._cornArea.copy()
        obj._areaNWP = obj._centerArea.copy()
        obj._condWP = obj._cornCond.copy()
        obj._condNWP = obj._centerCond.copy()

        obj.areaWPhase = obj._areaWP.view()
        obj.areaNWPhase = obj._areaNWP.view()
        obj.gWPhase = obj._condWP.view()
        obj.gNWPhase = obj._condNWP.view()

        obj.results_dir = "percolation_without_trapping_results/"
        obj.results_str = ""
        obj.writeData = writeData
        if obj.writeData: do.__fileName__(obj)        
        

def popUpdateWaterInj(self):
    self.pop += 1
    k = self.ElemToFill.pop(0)
    capPres = self.PcI[k]
    self.capPresMin = np.min([self.capPresMin, capPres])

    self.cWP.fill_with_phase(k, self.capPresMin, self)
    self.cNWP.unfill_phase(k, self.capPresMin)
    neigh = self.connectivity_graph[k]
    neigh = neigh[self.cNWP.hasFluid[neigh]]
    if neigh.size>0:
        tPhaseImb.__computePc__(self, self.capPresMin, neigh, trapping=False)
    
    self.specialPcD[k] = self.capPresMin
    self.fillmech[k] = 1*(self.PistonPcAdv[k]==capPres)+2*(
        self.porebodyPc[k]==capPres)+3*(self.snapoffPc[k]==capPres)
    self.cnt += 1
    self.invInsideBox += self.isinsideBox[k]

    