import numpy as np
from time import time
import math
import os
import dill

import pnflowPy.tPhaseD as tPhaseD
import pnflowPy.tPhaseImb as tPhaseImb
from OstRipening.network_numba import NetworkJit, ClusterJit
import OstRipening.temp as temp
from OstRipening.utilities_numba import *
from numba import prange, njit
import OstRipening.cluster_cpp as cluster_cpp


#MEMORY_DIR = f"../ostwald_ripening_results/impP_10000_D_1dot8minus9_H_6dot9minus6_untrappedConstraned_PcDupdated"
MEMORY_DIR = f"ostwald_ripening_results/03012026_maxPc1e5/impP_1e6_D_4dot89minus9_H_7dot8minus6_clustPcalpha0dot56_dt1dot5_maxDP2000"
MEMORY_DIR = f"ostwald_ripening_results/201225_maxPc1e5/impP_1e6_D_4dot89minus9_H_7dot8minus6_clustPcalpha0dot58"
os.makedirs(MEMORY_DIR, exist_ok=True)
_temp = temp.TempArrays()
alpha = 0.58

class TimeDependency:
    def __init__(self, network, Pc, T=298, imposedP=1e6, H=7.8e-6, D=7.3e-9, 
                 steps=10, dt=0.0005, adjustTime=False, moles_tol=1e-23, pc_tol=1e-4, max_iter=50):
        
        print('----------------------------------------------------------------------------------')
        print('-----------------------------Time Dependent Ostwald Ripening ---------------------')
 
        # initialising the parameters
        network.steps = steps
        network.H = H  #mol m-3 pa-1
        #network.D = D*network.porosity**2.3  #m2/s
        network.D = D #1.2e-9  #m2/s
        network.num = 1
        network.R = 8.314  #J/mol.K
        network.T = T    #K
        network.imposedP = imposedP    #1MPa
        network.Pc = Pc
        network._dt = dt
        network.adjustTime = adjustTime
        network.moles_tol = moles_tol
        network.pc_tol = pc_tol
        network.max_iter = max_iter
        network.RT = 8.314*T
        network.half_pi = np.pi/2.0
        

def LenEq(self):
    ''' returns the diffusion characteristic length for each P1-T-P2 pair '''
    len_diff = 0.5*(self.LP1array+self.LTarray+self.LP2array)
    #len_diff = 0.5*self.LTarray
    areaArray = np.array([
        self.areaSPhase[self.P1array], self.areaSPhase[self.tList], self.areaSPhase[self.P2array]])
    area_min = np.nanmin(np.where(areaArray==0.0, np.nan, areaArray), axis=0) 
    lenEq = area_min/len_diff
    return np.column_stack([lenEq, lenEq])

def initialize(self):
    # setting up the arrays for elements ...
    st=time()
    self.len_tij = LenEq(self)
    #from IPython import embed; embed()

    self.clusterNW.ostMode = True
    self.clusterNW.resizeClusters(0, True)
    self.clusterW.resizeClusters(0, True)
    #from IPython import embed; embed()

    # re-initialize the clusters   
    self.clusterW.__neighbours__(self)
    self.clusterNW.__neighbours__(self)

    # update the entry capillary pressure for drainage
    #updateEntryPresures(self)  
    
    # set up necessary arrays
    settingUpArrays(self, self.clusterNW, False)

    # initialize the flow rates of the phases
    #initializeFlowrate(self)



    # for writing data/results
    # self.resultsP_str = "# Step,totalFlux,total_delta_nMoles,Sw, \
    #     #cluster_growth,#cluster_shrinkage,AvgPressure,totMoles_W,totMoles_NW,totMoles"
    # self.__fileName__()
    print('Initializing timeDependency completed!!!')
    print('time spent:  ', time()-st)
    

def updateEntryPresures(self):
    self.PcD[:] = np.minimum(self.PistonPcRec, self.PcD)
    self.PcI[:] = np.maximum.reduce([self.PistonPcAdv, self.porebodyPc])


def settingUpArrays(self, clust, excludeCircles=False, pc_min=1.0e-30, pc_max=1.0e30, moles_tol=1e-18):
    ''' sets up useful arrays for the time-dependent simulation '''
    temp.TempArrays.formTempClustArrays(clust.nClusters)
    clust.valClust[:] = (clust.size>0) # returns valid clusters
    self.len_tij_valid = self.len_tij[self.TPCond[1:]] # returns len_tij of valid throat-pore pairs
    self.len_tij_valid[self.isCircle[self.TPValid]|self.isCircle[self.TValid]] = 0.0 # circular elem
    valkeys = np.flatnonzero(clust.valClust).astype(np.int32)

    #from IPython import embed; embed()
    #self.Fd_Tr = do.__computeFd__(self, self.elemTriangle, self.halfAnglesTr)
    #self.Fd_Sq = do.__computeFd__(self, self.elemSquare, self.halfAnglesSq)
    #do.__initCornerApex__(self)
    #tPhaseD.__computePistonPc__(self)
    #self.PcD[:] = self.PistonPcRec
    #self.centreEPOilInj =  2*self.sigma*np.cos(self.thetaRecAng[self.elementListS])/self.Rarray[self.elementListS]
    #self.PcD[self.fluid==1] =  self.centreEPOilInj[self.fluid==1]
    #self.PcD[:] = self.PistonPcRec*alpha + (1-alpha)*self.PcI

    
    
    # setting up arrays for cluster
    nClust = clust.nClusters
    nwID = np.flatnonzero(self.hasNWFluid)
    updateToDrainImbibe_numba(valkeys, clust.members, clust.members_offset, clust.size, clust.neighbours, 
        clust.toDrain, clust.toImbibe, clust.pcShrink, clust.pcMax, self.PcI, self.PcD, self.Rarray,
        self.totElements, _temp.mem, True, pc_min, pc_max)
    clust.valClustD[:] = clust.valClust & (clust.toDrain<=self.totElements)
    
    # updating upper and lower values of volumes and moles for NW clusters
    #clust.updateVolumeNWCluster(nwID, valkeys, self)
    self.satList = np.zeros(self.totElements, dtype=np.float64)
    self.satList[1:-1] = self.areaWPhase[1:-1]/self.areaSPhase[1:-1]
    clust.volume[valkeys] = np.bincount(
        clust.clusterID[self.hasNWFluid], (1-self.satList[self.hasNWFluid])*self.volarray[self.hasNWFluid])[valkeys]

    

    
    # adjust the cluster pc and updating moles of NW clusters
    cond = np.flatnonzero(clust.valClustD)
    clust.pc[cond] = alpha*clust.pcMax[cond]+(1-alpha)*clust.pc[cond]
    clust.moles[valkeys] = ((self.imposedP + clust.pc[valkeys])*clust.volume[valkeys]/self.RT)
    self.aqAvgPres = (clust.pc*clust.volume).sum()/clust.volume.sum() #- 100 #self.Pc
    #self.aqAvgPres = 0.0 #self.Pc
    self.oldaqAvgPres = self.aqAvgPres
    self.minConc = self.H*self.aqAvgPres
    #self.minConc = np.zeros(self.totElements)
    self.minMoles = self.minConc*self.volarray
    #self.minMoles = self.H*2456*self.volarray
    
    #clust.updateMolesNWCluster(valkeys, self)
    
    # setting up arrays for elements
    self.elemToUpdateW = (self.fluid==0)&(self.volarray>0.0)
    self.elemToUpdateW[[-1,0]] = False                           # boolean
    self._elemToUpdateW = np.flatnonzero(self.elemToUpdateW).astype(np.int32)  # index
    self.elemToUpdateNW = self.hasNWFluid.copy()
    self._elemToUpdateNW = np.flatnonzero(self.elemToUpdateNW)           # index
    self.elemToUpdate = (self.elemToUpdateW|self.elemToUpdateNW)
    self._elemToUpdate = np.flatnonzero(self.elemToUpdate).astype(np.int32)   # index
    self.validVolumes = (self.volarray > 0.0)
    self.validToShrink = clust.valClust & (clust.pcShrink>self.oldaqAvgPres)
    self.validToGrow = clust.valClustD & (clust.pcMax<self.oldaqAvgPres)
    self.clustersToShrink = np.zeros(nClust, dtype=np.bool_)
    self.clustersToGrow = np.zeros(nClust, dtype=np.bool_)

    # initializing and updating moles and concentration for elements
    self.gasConc = np.zeros((2, self.totElements), dtype=np.float64)
    self.aqueousMoles = np.zeros(self.totElements, dtype=np.float64)
    #from IPython import embed; embed()
    
    updateMolesConcElements(self, clust)
    self.flux = np.zeros(self.nValid, dtype=np.float64)
    self.netFlux = np.zeros(self.totElements, dtype=np.float64)
    self.netFluxClusters = np.zeros(nClust, dtype=np.float64)
    self.netMoles = np.zeros(self.totElements, dtype=np.float64)
    self.clustersToGrow = np.zeros(nClust, dtype=np.bool_)
    self.clustersToShrink = np.zeros(nClust, dtype=np.bool_)
    #self.elemValidToLoseFlux = (self.gasConc[:,0] > self.minConc) & (self.aqueousMoles - self.minMoles > moles_tol)
    

def adjustClusterPc(self, clust, clustID):
    ''' adjust the cluster pc '''

    nwID = np.flatnonzero(clust.hasFluid)
    clust.pc[clust.valClust] = 0.0
    np.maximum.at(clust.pc, clust.clusterID[nwID], self.PcD[nwID])


def updateMolesConcElements(self, clust):
    ''' update moles and concentration for elements '''

    self.gasConc[0, self._elemToUpdateNW] = self.H*clust.pc[clust.clusterID[self._elemToUpdateNW]]
    self.gasConc[0, self._elemToUpdateW] = self.H*self.aqAvgPres
    self.aqueousMoles[self._elemToUpdate] = (
        self.gasConc[0, self._elemToUpdate]*self.volarray[self._elemToUpdate])


def updateLists(self, arr, fluid=0):
    cond = True if fluid==0 else False
    arrW = arr[self.volarray[arr]>0.0]
    self.elemToUpdateW[arrW] = cond
    self.elemToUpdateNW[arr] = not cond
    self._elemToUpdateW = self.elementListS[self.elemToUpdateW]
    self._elemToUpdateNW = self.elementListS[self.elemToUpdateNW]
    self.elemToUpdate = (self.elemToUpdateW|self.elemToUpdateNW)
    self._elemToUpdate = self.elementListS[self.elemToUpdate]
    self.clusterNW.valClust[:] = (self.clusterNW.size>0)


@njit(fastmath=False, parallel=True)
def fast_flux_moles_pc_conc_kernel(flux, netFlux, len_tij_valid, gasConc0, gasConc1, volarray,
    TPValid, TValid, clusterID, clust_moles, clust_volume, clust_pc, clust_members, clust_size, 
    members_offset, elemToUpdateW, elemToUpdateNW, aqMoles, minMoles, imposedP, D, H, RT, _dt, moles_tol):
    
    fSize = flux.size
    for i in range(fSize):
        p, t = TPValid[i], TValid[i]
        #f = D * len_tij_valid[i] * (gasConc0[p] - gasConc0[t])
        f = flux[i]
        if f > 0:
            if (aqMoles[p] - minMoles[p]) < moles_tol:
                f = 0.0
        elif f < 0:
            if (aqMoles[t] - minMoles[t]) < moles_tol:
                f = 0.0

        # if abs(f) < 1e-25:
        #     f = 0.0
        flux[i] = f

    for i in range(fSize): 
        f = flux[i]
        p, t = TPValid[i], TValid[i]
        netFlux[t] += f
        netFlux[p] -= f

    nfSize = netFlux.size
    dt = _dt
    for e in range(nfSize):
        if netFlux[e] < 0.0:
            tratio =  (minMoles[e] - aqMoles[e])/netFlux[e]
            if tratio < 0.0:
                return 0.0
            if dt > tratio:
                dt = tratio

    for e in range(nfSize):
        nMoles = netFlux[e] * dt
        if nMoles < -aqMoles[e]:
            nMoles = -aqMoles[e]

        if elemToUpdateNW[e]:
            k = clusterID[e]
            clust_moles[k] += nMoles
        elif elemToUpdateW[e]:
            aqMoles[e] += nMoles
            gasConc1[e] = aqMoles[e]/volarray[e]

    nClusters = clust_moles.size
    for k in prange(nClusters):
        if clust_size[k] == 0: continue
        if clust_moles[k] < 0.0:
            clust_moles[k] = 0.0

        clust_pc[k] = clust_moles[k]*RT/clust_volume[k] - imposedP
        if clust_pc[k] < 1e-3:
            clust_pc[k] = 1e-3
        st = members_offset[k]
        mem = clust_members[st:st+clust_size[k]]
        gasConc1[mem] = H*clust_pc[k]

    return dt    


@njit(parallel=True)
def simulate_kernel(flux, netFlux, len_tij_valid, gasConc0, gasConc1, 
    volarray, TPValid, TValid, clusterID, clust_moles, clust_volume, clust_pc, clust_members, 
    clust_size, clust_members_offset, elemToUpdateW, elemToUpdateNW, aqueousMoles, minMoles, 
    validToShrink, validToGrow, clustersToShrink, clustersToGrow, pcShrink, pcMax,
    imbEvents, drainEvents, imposedP, D, H, RT, _dt, totTime, moles_tol):

    nClusters = clust_pc.size

    for k in prange(nClusters):
        if validToShrink[k] and (clust_pc[k] < pcShrink[k]):
            clustersToShrink[k] = True
        elif validToGrow[k] and (clust_pc[k] > pcMax[k]):
            clustersToGrow[k] = True
    
    triggerShrinkage = clustersToShrink.any()
    triggerGrowth = clustersToGrow.any()
    if not triggerShrinkage and not triggerGrowth:
        netFlux[:] = 0.0
        dt = fast_flux_moles_pc_conc_kernel(flux, netFlux, len_tij_valid, gasConc0, gasConc1, 
            volarray, TPValid, TValid, clusterID, clust_moles, clust_volume, clust_pc, clust_members, 
            clust_size, clust_members_offset, elemToUpdateW, elemToUpdateNW, aqueousMoles, minMoles, 
            imposedP, D, H, RT, _dt, moles_tol)
        
        totTime += dt

    # if triggerShrinkage:
    #     imbEvents += clustersToShrink.sum()
    #     shrinkCluster(clustersToShrink)
    
    # if triggerGrowth:
    #     drainEvents += condGrowth.sum()
    #     growCluster(clustersToGrow)

    return totTime, dt, bool(triggerShrinkage), bool(triggerGrowth)


@njit
def checkEvents(validToShrink, validToGrow, clustSize, clustPc, clustPcShrink, clustPcMax, 
    clustersToShrink, clustersToGrow, initialAqAvgPres, nClusters):
    for k in prange(nClusters):
        clustersToShrink[k] = False
        clustersToGrow[k] = False
        if validToShrink[k]:
            if clustSize[k]==1:
                if clustPc[k]<=initialAqAvgPres:
                    clustersToShrink[k] = True
            elif clustSize[k]>1:
                if clustPc[k] < clustPcShrink[k]:
                    clustersToShrink[k] = True
        elif validToGrow[k]:
            if clustPc[k]>clustPcMax[k]:
                clustersToGrow[k] = True

    triggerShrinkage = self.clustersToShrink.any()
    triggerGrowth = self.clustersToGrow.any()

    return triggerShrinkage, triggerGrowth




def simulate(self, clust, ii, startTime, totTime, totalTime, minTime, moles_tol, pc_tol, max_iter):
    ''' main simulation loop for time-dependent Ostwald ripening '''

    triggerShrinkage, triggerGrowth = cluster_cpp.checkEvents(self.validToShrink, self.validToGrow, 
        clust.size, clust.pc, clust.pcShrink, clust.pcMax, self.clustersToShrink, self.clustersToGrow, 
        clust.moles, clust.toImbibe, self.volarray, self._factor, clust.nClusters)
    
    if triggerShrinkage or triggerGrowth or (totTime>minTime):
        self.gasConc[1, self._elemToUpdate] = self.gasConc[0, self._elemToUpdate]
        if triggerShrinkage:
            condShrink = self.clustersToShrink
            clust.imbEvents += condShrink.sum()
            keys = np.flatnonzero(condShrink).astype(np.int32)
            toImbibe = clust.toImbibe[keys]
            print('@@:   ', keys, toImbibe) 
            # if 829 in keys and 17308 in toImbibe:
            #     from IPython import embed; embed()
            clust.shrinkCluster(keys, self)
            updateLists(self, toImbibe)
            self.updateClust = False

        if triggerGrowth:
            condGrowth = self.clustersToGrow
            clust.drainEvents += condGrowth.sum()
            keys = np.flatnonzero(condGrowth).astype(np.int32)
            toDrain = clust.toDrain[keys]
            print('$$:  ', keys, toDrain)
            toDrain = clust.growCluster(keys, self)
            updateLists(self, toDrain, 1)
            self.updateClust = False

        totalMoles = self.aqueousMoles[self._elemToUpdateW].sum()+clust.moles.sum()
        gasVol =  ((1-self.satList)*self.volarray).sum()
        self.satW = ((self.satList[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()/
                            self.totVoidVolume)
        totalTime += totTime 
        totTime = 0.0
        
        self.aqAvgPres = (
            self.aqueousMoles[self.elemToUpdateW].sum()/
            (self.H*self.volarray[self.elemToUpdateW].sum()))
        gasAvgPres = (
            (clust.pc[clust.valClust]*clust.volume[clust.valClust]).sum()/
            clust.volume[clust.valClust].sum())
        
        print(("#:%8.6g  \tSimulation Time:%12.6g  \tActual runtime:%6.6g\
            \tTotal cluster moles:%8.6e  \tTotal dissolved moles:%6.6e\
            \tTotal gas moles:%4.10e \tAvg Pressure:%8.6e \tNo Shrinkage:%8.6g\
            \tNo Growth:%8.6g \tSat:%6.6g \tAqAvgPres:%6.6g" %(
            ii, round(totalTime,3), round(time()-startTime,3), clust.moles.sum(), 
            #self.dissolvedMoles.sum(), totalMoles,
            self.aqueousMoles.sum(), totalMoles,
            #clust.pc[self.valClust].mean(),
            gasAvgPres,
            clust.imbEvents, clust.drainEvents, self.satW, self.aqAvgPres)))
            
        writeData(self, round(totalTime,3))
        ii += 1
                
    else:
        self.netFlux[:] = 0.0
        self.dt = cluster_cpp.compute_fluxes_moles_pc_conc(self.D, self.H, self.RT, self.flux, self.netFlux, 
            self.len_tij_valid, self.gasConc[0], self.gasConc[1], self.volarray, self.TPValid, self.TValid, 
            clust.clusterID, clust.moles, clust.volume, clust.pc, clust.members, clust.size, clust.members_offset, 
            self.elemToUpdateW, self.elemToUpdateNW, self.aqueousMoles, self.minMoles, self.imposedP, self.moles_tol, 
            self._dt)
        # print(self.dt)
        # input('waitt and check')
        totTime += self.dt
                
    self.gasConc[0, self.elemToUpdate] = self.gasConc[1, self.elemToUpdate]
        
    return ii, totTime, totalTime


def simulateOstRip(self, implicit=True, freshStart=True):
    print('Im in simulateOstRip  ', freshStart)
    duration = 3600*24  # 24-hours
    #duration = 10
    st = time()
    
    if not os.path.isfile('volarray_bent.dat'):
        np.savetxt('volarray_bent.dat', self.volarray)
    if not os.path.isfile('coordinates_bent.dat'):
        np.savetxt('coordinates_bent.dat', 
                    np.column_stack((self.x_array, self.y_array, self.z_array)))
        
    # records data in at most 10-mins intervals if no event occur
    #toShrink_elements = np.loadtxt('bent_toShrink_elements.dat', delimiter=',', dtype=np.int32)
    if hasattr(self, 'imposedP'):
        print(f"\nalpha={alpha}       imposedP={self.imposedP}        aqAvgPres={self.oldaqAvgPres}\n")
    
    self.timesWithNoEvent = 0
    minTime = 3600
    print('$$$$$$$$$$$$$$$')
    if freshStart:
        self.j , self.ii, self.totTime, self.totalTime = 1, 1, 0.0, 0.0
        self.updateClust = False
        clust = self.clusterNW
        clust.drainEvents, clust.imbEvents = 0, 0
        totalMoles = (
            (self.volarray*self.gasConc[0])[~self.elemToUpdateNW].sum()+
            clust.moles.sum())
        gasAvgPres = (
                (clust.pc[clust.valClust]*clust.volume[clust.valClust]).sum()/
                clust.volume[clust.valClust].sum())
        print(("#:%8.6g  \tSimulation Time:%12.6g  \tActual runtime:%6.6g\
                \tTotal cluster moles:%8.6e  \tTotal dissolved moles:%6.6e\
                \tTotal gas moles:%6.10e \tAvg Pressure:%8.6e \tNo Shrinkage:%8.6g\
                \tNo Growth:%8.6g \tSat:%6.6g \tAqAvgPres:%6.6g" % (
            self.ii, round(self.totalTime,3), round(time()-st,3), clust.moles.sum(), 
            self.aqueousMoles.sum(), totalMoles, gasAvgPres,
            clust.imbEvents, clust.drainEvents, self.satW, self.aqAvgPres)))
        #from IPython import embed; embed()
        writeData(self, round(self.totalTime,3))
        
    else:
        from IPython import embed; embed()
        with open(os.path.join(MEMORY_DIR, f"netsim_time_Dependent_72_hours.pkl"), "rb") as f:
            loaded_obj = dill.load(f)
            #self = dill.load(f)
        
        do.updateObj(self, loaded_obj)
        from IPython import embed; embed()

        #totTime, totalTime = self.j, self.ii, self.totTime, self.totalTime
        #j, ii, totTime, totalTime = self.j, self.ii, self.totTime, self.totalTime
        print(self.j, self.ii, self.totTime, self.totalTime)
        clust = self.clusterNW
        temp.TempArrays.formTempClustArrays(clust.nClusters)

    # size0 = clust.size[0]
    # from pnflowPy.cluster import doClustering_numba
    
    #self.oldaqAvgPres = self.aqAvgPres
    
    timeToSave = minTime*self.j
    self._factor = self.H*self.totElements*2000         # no cluster disappearance should raise avg aq pressure > 2000 Pa
    from IPython import embed; embed()
    
    while self.totalTime < duration:
        #while totalTime < 25.33:
        try:
            self.ii, self.totTime, self.totalTime = simulate(
                self, clust, self.ii, st, self.totTime, self.totalTime, minTime, self.moles_tol, 
                self.pc_tol, self.max_iter)
            if self.totalTime > timeToSave:
                try:
                    #self.j, self.ii = j+1, ii
                    #self.j += 1
                    #self.totTime, self.totalTime = totTime, totalTime
                    # with open(os.path.join(MEMORY_DIR, f"netsim_time_Dependent_{self.j}_hours.pkl"),
                    #         "wb") as f:
                    #     dill.dump(self, f)

                    filename = os.path.join(MEMORY_DIR, f"netsim_time_Dependent_{self.j}_hours.pkl")
                    saveState(self, filename)
                    
                    self.j += 1
                    timeToSave = minTime*self.j
                except:
                    print('there is an error in pickling!!!')
                    from IPython import embed; embed()
                
        except:
            print('there is an error in the simulation, check line 1438!!!')
            from IPython import embed; embed()
            
    print('::::::::::::::::::::::::::::')
    from IPython import embed; embed()


def writeData(self, totalTime):
    with open(os.path.join(MEMORY_DIR, 'clustPcOstRipening_bent.dat'), 'a') as f1,\
            open(os.path.join(MEMORY_DIR, 'clustVolOstRipening_bent.dat'), 'a') as f2,\
            open(os.path.join(MEMORY_DIR, 'clustMolesOstRipening_bent.dat'), 'a') as f3,\
            open(os.path.join(MEMORY_DIR, 'clustIDOstRipening_bent.dat'), 'a') as f4,\
            open(os.path.join(MEMORY_DIR, 'dissolvedMolesOstRipening_bent.dat'), 'a') as f5,\
            open(os.path.join(MEMORY_DIR, 'timeArrayOstRipening_bent.dat'), 'a') as f6,\
            open(os.path.join(MEMORY_DIR, 'numberEventsOstRipening_bent.dat'), 'a') as f7,\
            open(os.path.join(MEMORY_DIR, 'saturationOstRipening_bent.dat'), 'a') as f8,\
            open(os.path.join(MEMORY_DIR, 'clustSizeOstRipening_bent.dat'), 'a') as f9,\
            open(os.path.join(MEMORY_DIR, 'saturation_by_element_OstRipening_bent.dat'), 'a') as f10:
       
        np.savetxt(f1, [self.clusterNW.pc], delimiter=',', fmt='%g')
        np.savetxt(f2, [self.clusterNW.volume], delimiter=',', fmt='%g')
        np.savetxt(f3, [self.clusterNW.moles], delimiter=',', fmt='%g')
        np.savetxt(f4, [self.clusterNW_ID], delimiter=',', fmt='%g')
        np.savetxt(f5, [self.aqueousMoles], delimiter=',', fmt='%g')
        f6.write(str(totalTime)+',')
        np.savetxt(f7, [[self.clusterNW.imbEvents, self.clusterNW.drainEvents]], 
                    delimiter=',', fmt='%g')
        f8.write(str(self.satW)+',')
        np.savetxt(f9, [self.clusterNW.size], delimiter=',', fmt='%g')
        np.savetxt(f10, [self.satList], delimiter=',', fmt='%g')


def saveState(self, fname):
    state_attrs = ['satW', 'rng', 'fluid',
       'hasWFluid', 'hasNWFluid', 'clusterW', 'clusterNW', 'trappedNW', 'trappedW', 
       'clusterW_ID', 'clusterNW_ID',  '_dt',  'satList', 'aqAvgPres', 
       'elemToUpdateW', '_elemToUpdateW', 'elemToUpdateNW', '_elemToUpdateNW', 
       'elemToUpdate', '_elemToUpdate', 'validToShrink', 'validToGrow', 
       'gasConc', 'aqueousMoles', 'flux', 'netFlux', 'j', 'ii', 'totTime', 'totalTime']

    state = {attr: getattr(self, attr) for attr in state_attrs}    
    joblib.dump(state, fname, compress=3)
    

def round_down_to_n_sf(x, n):
    ''' round down x to n significant figures '''
    if x == 0:
        return 0
    factor = 10 ** (math.floor(math.log10(abs(x))) - (n - 1))
    return math.floor(x / factor) * factor


def initialize_test5D(self):
    ''' reinitialize the test5D network ''' 
    self.fluid[:] = 0
    self.fluid[self.tList[0]] = 1
    self.hasWFluid[:] = True
    self.hasWFluid[self.tList[0]] = False
    self.hasNWFluid[:] = False
    self.hasNWFluid[self.tList[0]] = True
    self.areaWPhase[self.fluid==0] = self.areaSPhase[self.fluid==0]
    self.areaWPhase[self.fluid==1] = 0.0
    self.areaNWPhase[self.fluid==0] = 0.0
    self.areaNWPhase[self.fluid==1] = self.areaSPhase[self.fluid==1]
    self.clusterW.members[:] = False
    self.clusterW.members[1] = True
    self.clusterW.members[1,self.fluid==1] = False
    self.clusterNW.members[:] = False
    self.clusterNW.members[1,self.fluid==1] = True
    self.clusterNW.pc[:] = 0.0
    self.clusterNW.pc[1] = 1/self.H
    self.clusterW_ID[1:-1] = 1
    self.clusterW_ID[self.fluid==1] = -5
    self.clusterNW_ID[:] = -5
    self.clusterNW_ID[self.fluid==1] = 1
    dt = 1
        

            
            
 