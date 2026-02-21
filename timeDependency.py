import numpy as np
from time import time
import math
import os

<<<<<<< Updated upstream
#from clustering import Cluster
from analytical_numerical_study import plotClass


=======
import pnflowPy.tPhaseD as tPhaseD
import pnflowPy.tPhaseImb as tPhaseImb
import cluster as doClust
from OstRipening.network_numba import NetworkJit, ClusterJit
import OstRipening.temp as temp
from OstRipening.utilities_numba import *
from numba import prange, njit


#MEMORY_DIR = f"../ostwald_ripening_results/impP_10000_D_1dot8minus9_H_6dot9minus6_untrappedConstraned_PcDupdated"
MEMORY_DIR = f"ostwald_ripening_results/201225_maxPc1e5/impP_1e6_D_4dot89minus9_H_7dot8minus6_clustPcalpha0dot56_test"
os.makedirs(MEMORY_DIR, exist_ok=True)
_temp = temp.TempArrays()
alpha = 0.56

>>>>>>> Stashed changes
class TimeDependency:
    def __init__(self, obj, Pc, T=298, imposedP=1e6, H=7.8e-6, D=7.3e-9, 
                 steps=10, dt=0.0005, adjustTime=False):
        self.obj = obj

        # initialising the parameters
<<<<<<< Updated upstream
        self.steps = steps
        self.obj.H = H  #mol m-3 pa-1
        self.D = D  #m2/s
        self.num = 1
        self.obj.R = 8.314  #J/mol.K
        self.obj.T = T    #K
        self.obj.imposedP = imposedP #imposedP    #1MPa
        self.Pc = Pc
        self._dt = dt
        self.adjustTime = adjustTime

        '''# for test5D
        initialize_test5D(self)
        '''

        # update cluster 0
        self.do.check_Trapping_Clustering(
            self.elementListS[self.clusterNW_ID==0], self.hasNWFluid.copy(), 1, 
            self.capPresMin, True)
=======
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
>>>>>>> Stashed changes
        
        # setting up the arrays for elements ...
        self.len_tij = self.LenEq()
        self.gasConc = np.zeros([self.totElements, 2], dtype='float')
        self.dissolvedMoles = np.zeros(self.totElements, dtype='float')
        self.delta_nMoles = np.zeros(self.totElements, dtype='float')
        self.obj.satList = np.zeros(self.totElements)
        self.obj.satList[1:-1] = self.areaWPhase[1:-1]/self.areaSPhase[1:-1]

<<<<<<< Updated upstream
        # resize the cluster arrays
        self.resizeClusters()
        
        # setting up useful arrays
        self.settingUpArrays()

        # initializing moles, pc and concentration for NW cluster
        self.initializingPcMolesNWCluster()
        
        # initializing moles, pc and concentration for W/NW pores/throats
        self.initializingMolesConcElements()

        # initialize the flow rates of the phases
        self.initializeFlowrate()

        # for writing data/results
        self.resultsP_str = "# Step,totalFlux,total_delta_nMoles,Sw, \
            #cluster_growth,#cluster_shrinkage,AvgPressure,totMoles_W,totMoles_NW,totMoles"
        self.__fileName__()
        print('Im in timeDependency!')
=======
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
    self.resultsP_str = "# Step,totalFlux,total_delta_nMoles,Sw, \
        #cluster_growth,#cluster_shrinkage,AvgPressure,totMoles_W,totMoles_NW,totMoles"
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

    # initializing and updating moles and concentration for elements
    self.gasConc = np.zeros((self.totElements, 2), dtype=np.float64)
    self.aqueousMoles = np.zeros(self.totElements, dtype=np.float64)
    #from IPython import embed; embed()
    
    updateMolesConcElements(self, clust)
    self.flux = np.zeros(self.nValid, dtype=np.float32)
    self.netFlux = np.zeros(self.totElements, dtype=np.float32)
    self.netFluxClusters = np.zeros(nClust, dtype=np.float32)
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

    self.gasConc[self._elemToUpdateNW,0] = self.H*clust.pc[clust.clusterID[self._elemToUpdateNW]]
    self.gasConc[self._elemToUpdateW,0] = self.H*self.aqAvgPres
    self.aqueousMoles[self._elemToUpdate] = (
        self.gasConc[self._elemToUpdate,0]*self.volarray[self._elemToUpdate])


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


def determineTimeStep(self):
    ''' choose a dt such that no updated concentration is negative and
        the direction of flow is not reversed '''
    
    # time-step based on moles and flux
    # cond = np.flatnonzero(self.elemToUpdateW & (self.netFlux<0.0))
    # condC = np.flatnonzero(self.netFluxClusters<0.0)
    # cond = self.elemToUpdateW & (self.netFlux<0.0)
    # ind = self._elemToUpdateW[(self.netFlux[self._elemToUpdateW] < 0.0)]
    # condC = (self.netFluxClusters<0.0)

    # self.dt = min([
    #     #(-self.aqueousMoles[cond]/self.netFlux[cond]).min(initial=self._dt),
    #     ((self.minMoles[ind]-self.aqueousMoles[ind])/self.netFlux[ind]).min(initial=self._dt),
    #     (-self.clusterNW.moles[condC]/self.netFluxClusters[condC]).min(initial=self._dt)])

    ind = np.flatnonzero(self.netFlux<0.0)
    if ind.any():
        self.dt = np.clip(
            ((self.minMoles[ind]-self.aqueousMoles[ind])/self.netFlux[ind]).min(), 0.0, self._dt)
    
    if self.dt==0.0:
        print(f'self.dt = {self.dt}')
        from IPython import embed; embed()


def computeFluxes1(self, advection=False):
    ''' compute flux for each element and cluster '''
    
    try:
        self.flux = self.D*self.len_tij_valid*(
            self.gasConc[self.TPValid, 0] - self.gasConc[self.TValid, 0])
        assert not advection
    except AssertionError:
        self.flux += (self.flowrateNW_TValid*self.aqueousMoles[self.upstreamElement]/
                    self.volarray[self.upstreamElement]*self.satList[self.upstreamElement])
    
    # compute the net flux for each element
    self.netFlux[self.tList] = np.bincount(self.tValid, self.flux, self.nThroats)[1:]
    self.netFlux[self.poreListS] = np.bincount(self.TPValid, -self.flux, self.nPores+2)
    
    # compute the net flux for each cluster
    self.netFluxClusters = np.bincount(
        self.clusterNW_ID[self._elemToUpdateNW], self.netFlux[self._elemToUpdateNW],
        self.clusterNW.nClusters)


@njit(fastmath=False, parallel=True)
def fast_flux_kernel(D, H, RT, flux, netFlux, len_tij_valid, gConc0, gConc1, volarray,
    TPValid, TValid, clusterID, clust_moles, clust_volume, clust_pc, clust_members, clust_size, 
    members_offset, elemToUpdateW, elemToUpdateNW, aqMoles, minMoles, imposedP, moles_tol, _dt):
    
    fSize = flux.size
    for i in prange(fSize):
        p, t = TPValid[i], TValid[i]
        f = D * len_tij_valid[i] * (gConc0[p] - gConc0[t])
        
        if f > 0:
            if (aqMoles[p] - minMoles[p]) < moles_tol:
                f = 0.0
        elif f < 0:
            if (aqMoles[t] - minMoles[t]) < moles_tol:
                f = 0.0

        if abs(f) < 1e-25:
            f = 0.0
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
            gConc1[e] = aqMoles[e]/volarray[e]

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
        gConc1[mem] = H*clust_pc[k]

    return dt



def computeFluxesMolesPcConc(self, clust, advection=False, moles_tol=1e-18):
    ''' compute flux for each element and cluster '''

    self.netFlux[:] = 0.0
    self.netFluxClusters[:] = 0.0
    self.dt = fast_flux_kernel(self.D, self.H, self.RT, self.flux, self.netFlux, self.netFluxClusters, self.len_tij_valid, 
        self.gasConc[:,0], self.gasConc[:,1], self.volarray, self.TPValid, self.TValid, 
        clust.clusterID, clust.moles, clust.volume, clust.pc, clust.members, clust.size, clust.members_offset, 
        self.elemToUpdateW, self.elemToUpdateNW, self.aqueousMoles, self.minMoles, self.imposedP, moles_tol, self._dt)
    

def computeFluxes2(self, advection=False, moles_tol=1e-18):
    ''' compute flux for each element and cluster '''

    try:
        gasConc0 = self.gasConc[:, 0]
        self.flux[:] = self.D*self.len_tij_valid*(gasConc0[self.TPValid] - gasConc0[self.TValid])
        indP, indT = np.flatnonzero(self.flux>0.0), np.flatnonzero(self.flux<0.0)
        arrP, arrT = self.TPValid[indP], self.TValid[indT]
        self.flux[indP] *= (self.aqueousMoles[arrP]-self.minMoles[arrP]>=moles_tol)
        self.flux[indT] *= (self.aqueousMoles[arrT]-self.minMoles[arrT]>=moles_tol)
        assert not advection
    except AssertionError:
        self.flux += (self.flowrateNW_TValid*self.aqueousMoles[self.upstreamElement]/
                    self.volarray[self.upstreamElement]*self.satList[self.upstreamElement])

    # compute the net flux for each element
    self.netFlux[self.tList] = np.bincount(self.tValid, self.flux, self.nThroats)[1:]
    self.netFlux[self.poreListS] = np.bincount(self.TPValid, -self.flux, self.nPores+2)

    #compute the net flux for each cluster
    self.netFluxClusters = np.bincount(
        self.clusterNW_ID[self._elemToUpdateNW], self.netFlux[self._elemToUpdateNW],
        self.clusterNW.nClusters)


def computeMolesPcConc(self, moles_tol, pc_tol, max_iter):
    ''' compute and update the moles, pc and concentration of gas in each element and cluster'''

    #determineTimeStep(self)
    clust = self.clusterNW

    ''' update element filled with WP properties '''
    
    netMoles = self.netFlux*self.dt
    ind = np.flatnonzero(netMoles<-self.aqueousMoles)
    netMoles[ind] = -self.aqueousMoles[ind]
    netMolesClusters = self.netFluxClusters*self.dt
    np.add.at(netMolesClusters, self.clusterNW_ID[ind], -(netMoles[ind]+self.aqueousMoles[ind]))
    
    ''' update NWP cluster properties '''
    keys = np.flatnonzero(netMolesClusters!=0.0)
    clust.moles[keys] = np.maximum(clust.moles[keys]+netMolesClusters[keys], 0.0)
    clust.pc[keys] = np.maximum((clust.moles[keys]*self.RT/clust.volume[keys]) - self.imposedP, 1e-3)
    self.gasConc[self._elemToUpdateNW,1] = self.H*clust.pc[
        clust.clusterID[self._elemToUpdateNW]]
    
    self.aqueousMoles[self._elemToUpdateW] += netMoles[self._elemToUpdateW]
    #self.gasConc[self._elemToUpdateW, 1] = self.aqueousMoles[self._elemToUpdateW].sum()/self.volarray[self._elemToUpdateW].sum()
    #self.aqueousMoles[self._elemToUpdateW] = self.gasConc[self._elemToUpdateW, 1]*self.volarray[self._elemToUpdateW]
    # if (self.volarray[self._elemToUpdateW]==0.0).any():
    #     print('one of the volume is now 0!!!')
    #     from IPython import embed; embed()
    #from IPython import embed; embed()
    
    #np.divide(self.aqueousMoles, self.volarray, out=self.gasConc[:,1], where=self.elemToUpdateW & self.validVolumes)
    self.gasConc[self._elemToUpdateW, 1] = self.aqueousMoles[self._elemToUpdateW]/self.volarray[self._elemToUpdateW]


    # target_vol = self.volarray[self._elemToUpdateW]
    # conc = np.zeros_like(target_vol)
    # np.divide(
    #     self.aqueousMoles[self._elemToUpdateW], 
    #     target_vol, 
    #     out=conc,
    #     where=target_vol != 0
    # )
    # self.gasConc[self._elemToUpdateW, 1] = conc
    # aqAvgPres = self.aqueousMoles[self._elemToUpdateW].sum()/self.volarray[self._elemToUpdateW].sum()
    # self.minMoles = aqAvgPres*self.volarray


def computeFluxesMolesPcConc1(self, clust, moles_tol):
    return computeFluxesMolesPcConc_numba(
        self.gasConc[:,0], self.gasConc[:,1], self.aqueousMoles, self.minMoles,
        self.len_tij_valid, self.TValid, self.TPValid, self.flux, self.netFlux, self.netFluxClusters,
        self._elemToUpdateNW, self._elemToUpdateW, clust.clusterID, clust.moles, clust.volume, clust.pc, 
        self.D, self.H, self.RT, self.imposedP, self.volarray, self._dt, moles_tol, self.tValid, self.tList, self.poreListS)


#@njit(cache=True, parallel=True)
def computeFluxesMolesPcConc_numba(
    gasConc0, gasConc1, aqueousMoles, minMoles, len_tij_valid, TValid, TPValid,
    flux, netFlux, netFluxClusters, elemToUpdateNW, elemToUpdateW, clusterNW_ID, 
    cluster_moles, cluster_volume, cluster_pc, D, H, RT, imposedP, volarray, _dt, moles_tol, tValid, tList, poreListS):

    nValid = TValid.size
    netFlux[:] = 0.0
    netFluxClusters[:] = 0.0

    for i in range(nValid):
        p = TPValid[i]
        t = TValid[i]

        # force same evaluation order as NumPy
        diff = gasConc0[p] - gasConc0[t]
        tmp  = diff * len_tij_valid[i]
        flux[i] = tmp * D
        if flux[i] > 0.0:
            if (aqueousMoles[p] - minMoles[p]) < moles_tol:
                flux[i] = 0.0
        elif flux[i] < 0.0:
            if (aqueousMoles[t] - minMoles[t]) < moles_tol:
                flux[i] = 0.0

        netFlux[t] += flux[i]
        netFlux[p] -= flux[i]

    from IPython import embed; embed()
    

    dt = determineTimeStep_numba(netFlux, _dt, minMoles, aqueousMoles)
    for e in elemToUpdateW:
        dm = netFlux[e] * dt
        if dm < -aqueousMoles[e]: 
            dm = -aqueousMoles[e]
        aqueousMoles[e] += dm
        v = volarray[e]
        if v > 0.0:
            gasConc1[e] = aqueousMoles[e] / v
        else:
            gasConc1[e] = 0.0

    for e in elemToUpdateNW:
        cid = clusterNW_ID[e]
        netFluxClusters[cid] += netFlux[e]

    nClusters = netFluxClusters.size
    for cid in range(nClusters):
        dm = netFluxClusters[cid] * dt
        if dm==0.0: continue
        
        m = cluster_moles[cid] + dm
        if m < 0.0: m = 0.0
        cluster_moles[cid] = m

        pc = (m * RT / cluster_volume[cid]) - imposedP
        if pc < 1e-3: pc = 1e-3
        cluster_pc[cid] = pc

    for i in range(elemToUpdateNW.size):
        e = elemToUpdateNW[i]
        gasConc1[e] = H * cluster_pc[clusterNW_ID[e]]

    return dt


@njit(cache=True)
def determineTimeStep_numba(netFlux, _dt, minMoles, aqueousMoles):
    ''' choose a dt such that no updated concentration is negative and
        the direction of flow is not reversed '''

    n = netFlux.size
    dt = _dt
    for e in range(n):
        if netFlux[e] < -1e-16:
            tratio =  (minMoles[e] - aqueousMoles[e])/netFlux[e]
            if tratio < 0.0:
                return 0.0
            if dt > tratio:
                dt = tratio

    return dt

@njit
def simulate_kernel(clust, ii, startTime, totTime, totalTime, minTime, moles_tol, pc_tol, max_iter):
    nClusters = clust_pc.size

    triggerGrowth, triggerShrinkage = False, False
    for k in prange(nClusters):
        if validToShrink[k] and clust_pc[k] < clust_pcShrink[k]:
            clustersToShrink[k] = True
            triggerShrinkage = True
        elif validToGrow[k] and clust_pc[k] > clust_pcMax[k]:
            clustersToGrow[k] = True
            triggerGrowth = True

    if not triggerShrinkage and not triggerGrowth:
        netFlux[:] = 0.0
        dt = fast_flux_kernel(D, H, RT, flux, netFlux, len_tij_valid, gasConc[:,0], gasConc[:,1], 
            volarray, TPValid, TValid, clusterID, clust_moles, clust_volume, clust_pc, clust_members, 
            clust_size, clust_members_offset, elemToUpdateW, elemToUpdateNW, aqueousMoles, minMoles, 
            imposedP, moles_tol, _dt)
        
        totTime += dt

    if triggerShrinkage:
        imbEvents += clustersToShrink.sum()
        shrinkCluster(clustersToShrink)
    
    if triggerGrowth:
        drainEvents += condGrowth.sum()
        growCluster(clustersToGrow)



def simulate(self, clust, ii, startTime, totTime, totalTime, minTime, moles_tol, pc_tol, max_iter):
    ''' main simulation loop for time-dependent Ostwald ripening '''
    self.updateClust = False
    cond1 = (clust.pc<clust.pcShrink)&clust.valClust&(clust.pcShrink>self.oldaqAvgPres)
    cond2 = ~cond1 & (clust.pc>clust.pcMax)&clust.valClustD&(clust.pcMax<self.oldaqAvgPres)
    #cond2 = ~cond1 & (clust.moles>clust.molesMax)&clust.valClustD&(clust.pc>self.oldaqAvgPres)
    if not cond1[clust.valClust].any() and not cond2[clust.valClustD].any():
        #computeFluxes(self)
        #computeMolesPcConc(self, moles_tol, pc_tol, max_iter)
        computeFluxesMolesPcConc(self, clust, moles_tol)
        
        totTime += self.dt
    else:
        self.dt = 0.0
        self.updateClust = True
        self.gasConc[self._elemToUpdate,1] = self.gasConc[self._elemToUpdate,0]
    
    if self.updateClust or (totTime>minTime):     #check for every second simulation
        totalTime += totTime 
        totTime = 0.0

        try:
            condShrink = cond1 #& self.valClust
        except ValueError:
            # a resizing had occured after the creation of cond1!!!
            #condShrink = clust.valClust&(clust.moles<clust.molesShrink)
            condShrink = clust.valClust & (clust.pc<clust.pcShrink)
        try:
            if condShrink.any():
                keys = np.flatnonzero(condShrink).astype(np.int32)
                toImbibe = clust.toImbibe[keys]
                print('@@:   ', keys, toImbibe) 
                #from IPython import embed; embed()
                # if 501 in keys and 22567 in toImbibe: # or (339 in keys and 7943 in toImbibe):
                #     from IPython import embed; embed()
                clust.shrinkCluster(keys, self)
                clust.imbEvents += condShrink.sum()
                updateLists(self, toImbibe)
                self.updateClust = False
        except:
            print('there is an error in the simulation, check line 1390!!!')
            from IPython import embed; embed()
        
        clust.valClustD[:] = clust.valClust & (clust.toDrain<=self.totElements)
        try:
            condGrowth = cond2 & clust.valClustD
        except ValueError:
            # an array had occured after the creation of cond1!!!
            #condGrowth = clust.valClustD&(clust.moles>=clust.molesMax
            condGrowth = clust.valClustD&(clust.pc>clust.pcMax)            
        try:
            if condGrowth.any():
                
                keys = np.flatnonzero(condGrowth).astype(np.int32)
                toDrain = clust.toDrain[keys]
                # if (clust.clusterID[toDrain]>=0).any():
                #     print('this is abomination !!!!')
                #     from IPython import embed; embed()
                print('$$:  ', keys, toDrain)
                # if 189 in keys and 7232 in toDrain:
                #     from IPython import embed; embed()
                #input('waittt!!!')
                #from IPython import embed; embed()
                toDrain = clust.growCluster(keys, self)
                clust.drainEvents += condGrowth.sum()
                updateLists(self, toDrain, 1)
                self.updateClust = False
        except:
            print('there is an error in the simulation, check line 1397!!!')
            from IPython import embed; embed()
>>>>>>> Stashed changes

        #from IPython import embed; embed()
    
    def __getattr__(self, name):
        return getattr(self.obj, name)
    
    def resizeClusters(self):
        ''' resize the W/NW clusters by removing empty clusters id at the end '''

        nClustW = self.clusterW_ID.max()+1
        self.clusterW.members = self.clusterW.members[:nClustW]
        self.clusterW.pc = self.clusterW.pc[:nClustW]
        self.clusterW.clustConToInlet = self.clusterW.clustConToInlet[:nClustW]
        self.clusterW.connected= self.clusterW.connected[:nClustW]
        self.clusterW.keys = self.clusterW.keys[:nClustW]
        self.clusterW.availableID.clear()

        nClustNW = self.clusterNW_ID.max()+1
        self.clusterNW.members = self.clusterNW.members[:nClustNW]
        self.clusterNW.pc = self.clusterNW.pc[:nClustNW]
        self.clusterNW.clustConToInlet = self.clusterNW.clustConToInlet[:nClustNW]
        self.clusterNW.connected = self.clusterNW.connected[:nClustNW]
        self.clusterNW.keys = self.clusterNW.keys[:nClustNW]
        self.clusterNW.availableID.clear()

    def settingUpArrays(self):
        ''' sets up useful arrays for the time-dependent simulation '''

        # identify valid pore-throat pairs and clusters
        self.TPCond = (self.TPConnections[1:]>0) # location of valid pores connected to each throats
        self.TValid = np.dstack((self.tList, self.tList))[0][self.TPCond] # valid throats (Oren)
        self.tValid = self.TValid-self.nPores-1
        self.TPValid = self.TPConnections[1:][self.TPCond] # valid pores connected to each throats
        self.valClust = (self.clusterNW.members.any(axis=1)) # returns valid clusters
        self.len_tij_valid = self.len_tij[self.TPCond] # returns len_tij of valid throat-pore pairs
        toUpdateNW = self.valClust & (~self.clusterNW.connected) # clusters that are to be updated

        # setting up arrays for clusters
        nClustNW = self.clusterNW_ID.max()+1
        self.clusterNW.satList = 1-self.satList
        self.clusterNW.satList[[-1,0]] = 0.0
        self.clusterNW.neighbours = np.zeros([nClustNW, self.totElements], dtype=bool)
        self.clusterNW.volume = np.bincount(
            self.clusterNW_ID[self.hasNWFluid], 
            self.volarray[self.hasNWFluid]) #*self.clusterNW.satList[self.hasNWFluid])
        self.clusterNW.updateNeighMatrix()
        self.clusterNW.toDrain = np.ones(nClustNW, dtype=int)*(self.totElements+1)
        self.clusterNW.toImbibe = np.ones(nClustNW, dtype=int)*(self.totElements+1)
        self.clusterNW.volHigh = np.zeros(nClustNW)
        self.clusterNW.volLow = np.zeros(nClustNW)
        self.clusterNW.pcHigh = np.full(nClustNW, 1e30)
        self.clusterNW.pcLow = np.full(nClustNW, -1e30)
        self.clusterNW.molesHigh = np.full(nClustNW, 1e30)      # maximum expected value
        self.clusterNW.molesLow = np.full(nClustNW, 1e-30)      # least expected value
        validKeys = np.where(self.valClust)[0]
        [self.clusterNW.updateToDrainImbibe(k) for k in validKeys]

        # setting up arrays for elements
        self.elemToUpdateW = (self.fluid==0)
        self.elemToUpdateW[[-1,0]] = False                           # boolean
        self._elemToUpdateW = self.elementListS[self.elemToUpdateW]  # index
        self.elemToUpdateNW = np.zeros(self.totElements, dtype=bool)  #boolean
        self.elemToUpdateNW[self.hasNWFluid] = toUpdateNW[self.clusterNW_ID[self.hasNWFluid]]
        self._elemToUpdateNW = self.elementListS[self.elemToUpdateNW]           # index
        self.toUpdateNW = np.array(self.clusterNW.keys)[toUpdateNW]  # index
        self.elemToUpdate = self.elementListS[self.elemToUpdateW|self.elemToUpdateNW]   # index


    def initializingPcMolesNWCluster(self):
        ''' initialize moles, pc and gas conc for  NW cluster '''
        condD = (self.clusterNW.toDrain<self.totElements)
        condI = (self.clusterNW.toImbibe<self.totElements)

         # initializing cluster moles and concentrations
        self.clusterNW.molesTotal = ((self.imposedP+self.clusterNW.pc)*
                                     self.clusterNW.volume/(self.R*self.T))
        self.clusterNW.molesHigh[condD] = ((self.imposedP+self.clusterNW.pcHigh[condD])*
                                 self.clusterNW.volHigh[condD]/(self.R*self.T))
        self.clusterNW.molesLow[condI] = np.clip((self.imposedP+self.clusterNW.pcLow[condI])*
                                 self.clusterNW.volLow[condI]/(self.R*self.T), 1e-30, None)

        self.initialMoles = self.clusterNW.molesTotal.copy()
        self.initialPc = self.clusterNW.pc.copy()
        self.betaLow = (self.initialPc-self.clusterNW.pcLow)/(
            self.initialMoles-self.clusterNW.molesLow)
        self.betaHigh = (self.clusterNW.pcHigh-self.initialPc)/(
            self.clusterNW.molesHigh-self.initialMoles)
        self.factLow = 1/(1+(self.betaLow*self.H*self.clusterNW.volume))
        self.factHigh = 1/(1+(self.betaHigh*self.H*self.clusterNW.volume))
        self.clusterNW.pc = self.factLow*(
            (self.clusterNW.pcLow+(self.betaLow*(self.initialMoles-self.clusterNW.molesLow))))   
        self.clusterNW.moles = self.initialMoles - (self.H*self.clusterNW.pc*self.clusterNW.volume)


    def initializingMolesConcElements(self):
        ''' 
            moles of gas in elements filled with WPhase are all updated
            but only members of NWPhase clusters not connected across 
            the network are updated.
        '''
        
<<<<<<< Updated upstream
        ''' initialize the brine to have conc equivalent to Pc!!! '''
        #self.gasConc[self.elemToUpdateW,0] = self.H*3350  #self.Pc
        #self.dissolvedMoles[self.elemToUpdateW] = (
         #   self.gasConc[self.elemToUpdateW,0]*self.volarray[self.elemToUpdateW])
        
        ''' initialize the NW elements according to their trapped Pc!!! '''
        self.gasConc[self.hasNWFluid,0] = self.H*self.clusterNW.pc[
            self.clusterNW_ID[self.hasNWFluid]]
        self.dissolvedMoles[self.hasNWFluid] = self.gasConc[self.hasNWFluid,0]*self.volarray[
            self.hasNWFluid]
        
    def initializeFlowrate(self):
        ''' returns the velocities of fluid across each throat'''
        #gwL = self.do.computegL(self.gWPhase)
        gnwL = self.do.computegL(self.gNWPhase)
        #self.flowrateW, self.flowDirectionW = self.do.computeFlowrate(gwL, 0, self.Pc, True)
        self.flowrateNW, self.flowDirectionNW = self.do.computeFlowrate(gnwL, 1, self.Pc, True)
        self.flowrateNW_TValid = np.dstack((self.flowrateNW, self.flowrateNW))[0][self.TPCond]
        upstreamP1T = self.P1array.copy()
        upstreamP1T[~self.flowDirectionNW] = self.tList[~self.flowDirectionNW]
        upstreamP2T = self.tList.copy()
        upstreamP2T[~self.flowDirectionNW] = self.P2array[~self.flowDirectionNW]
        self.upstreamElement = np.dstack((upstreamP1T, upstreamP2T))[0][self.TPCond]
        
    def updateLists(self, arr, fluid=0):
        cond = True if fluid==0 else False
        self.elemToUpdateW[arr] = cond
        self.elemToUpdateNW[arr] = not cond
        self._elemToUpdateW = self.elementListS[self.elemToUpdateW]
        self._elemToUpdateNW = self.elementListS[self.elemToUpdateNW]
        self.elemToUpdate = self.elementListS[self.elemToUpdateW|self.elemToUpdateNW]
        self.valClust = (self.clusterNW.members.any(axis=1))
        self.toUpdateNW = np.array(self.clusterNW.keys)[self.valClust & (~self.clusterNW.connected)]
    
    def LenEq(self):
        term1 = self.Rarray[self.TPConnections[1:]]/self.areaSPhase[self.TPConnections[1:]]
        term1[np.isnan(term1)] = 0.0
        term2 = self.LTarray/(2*self.areaSPhase[self.tList])
        return 1/(term1+term2[:,np.newaxis])

    def computeFluxes(self, advection=False):
        ''' compute flux for each element and cluster '''
        try:
            flux = self.D*self.len_tij_valid*(
                self.gasConc[self.TPValid, 0] - self.gasConc[self.TValid, 0])
            assert not advection
        except AssertionError:
            flux += (self.flowrateNW_TValid*self.dissolvedMoles[self.upstreamElement]/
                     self.volarray[self.upstreamElement])
        
        # compute the net flux for each element
        self.netFlux[self.tList] = np.bincount(self.tValid, flux, self.nThroats)
        self.netFlux[self.poreListS] = np.bincount(self.TPValid, -flux, self.nPores+2)

        # compute the net flux for each cluster
        self.netFluxClusters = np.bincount(
            self.clusterNW_ID[self._elemToUpdateNW], self.netFlux[self._elemToUpdateNW],
            len(self.clusterNW.keys))
        

    def computeMolesPcConc(self):
        ''' compute and update the moles, pc and concentration of gas in each element and cluster'''
        try:
            assert (self.dissolvedMoles+(self.netFlux*self._dt)>=0).all()
            self.dt = self._dt
        except AssertionError:
            cond=(self.netFlux<0.0)
            tmax = (-self.dissolvedMoles[cond]/self.netFlux[cond]).min()
            print(f'dt = {tmax}')
            self.dt=tmax+1e-7
            self.updateClust = True
=======
        print(("#:%8.6g  \tSimulation Time:%12.6g  \tActual runtime:%6.6g\
                \tTotal cluster moles:%8.6e  \tTotal dissolved moles:%6.6e\
                \tTotal gas moles:%4.10e \tAvg Pressure:%8.6e \tNo Shrinkage:%8.6g\
                \tNo Growth:%8.6g \tSat:%6.6g \tAqAvgPres:%6.6g" % (
                ii, round(totalTime,3), round(time()-startTime,3), self.clusterNW.moles.sum(), 
                #self.dissolvedMoles.sum(), totalMoles,
                self.aqueousMoles.sum(), totalMoles,
                #self.clusterNW.pc[self.valClust].mean(),
                gasAvgPres,
                self.clusterNW.imbEvents, self.clusterNW.drainEvents, self.satW, self.aqAvgPres)))
>>>>>>> Stashed changes
            
        try:
            ''' update element properties '''
            self.delta_nMoles = self.netFlux*self.dt
            newDissolvedMoles = (self.dissolvedMoles[self._elemToUpdateW]+
                                 self.delta_nMoles[self._elemToUpdateW])
            assert not (newDissolvedMoles<0.0).any()
            self.dissolvedMoles[self._elemToUpdateW] = newDissolvedMoles
            self.gasConc[self._elemToUpdateW, 1] = (
                newDissolvedMoles/self.volarray[self._elemToUpdateW])

            ''' update cluster properties '''
            self.delta_nMoles_clusters = self.netFluxClusters*self.dt
            if ((self.clusterNW.molesTotal[self.toUpdateNW]+
                self.delta_nMoles_clusters[self.toUpdateNW]<0.0).any()):
                print('total moles less than zero!!!')
                from IPython import embed; embed()
            self.clusterNW.molesTotal[self.toUpdateNW] += (
                self.delta_nMoles_clusters[self.toUpdateNW])

            toUpdateNW1 = self.toUpdateNW[self.clusterNW.molesTotal[self.toUpdateNW]>
                                          self.initialMoles[self.toUpdateNW]]
            toUpdateNW2 = self.toUpdateNW[self.clusterNW.molesTotal[self.toUpdateNW]<
                                          self.initialMoles[self.toUpdateNW]]
            
            # clusters that gained moles of gas
            self.clusterNW.pc[toUpdateNW1] = self.factHigh[toUpdateNW1]*(
                self.initialPc[toUpdateNW1]+(self.betaHigh[toUpdateNW1]*(
                self.clusterNW.molesTotal[toUpdateNW1]-self.initialMoles[toUpdateNW1]))) 
            # clusters that lost moles of gas
            self.clusterNW.pc[toUpdateNW2] = self.factLow[toUpdateNW2]*(
                self.clusterNW.pcLow[toUpdateNW2]+(self.betaLow[toUpdateNW2]*(
                self.clusterNW.molesTotal[toUpdateNW2]-self.clusterNW.molesLow[toUpdateNW2])))
            
            # update conc and moles but check if the cluster gas becomes completely dissolved
            gasConcClusters = self.H*self.clusterNW.pc
            dissolvedMolesClusters = (gasConcClusters[self.toUpdateNW]*
                                      self.clusterNW.volume[self.toUpdateNW])
            cond = (dissolvedMolesClusters>self.clusterNW.molesTotal[self.toUpdateNW])
            dissolvedMolesClusters[cond] = self.clusterNW.molesTotal[self.toUpdateNW][cond]
            gasConcClusters[self.toUpdateNW[cond]] = (
                dissolvedMolesClusters[cond]/self.clusterNW.volume[self.toUpdateNW[cond]])
            self.gasConc[self._elemToUpdateNW,1] = gasConcClusters[
                self.clusterNW_ID[self._elemToUpdateNW]]
            self.dissolvedMoles[self._elemToUpdateNW] = (
                self.gasConc[self._elemToUpdateNW,1]*self.volarray[self._elemToUpdateNW])
            self.clusterNW.moles[self.toUpdateNW] = (
                self.clusterNW.molesTotal[self.toUpdateNW]-dissolvedMolesClusters)
            
        except AssertionError:
            print('Mole is negative at element level,  try a smaller time step!!!')
            from IPython import embed; embed()
            raise AssertionError
        
<<<<<<< Updated upstream
=======
    if (self.gasConc[:,1]<0.0).any():
        print('## a new element  concentration is negative!!!')
        from IPython import embed; embed()    
        
    return ii, totTime, totalTime


def simulateOstRip(self, implicit=True, freshStart=True):
    print('Im in simulateOstRip  ', freshStart)
    duration = 3600*24*2  # 24-hours
    duration = 10
    st = time()
>>>>>>> Stashed changes
    
    def simulateOstRip(self, implicit=True):
        duration = 3600*24*2  # 24-hours
        st = time()
        totalTime, totTime = 0.0, 0.0
        self.netFlux = np.zeros(self.totElements)
        self.updateClust = False
<<<<<<< Updated upstream

        if not os.path.isfile('volarray_bent.dat'):
            np.savetxt('volarray_bent.dat', self.volarray)
        if not os.path.isfile('coordinates_bent.dat'):
            np.savetxt('coordinates_bent.dat', 
                       np.column_stack((self.x_array, self.y_array, self.z_array)))
=======
        clust = self.clusterNW
        clust.drainEvents, clust.imbEvents = 0, 0
        totalMoles = (
            (self.volarray*self.gasConc[:,0])[~self.elemToUpdateNW].sum()+
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
        #from IPython import embed; embed()
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

    size0 = clust.size[0]
    from pnflowPy.cluster import doClustering_numba
    
    #self.oldaqAvgPres = self.aqAvgPres
    
    timeToSave = minTime*self.j
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
                    with open(os.path.join(MEMORY_DIR, f"netsim_time_Dependent_{self.j}_hours.pkl"),
                            "wb") as f:
                        dill.dump(self, f)
                    
                    self.j += 1
                    timeToSave = minTime*self.j
                except:
                    print('there is an error in pickling!!!')
                    from IPython import embed; embed()
                
        except:
            print('there is an error in the simulation, check line 1438!!!')
            from IPython import embed; embed()
>>>>>>> Stashed changes
            
        # to run for 1 day in 1s intervals
        minTime = 10
        '''nRow = int(duration/minTime)+1
        gasConc = np.memmap('gasConcOstRipening_bent.dat', dtype='float32', 
                            mode='w+', shape=(nRow,self.totElements))
        clustPc = np.memmap('clustPcOstRipening_bent.dat', dtype=float,
                            mode='w+', shape=(nRow, self.clusterNW.pc.size))
        timeArray = np.zeros(nRow)

        gasConc[0] = self.gasConc[:, 0]
        clustPc[0] = self.clusterNW.pc'''
        ii = 1
        self.clusterNW.drainEvents, self.clusterNW.imbEvents = 0, 0
        self.valClustD = self.valClust & (self.clusterNW.toDrain<=self.totElements)

        def _fff(ii, totTime, totalTime):
            self.computeFluxes()
            self.computeMolesPcConc()
            try:
                totTime += self.dt
                assert self.updateClust or (totTime>minTime)     #check for every second simulation
                condShrink = (self.clusterNW.moles<=self.clusterNW.molesLow) & self.valClust
                #condGrowth = (self.clusterNW.moles>=self.clusterNW.molesHigh) & self.valClust
                condGrowth = ((self.clusterNW.moles[self.valClustD]+
                               self.dissolvedMoles[self.clusterNW.toDrain[self.valClustD]])>=
                                self.clusterNW.molesHigh[self.valClustD])
                totalTime += totTime
                totTime = 0.0

                try:
                    assert condShrink.any()
                    keys = np.array(self.clusterNW.keys)[condShrink]
                    print('@@:   ', keys)
                    toImbibe = self.clusterNW.toImbibe[keys]
                    self.clusterNW.shrinkCluster(keys, self)
                    self.clusterNW.imbEvents += condShrink.sum()
                    self.updateLists(toImbibe)
                    self.updateClust = False
                except AssertionError:
                    pass

                try:
                    assert condGrowth.any()
                    keys = np.array(self.clusterNW.keys)[self.valClustD][condGrowth]
                    print('$$:  ', keys)
                    toDrain = self.clusterNW.toDrain[keys]
                    #from IPython import embed; embed()
                    self.clusterNW.growCluster(keys, self)
                    self.clusterNW.drainEvents += condGrowth.sum()
                    self.updateLists(toDrain, 1)
                    self.updateClust = False
                except AssertionError:
                    pass
                
                sat = (self.satList[self.isinsideBox]*self.volarray[
                    self.isinsideBox]).sum()/self.totVoidVolume
                print(("#:%8.6g  \tSimulation Time:%12.6g  \tActual runtime:%6.6g\
                    \tTotal cluster moles:%8.6e  \tTotal dissolved moles:%6.6e\
                    \tTotal gas moles:%6.6e \tAvg Pressure:%8.6e \tNo Shrinkage:%8.6g\
                    \tNo Growth:%8.6g \tSat:%6.6g" % (
                ii, round(totalTime,3), round(time()-st,3), self.clusterNW.moles.sum(), 
                self.dissolvedMoles.sum(), self.clusterNW.moles.sum()+self.dissolvedMoles.sum(),
                self.clusterNW.pc[self.valClust].mean(),
                self.clusterNW.imbEvents, self.clusterNW.drainEvents, sat)))

                #gasConc[ii] = self.gasConc[:, 1] 
                #clustPc[ii] = self.clusterNW.pc
                #timeArray[ii] = totalTime
                ii += 1
            except AssertionError:
                pass
                        
            self.gasConc[self.elemToUpdate,0] = self.gasConc[self.elemToUpdate,1]
            return ii, totTime, totalTime

        #from IPython import embed; embed()
        while totalTime < duration:
            ii, totTime, totalTime = _fff(ii, totTime, totalTime)
            print(ii)
        timeArray = np.savetxt('timeArray_bent.dat', timeArray)
        print('::::::::::::::::::::::::::::')
        from IPython import embed; embed()
        plot.plot_conc_t()
        plot = plotClass('bent', (nRow, self.totElements), D=self.D)
        plot.plot_conc_x()
        plot.plot_conc_universal()
        from IPython import embed; embed()


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
        

            
            
            

    

    



            

        
    
        



