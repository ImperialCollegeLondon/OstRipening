import numpy as np
from time import time
from joblib import Parallel, delayed
from multiprocessing import shared_memory
from joblib import dump, load, Parallel, delayed
import os
from scipy.optimize import bisect, brentq
from scipy.optimize import root_scalar
from scipy.optimize import curve_fit
import multiprocessing
import dill

import pnflowPy.utilities as do
from analytical_numerical_study import plotClass
import pnflowPy.tPhaseD as tPhaseD
import pnflowPy.tPhaseImb as tPhaseImb
import clustering as doClust

# Define a memory-mapped directory (ensure it exists)
#MEMORY_DIR = "./joblib_memmap_new"
MEMORY_DIR = "./ostrip_initial_0_clusterPc_equal_PcD_and_PcI_alpha_070_latest"
#MEMORY_DIR = "./ostrip_initial_0_PcD_equal_PistonPcRec_and_PcI_alpha_040_imposedP_2MPa"
MEMORY_DIR = "./ostrip_initial_0_PcD_equal_PistonPcRec_maxPc_1e6_10072025_test_2"
os.makedirs(MEMORY_DIR, exist_ok=True)
global_memmaps = {}
memmap_shapes = {} 
alpha = 1.0
#maxCount = 0
class TimeDependency:
    def __init__(self, obj, Pc, T=298, imposedP=1e6, H=7.8e-6, D=7.3e-9, 
                 steps=10, dt=0.0005, adjustTime=False):
    
        # initialising the parameters
        obj.steps = steps
        obj.H = H  #mol m-3 pa-1
        obj.D = D  #m2/s
        obj.num = 1
        obj.R = 8.314  #J/mol.K
        obj.T = T    #K
        obj.imposedP = imposedP    #1MPa
        obj.Pc = Pc
        obj._dt = dt
        obj.adjustTime = adjustTime
        obj.lookup = LookupClass(obj)
        #print('@@@@@@@@@@@@@@@@@@@')

class LookupClass:
    def __init__(self, other):        
        self.sinBeta = np.sin(other.m_halfAngles)
        self.cosBeta = np.cos(other.m_halfAngles)
        self.sinBetacosBeta = self.sinBeta*self.cosBeta
        numerator = other.sigma*np.cos(other.thetaAdvAng.reshape(-1,1)+other.m_halfAngles)
        self._factor = np.divide(numerator, self.sinBeta, where=(self.sinBeta != 0.0))
        
        self._part = (other.m_initedApexDist*np.sin(other.m_halfAngles)/other.sigma)
        cnt = other.clusterNW.pc.size
        self.guessPc, self.guessPc1, self.guessPc2 = np.zeros(cnt),np.zeros(cnt),np.zeros(cnt)
        self.guessVol, self.guessVol1, self.guessVol2 = np.zeros(cnt),np.zeros(cnt),np.zeros(cnt)
        self.guessMoles, self.guessMoles1, self.guessMoles2 = (
            np.zeros(cnt),np.zeros(cnt),np.zeros(cnt))
        
        self.RT = other.R*other.T
        self._impP_RT = other.imposedP/self.RT
        self._RT_H = (1/self.RT -other.H)


def LenEq1(self):
    term1 = self.Rarray[self.TPConnections[1:]]/self.areaSPhase[self.TPConnections[1:]]
    term1[np.isnan(term1)] = 0.0
    term2 = self.LTarray/(2*self.areaSPhase[self.tList])
    return 1/(term1+term2[:,np.newaxis])


def LenEq(self):
    ''' returns the diffusion characteristic length for each P1-T-P2 pair '''
    len_diff = 0.5*(self.LP1array+self.LTarray+self.LP2array)
    radarray = np.array(
        [self.Rarray[self.P1array], self.Rarray[self.tList], self.Rarray[self.P2array]])
    rad_diff = np.nanmin(np.where(radarray==0.0, np.nan, radarray), axis=0)
    area_diff = np.pi*rad_diff**2
    lenEq = area_diff/len_diff
    return np.column_stack([lenEq, lenEq])


def initialize(self):
    # setting up the arrays for elements ...
    st=time()
    self.lkup = LookupClass(self)
    self.len_tij = LenEq(self)
    self.gasConc = np.zeros([self.totElements, 2], dtype=np.float64)
    self.aqueousMoles = np.zeros(self.totElements, dtype=np.float64)
    self.cumNetMolesClusters = np.zeros(self.clusterNW.pc.size, dtype=np.float64)
    self.satList = np.divide(self.areaWPhase, self.areaSPhase, where=(self.areaSPhase != 0.0))
    do.check_Trapping_Clustering(
        self, self.clusterNW[0].members, self.hasNWFluid.copy(), 1, self.clusterNW[0].pc, True)
    
    # resize the cluster arrays
    resizeClusters(self, self.clusterW, self.clusterNW)
    
    # adjust the cluster pc
    adjustClusterPc(self,self.clusterNW,self.clusterNW_ID)
    
    # setting up useful arrays
    settingUpArrays(self, self.clusterNW, False)
    
    # update the entry capillary pressure for drainage
    updatePcD(self)

    # initializing moles, pc and concentration for NW cluster
    initializingPcMolesNWCluster(self, self.clusterNW)

    # initializing moles, pc and concentration for W/NW pores/throats
    self.netFlux = np.zeros(self.totElements)
    self.netMoles = np.zeros(self.totElements)
    initializingMolesConcElements(self)
    
    # initialize the flow rates of the phases
    #initializeFlowrate(self)

    # for writing data/results
    self.resultsP_str = "# Step,totalFlux,total_delta_nMoles,Sw, \
        #cluster_growth,#cluster_shrinkage,AvgPressure,totMoles_W,totMoles_NW,totMoles"
    # self.__fileName__()
    print('Im in timeDependency!')
    print('time spent:  ', time()-st)
    print('alpha:  ', alpha)


def resizeClusters(self, clustW, clustNW):
    ''' resize the W/NW clusters by removing empty clusters id at the end '''

    id = np.setdiff1d(np.where(clustW.size==0)[0], clustW.availableID)
    clustW.availableID.update(id[id>0])
    clustW.neighbours = np.zeros(clustW.members.shape, dtype=bool)
    _ = clustW.updateNeighMatrix(self)

    id = np.setdiff1d(np.where(clustNW.size==0)[0], clustNW.availableID)
    clustNW.availableID.update(id[id>0])
    clustNW.neighbours = np.zeros(clustNW.members.shape, dtype=bool)
    _ = clustNW.updateNeighMatrix(self)


def adjustClusterPc(self, clust, clustID):
    ''' adjust the cluster pc '''
    nC = clust.pc.size
    neighPc = self.PistonPcRec*clust.neighbours
    neighPc[neighPc==0.0] = np.inf
    clustPcD = np.min(neighPc, axis=1)
    clustPcD[np.isinf(clustPcD)] = clust.pc[np.isinf(clustPcD)]
    
    clust.pc[:] = alpha*clust.pc + (1-alpha)*clustPcD
    clust.volume = np.zeros_like(clust.pc)
    mem = self.hasNWFluid.copy()
    tPhaseImb.__CondTPImbibition__(self, mem, clust.pc[self.clusterNW_ID], 
        updateArea=True, overrideTrapping=True)
    vol = doClust.volume_bincount_numba(mem, self.clusterNW_ID, self.areaNWPhase, 
        self.areaSPhase, self.volarray)
    clust.volume[:vol.size] = vol
    
    self.satList[mem] = self.areaWPhase[mem]/self.areaSPhase[mem]
    self.satW = (self.satList[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()/self.totVoidVolume
    
    
def settingUpArrays(self, clust, excludeCircles=False):
    ''' sets up useful arrays for the time-dependent simulation '''
    
    self.tValid = self.TValid-self.nPores-1
    self.valClust = (clust.members.any(axis=1)) # returns valid clusters
    self.len_tij_valid = self.len_tij[self.TPCond[1:]] # returns len_tij of valid throat-pore pairs
    self.len_tij_valid[self.isCircle[self.TPValid]|self.isCircle[self.TValid]] = 0.0 # circular elem
    self.toUpdateNW = np.where((self.valClust & (~clust.connected)))[0] # clusters to be updated

    # setting up arrays for cluste
    clust.totalVolume  = np.bincount(
        self.clusterNW_ID[self.hasNWFluid], self.volarray[self.hasNWFluid], clust.pc.size)
    clust.dVg_dPc = np.zeros(clust.pc.size, dtype=np.float64)  # dVg/dPc for each cluster
    
    # setting up arrays for elements
    self.elemToUpdateW = (self.fluid==0)
    self.elemToUpdateW[[-1,0]] = False                           # boolean
    self._elemToUpdateW = np.flatnonzero(self.elemToUpdateW)  # index
    self.elemToUpdateNW = self.hasNWFluid.copy()
    self._elemToUpdateNW = np.flatnonzero(self.elemToUpdateNW)           # index
    self.elemToUpdate = (self.elemToUpdateW|self.elemToUpdateNW)
    self._elemToUpdate = np.flatnonzero(self.elemToUpdate)   # index
    

def initializingPcMolesNWCluster(self, clust):
    ''' initialize moles, pc and gas conc for  NW cluster '''
    nClustNW = clust.pc.size
    clust.toDrain = np.ones(nClustNW, dtype=np.int64)*(self.totElements+1)
    clust.toImbibe = np.ones(nClustNW, dtype=np.int64)*(self.totElements+1)
    
    clust.pcMax = np.full(nClustNW, 1e30)           # maximum expected value
    clust.pcShrink = np.full(nClustNW, -1e30)       # least expected value
    clust.molesMax = np.full(nClustNW, 1e30)      # maximum expected value
    clust.molesShrink = np.zeros(nClustNW)      # least expected value
    clust.volMax = np.zeros(nClustNW)
    clust.volShrink = np.zeros(nClustNW)
    
    clust.nextPcShrink = clust.pcShrink.copy()
    clust.nextVolShrink = clust.volShrink.copy()
    clust.nextMolesShrink = clust.molesShrink.copy()

    # initializing cluster moles and concentrations
    self.initialPc = clust.pc.copy()
    self.initialVolume = clust.volume.copy()
    clust.moles = (
        (self.imposedP+self.initialPc)*clust.volume/(self.R*self.T)
        + self.H*self.initialPc*(clust.totalVolume-clust.volume))
    self.initialMoles = clust.moles.copy()
    validKeys = np.where(self.valClust)[0]
    updateVolumeShrinkMax(self, clust, validKeys)
    
    #updateVolumeShrinkMax(self, clust, self.valClust)
    clust.updateMolesShrinkGrowth(validKeys, self)
    clust.dVg_dPc[validKeys] = (clust.volMax[validKeys]-clust.volShrink[validKeys])/(
        clust.pcMax[validKeys]-clust.pcShrink[validKeys])
        
    


def updateVolumeShrinkMax1(self, clust, validKeys, update=False):
    clust.elemCornerArea = np.zeros(self.totElements, dtype=bool)
    [clust.updateToDrainImbibe(k, self, update) for k in validKeys]
    
    elemCornerArea = clust.elemCornerArea&(clust.toDrain[self.clusterNW_ID]<self.totElements)
    mem = self.elementListS[elemCornerArea]
    elemCornerArea = (elemCornerArea&(clust.pcMax[self.clusterNW_ID]!=self.elementPc_area))
    clust.computeCornerArea(elemCornerArea, clust.pcMax[self.clusterNW_ID], self)
    vol = self.areaNWPhase[mem]/self.areaSPhase[mem]*self.volarray[mem]
    clust.volMax[validKeys] = np.bincount(self.clusterNW_ID[mem], vol, validKeys.max()+1)[validKeys]
    
    clust.elemCornerArea[clust.toImbibe[validKeys]] = False
    mem = self.elementListS[clust.elemCornerArea]
    elemCornerArea = (clust.elemCornerArea&(clust.pcShrink[self.clusterNW_ID]!=self.elementPc_area))
    clust.computeCornerArea(elemCornerArea, clust.pcShrink[self.clusterNW_ID], self)
    vol = self.areaNWPhase[mem]/self.areaSPhase[mem]*self.volarray[mem]
    clust.volShrink[validKeys] = np.bincount(
        self.clusterNW_ID[mem], vol, validKeys.max()+1)[validKeys]
    

def updateVolumeShrinkMaxOld(self, clust, arrr,  validKeys, update=False):
    print('>>>><<<<<Im in uodateVolumeShrinkMax!!!')
    from IPython import embed; embed()
    clust.elemCornerArea = np.zeros(self.totElements, dtype=bool)
    clust.updateToDrainImbibe(arrr, self)
    #[clust.updateToDrainImbibe(k, self, update) for k in validKeys]
 
    #print('Im in updateVolumeShrinkMax!!!')
    #from IPython import embed; embed()
    mem = clust.elemCornerArea&(clust.toDrain[self.clusterNW_ID]<self.totElements)&self.hasNWFluid
    clust.computeCornerArea(mem, clust.pcMax[self.clusterNW_ID], self)
    mem = self.elementListS[mem]
    vol =  np.bincount(
        self.clusterNW_ID[mem], self.areaNWPhase[mem]/self.areaSPhase[mem]*self.volarray[mem])
    keys =  validKeys[clust.toDrain[validKeys]<self.totElements]
    clust.volMax[keys] = vol[keys]
    
    mem = clust.elemCornerArea
    clust.computeCornerArea(mem, clust.pcShrink[self.clusterNW_ID], self)
    mem = self.elementListS[mem]
    vol =  np.bincount(
        self.clusterNW_ID[mem], self.areaNWPhase[mem]/self.areaSPhase[mem]*self.volarray[mem])
    clust.volShrink[validKeys] = vol[validKeys]
   
    
def updateVolumeShrinkMax(self, clust, arrkeys):
    clust.elemCornerArea = np.zeros(self.totElements, dtype=np.bool_)
    clust.updateToDrainImbibe(arrkeys, self)
    self.valClust = (clust.size>0)
    self.valClustD = self.valClust & (clust.toDrain<=self.totElements)
    self.toUpdateNW = np.array(clust.keys)[self.valClust & (~clust.connected)]
    
    # print('Im in updateVolumeShrinkMax!!!')
    # from IPython import embed; embed()
    
    mem = clust.elemCornerArea & self.valClustD[self.clusterNW_ID]
    tPhaseImb.__CondTPImbibition__(self, mem, clust.pcMax[self.clusterNW_ID], 
        updateArea=True, overrideTrapping=True)
    vol = doClust.volume_bincount_numba(mem, self.clusterNW_ID, self.areaNWPhase, 
        self.areaSPhase, self.volarray)
    clust.volMax[:vol.size] = vol
    
    mem = clust.elemCornerArea
    mem[clust.toImbibe[arrkeys]] = False
    tPhaseImb.__CondTPImbibition__(self, mem, clust.pcShrink[self.clusterNW_ID], 
        updateArea=True, overrideTrapping=True)
    vol = doClust.volume_bincount_numba(mem, self.clusterNW_ID, self.areaNWPhase, 
        self.areaSPhase, self.volarray)
    clust.volShrink[:vol.size] = vol
    
    # cond = np.zeros_like(self.valClustD)
    # cond[self.valClustD] =  (self.PcI[clust.toDrain[self.valClustD]]>clust.pcShrink[self.valClustD])
    # toDrain = clust.toDrain[cond]
    # clust.nextPcShrink[cond] = self.PcI[toDrain]
    # clustID = self.clusterNW_ID.copy()
    # cID = np.flatnonzero(cond)
    # clustID[toDrain] = cID
    # self.ha
    # mem = cond[clustID] & (clustID>=0)
    # tPhaseImb.__CondTPImbibition__(self, mem, nextPcShrink[clustID], updateArea=True, overrideTrapping=True)
    # vol = doClust.volume_bincount_numba(mem, clustID, self.areaNWPhase, self.areaSPhase, self.volarray)
    # clust.nextVolShrink[cID] = vol[cID]
    
    
    

def updatePcD(self):
    #global alpha
    PcD = np.minimum(self.PistonPcRec, self.PcD)
    #PcI = np.maximum(self.PistonPcAdv, self.PcI)
    #self.specialPcD[self.specialPcD==0.0] = self.PcI[self.specialPcD==0.0]
    #self.PcD[:] = (1-alpha)*self.specialPcD + alpha*PcD
    self.PcD[:] = PcD
    # print('Im in updatePcD!!')
    # from IPython import embed; embed()
    cc = (self.PcI==self.porebodyPc)&(self.fluid==1)
    self.PcI[cc] = self.PistonPcAdv[cc]
    #self.PcI[cc] = np.maximum(self.PcI[cc], self.PistonPcAdv[cc])
    
     
def initializingMolesConcElements(self):
    # moles of gas in elements filled with WPhase are all updated
    # but only members of NWPhase clusters not connected across s
    # the network are updated.
    
    ''' initialize the NW elements according to their trapped Pc!!! '''
    self.gasConc[self._elemToUpdateNW,0] = self.H*self.clusterNW.pc[
        self.clusterNW_ID[self._elemToUpdateNW]]
    
    ''' initialize the brine to have conc equivalent to Pc!!! '''
    #self.computeAqAvgPres()
    #self.aqAvgPres = self.Pc
    #self.aqAvgPres = 3000
    self.aqAvgPres = 0
    self.gasConc[self._elemToUpdateW,0] = self.H*self.aqAvgPres
    #self.assignConcToAqElements()
    # self.dissolvedMoles[self._elemToUpdateW] = (
    #     self.gasConc[self._elemToUpdateW,0]*self.volarray[self._elemToUpdateW])

    self.aqueousMoles[self._elemToUpdate] = (
        self.gasConc[self._elemToUpdate,0]*self.volarray[self._elemToUpdate]*
        self.satList[self._elemToUpdate])
    

def computeAqAvgPres(self):
    cond1 = (self.elemToUpdateNW[self.TPValid]&self.elemToUpdateW[self.TValid])
    cond2 = (self.elemToUpdateW[self.TPValid]&self.elemToUpdateNW[self.TValid])

    a=(self.len_tij_valid*self.clusterNW.pc[self.clusterNW_ID[self.TPValid]])[cond1].sum()
    b=(self.len_tij_valid*self.clusterNW.pc[self.clusterNW_ID[self.TValid]])[cond2].sum()
    c=self.len_tij_valid[cond1].sum()
    d=self.len_tij_valid[cond2].sum()
    self.aqAvgPres = ((a+b)/(c+d))

def findNextT(arrP, notdone, done, PTConnections, PTValid):
    arrT = PTConnections[arrP][PTValid[arrP]]
    arrT = arrT[notdone[arrT]]
    done[arrT] = True
    notdone[arrT] = False
    return arrT

def findNextP(arrT, notdone, done, TPConnections, TPCond):
    arrP = TPConnections[arrT][TPCond[arrT]]
    arrP = arrP[notdone[arrP]]
    done[arrP] = True
    notdone[arrP] = False
    return arrP
    
def findAqElements(self, arr, notdone):
    done = np.zeros(self.totElements, dtype=bool)
    arr = arr[notdone[arr]]
    done[arr] = True
    notdone[arr] = False

    arrP = arr[arr<=self.nPores]
    arrT = np.append(arr[arr>self.nPores], self.findNextT(arrP, notdone, done))
    while True:
        try:
            assert arrT.size>0
            arrP = self.findNextP(arrT-self.nPores, notdone, done)
            assert arrP.size>0
            arrT = self.findNextT(arrP, notdone, done)
        except AssertionError:
            return self.elementListS[done]
      

@staticmethod
def findNeighbouringClusters(
    c, totElements, nPores, clusterNW_mem, clusterNW_neigh, clusterID, 
    PTConnections, PTValid, TPConnections, TPCond, notdone):
       
    done = np.zeros(totElements, dtype=bool)
    foundClustID = []
    
    neigh = np.where(clusterNW_neigh[c])[0]
    done[neigh] = True
    notdone[neigh] = False
    notdone[clusterNW_mem[c]] = False

    nextT = findNextT(neigh[neigh<=nPores], notdone, done, PTConnections, PTValid)
    arrT = np.concatenate((neigh[neigh>nPores], nextT[clusterID[nextT]<0]))
    newClustID = list(set(clusterID[nextT][clusterID[nextT]>=0]))
    foundClustID.extend(newClustID)
    notdone[clusterNW_mem[newClustID].any(axis=0)] = False
    
    ij = 0
    while True:
        try:
            assert arrT.size>0
            arrP = findNextP(arrT-nPores, notdone, done, TPConnections, TPCond)
            newClustID = list(set(clusterID[arrP][clusterID[arrP]>=0]))
            foundClustID.extend(newClustID)
            notdone[clusterNW_mem[newClustID].any(axis=0)] = False
            arrP = arrP[clusterID[arrP]<0]
            assert arrP.size>0
            arrT = findNextT(arrP, notdone, done, PTConnections, PTValid)
            newClustID = list(set(clusterID[arrT][clusterID[arrT]>=0]))
            foundClustID.extend(newClustID)
            notdone[clusterNW_mem[newClustID].any(axis=0)] = False
            arrT = arrT[clusterID[arrT]<0]
            ij += 1
            assert ij!=3
        except AssertionError:
            #neighbouringClusters[c] = np.array(foundClustID, dtype=int)
            #return
            return np.array(foundClustID, dtype=int)



def initializeNeighbouringClusters(self):
    global memmap_shapes
    cluster_size = self.clusterNW.pc.size
    
    # Save arrays as memory-mapped files
    self.neighbouringClusters = np.zeros(cluster_size, dtype="object")
    initialize_memmaps(self)
    
    # Set multiprocessing to use 'forkserver' (faster & prevents memory duplication)
    multiprocessing.set_start_method("forkserver", force=True)

    update_memmap_shapes(self)

    args = (self.totElements, self.nPores, self.elemToUpdate, global_memmaps)

    results = Parallel(n_jobs=-1, batch_size="auto", prefer="processes")(
        delayed(findNeighbouringClusters_with_joblib)(c, *args) for c in range(cluster_size)
    )

    for res in results:
        self.neighbouringClusters[res[0]] = res[1]
       


def findNeighbouringClusters_with_joblib(c, totElements, nPores, elemToUpdate, global_memmaps):
    """Worker function that loads preloaded memory-mapped arrays."""
    
    # Use preloaded memmap files (shared memory, no copies)
    members = global_memmaps["members"]
    neighbours = global_memmaps["neighbours"]
    clusterNW_ID = global_memmaps["clusterNW_ID"]
    PTConnections = global_memmaps["PTConnections"]
    PTValid = global_memmaps["PTValid"]
    TPConnections = global_memmaps["TPConnections"]
    TPCond = global_memmaps["TPCond"]

    return c, findNeighbouringClusters(
        c, totElements, nPores, members, neighbours, 
        clusterNW_ID, PTConnections, PTValid, 
        TPConnections, TPCond, elemToUpdate)


def assignConcToAqElements(self):
    notdone = self.elemToUpdateW.copy()
    valClust = self.valClust.copy()
    while True:
        try:
            minPc = self.clusterNW.pc[valClust].min()
            clustID = np.where((self.clusterNW.pc==minPc)&valClust)[0]
            arr = self.elementListS[self.clusterNW.neighbours[clustID].any(axis=0)]
            arr = self.findAqElements(arr, notdone)
            self.gasConc[arr,0] = self.H*minPc
            valClust[clustID] = False
        except ValueError:
            break
    

def initializeFlowrate(self):
    ''' returns the velocities of fluid across each throat'''
    gwL = do.computegL(self, self.gWPhase)
    gnwL = do.computegL(self, self.gNWPhase)
    self.flowrateW, self.flowDirectionW = do.computeFlowrate(self, gwL, 0, self.Pc, True)
    self.flowrateNW, self.flowDirectionNW = do.computeFlowrate(self, gnwL, 1, self.Pc, True)
    self.flowrateNW_TValid = np.dstack((self.flowrateNW, self.flowrateNW))[0][self.TPCond[1:]]
    upstreamP1T = self.P1array.copy()
    upstreamP1T[~self.flowDirectionNW] = self.tList[~self.flowDirectionNW]
    upstreamP2T = self.tList.copy()
    upstreamP2T[~self.flowDirectionNW] = self.P2array[~self.flowDirectionNW]
    self.upstreamElement = np.dstack((upstreamP1T, upstreamP2T))[0][self.TPCond[1:]]
    

def updateLists(self, arr, fluid=0):
    cond = True if fluid==0 else False
    self.elemToUpdateW[arr] = cond
    self.elemToUpdateNW[arr] = not cond
    self._elemToUpdateW = self.elementListS[self.elemToUpdateW]
    self._elemToUpdateNW = self.elementListS[self.elemToUpdateNW]
    self.elemToUpdate = (self.elemToUpdateW|self.elemToUpdateNW)
    self._elemToUpdate = self.elementListS[self.elemToUpdate]
    # self.valClust = (self.clusterNW.size>0)
    # self.valClustD = self.valClust & (self.clusterNW.toDrain<=self.totElements)
    # self.toUpdateNW = np.array(self.clusterNW.keys)[self.valClust & (~self.clusterNW.connected)]


def computeFluxes(self, advection=False):
    ''' compute flux for each element and cluster '''

    #print('Im in computeFluxes!!!')
    # cond = (self.fluid==1)
    # cond[self.TValid[cond[self.TPValid]]] = True
    # cond[self.TPValid[cond[self.TValid]]] = True
    # len_tij_valid = np.where(cond[self.TPValid]|cond[self.TValid], self.len_tij_valid, 0.0)

    # len_tij_valid = np.where(
    #     (self.fluid[self.TPValid]==1)|(self.fluid[self.TValid]==1), self.len_tij_valid, 0.0)
    #print('Im in computeFluxes!!!')
    #from IPython import embed; embed()
    try:
        self.flux = self.D*self.len_tij_valid*(
        #self.flux = self.D*len_tij_valid*(
            self.gasConc[self.TPValid, 0] - self.gasConc[self.TValid, 0])
        assert not advection
    except AssertionError:
        self.flux += (self.flowrateNW_TValid*self.aqueousMoles[self.upstreamElement]/
                    self.volarray[self.upstreamElement]*self.satList[self.upstreamElement])
    if (np.abs(self.flux)>1e-2).any():
        print('there is an unusual flux herereer, check!!!!')
        from IPython import embed; embed()

    #print('Im in computeFluxes!!!')
    #from IPython import embed; embed()
    
    # compute the net flux for each element
    self.netFlux[self.tList] = np.bincount(self.tValid, self.flux, self.nThroats)
    self.netFlux[self.poreListS] = np.bincount(self.TPValid, -self.flux, self.nPores+2)

    # self.maxMoles[self.tList] = np.bincount(self.tValid, np.nan_to_num(
    #     self.flux/self.netFlux[self.TValid]*self.aqueousMoles[self.TValid]), self.nThroats)
    # self.maxMoles[self.poreListS] = np.bincount(self.TPValid, np.nan_to_num(
    #     -self.flux/self.netFlux[self.TPValid]*self.aqueousMoles[self.TPValid]), self.nPores+2)
    
    # compute the net flux for each cluster
    self.netFluxClusters = np.bincount(
        self.clusterNW_ID[self._elemToUpdateNW], self.netFlux[self._elemToUpdateNW],
        len(self.clusterNW.keys))
    # self.maxMolesClusters = np.bincount(
    #     self.clusterNW_ID[self._elemToUpdateNW], self.maxMoles[self._elemToUpdateNW], 
    #     len(self.clusterNW.keys))
    

def determineTimeStep1(self):
    ''' choose a dt such that no updated concentration is negative and 
        the direction of flow is not reversed '''
    #print('im in determineTimeStep!!!')
    #from IPython import embed; embed()
    concP = (self.aqueousMoles[self.TPValid]-self.flux*self._dt)/self.volarray[self.TPValid]
    concT = (self.aqueousMoles[self.TValid]+self.flux*self._dt)/self.volarray[self.TValid]
    #print('im in determineTimeStep!!!')
    #from IPython import embed; embed()

    # cond = ((self.flux<0.0)&(concP>concT)&(
    #     self.elemToUpdateW[self.TPValid]&self.elemToUpdateW[self.TValid]))
    # cond = ((self.flux!=0.0)&(concP>concT)&(
    #     self.elemToUpdateW[self.TPValid]&self.elemToUpdateW[self.TValid]))
    #gasConc = np.nan_to_num(self.aqueousMoles/self.volarray)
    #cond = (self.elemToUpdateW[self.TPValid]|self.elemToUpdateW[self.TValid])
    cond1 = (self.gasConc[self.TPValid,0]>self.gasConc[self.TValid,0])
    #cond1 = (gasConc[self.TPValid]>gasConc[self.TValid])
    cond2 = (concP<concT)
    #cond = cond&(cond1==cond2)&(self.flux!=0.0)
    cond = (cond1==cond2)&(self.flux!=0.0)
    #cond = cond&((self.fluid[self.TPValid]==1)|(self.fluid[self.TValid]==1))
    try:
        assert not cond.any()
        self.dt = self._dt
    except AssertionError:
        #ind = (concP[cond]/concT[cond]).argmax()
        #p, t = self.TPValid[cond][ind], self.TValid[cond][ind]
        p, t = self.TPValid[cond], self.TValid[cond]
        #num = (self.aqueousMoles[p]/self.volarray[p]-self.aqueousMoles[t]/self.volarray[t])
        num = (self.gasConc[p,0]-self.gasConc[t,0])
        #den = self.flux[cond][ind]*(1/self.volarray[p]+1/self.volarray[t])
        den = self.flux[cond]*(1/self.volarray[p]+1/self.volarray[t])
        self.dt = np.clip((num/den).min(), 1e-5, self._dt)

        if self.dt<0.0054:
            print(f'I am in dt:  {self.dt}', (num/den).argmin(), cond.sum())
            #input('wait!!!')
            #from IPython import embed; embed()


def determineTimeStep(self):
    ''' choose a dt such that no updated concentration is negative and
        the direction of flow is not reversed '''
    
    # time-step based on moles and flux
    cond = self.elemToUpdateW&(self.netFlux<0.0)
    condC = (self.netFluxClusters<0.0)
    self.dt = min([
        (-self.aqueousMoles[cond]/self.netFlux[cond]).min(initial=self._dt),
        (-self.clusterNW.moles[condC]/self.netFluxClusters[condC]).min(initial=self._dt)])
    
    if self.dt==0.0:
        print(f'self.dt = {self.dt}')
        from IPython import embed; embed()
        

def determinePc_Vol(self, moles, pcL, volL, pcU, volU, volT):
    fact = (pcU-pcL)/(volU-volL)
    Hfact = self.H*fact
    RT = self.R*self.T
    a = fact - (Hfact*RT)
    b = self.imposedP + pcL - (fact*volL) + (Hfact*volT - self.H*pcL + Hfact*volL)*RT
    c = -(moles - (volT*self.H*pcL) + (Hfact*volT*volL))*RT
    d = (b**2 -4*a*c)
    
    vol = np.zeros(moles.size)
    vol[a!=0] = (-b[a!=0]+np.sqrt(d[a!=0]))/(2*a[a!=0])
    vol[a==0] = -c[a==0]/b[a==0]
    pc = pcL + (vol-volL)*fact
    return pc, vol


def computeMolesPcConc(self):
    ''' compute and update the moles, pc and concentration of gas in each element and cluster'''
    #print('im in computeMolesPcConc!!!')
    determineTimeStep(self)
    clust = self.clusterNW

    ''' update element filled with WP properties '''
    try:
        netMoles = self.netFlux*self.dt
        netMoles = np.where(
            self.elemToUpdateW, np.clip(netMoles, -self.aqueousMoles, None), netMoles)
      
        netMolesClusters = np.bincount(
            self.clusterNW_ID[self._elemToUpdateNW], netMoles[self._elemToUpdateNW],
            len(self.clusterNW.keys))

        ''' update NWP cluster properties '''
        newClustMoles = np.clip(clust.moles+netMolesClusters, 0.0, None)
        # if (newClustMoles<0.0).any():
            # print(' a new clust moles is negative!!!')
            # from IPython import embed; embed()     
        keys = (netMolesClusters!=0.0)
        mem = keys[self.clusterNW_ID]&self.hasNWFluid
        updateClusterPcVolume(self, mem, keys, clust, self.clusterNW_ID, newClustMoles)
        clust.moles[keys] = newClustMoles[keys]
        
        if (self.valClust &
            (newClustMoles<=self.clusterNW.molesShrink)|(newClustMoles>=self.clusterNW.molesMax)).any():
            self.updateClust = True
        
        gasConcClusters = self.H*clust.pc
        self.gasConc[self.elemToUpdateNW,1] = gasConcClusters[
            self.clusterNW_ID[self.elemToUpdateNW]]

        self.aqueousMoles[self._elemToUpdateW] += netMoles[self._elemToUpdateW]
        self.gasConc[self._elemToUpdateW, 1] = np.nan_to_num(
            self.aqueousMoles[self._elemToUpdateW]/self.volarray[self._elemToUpdateW])
        
        if (self.aqueousMoles[self._elemToUpdateW]<0.0).any():
            print(' 1 or more aqueous moles are negative!!!')
            from IPython import embed; embed()
    except:
        print('encountered error in computeMolesPcConc!!!')
        from IPython import embed; embed()
    
    if (self.gasConc[:,1]<0.0).any():
        print('>> a new element  concentration is negative!!!')
        from IPython import embed; embed()    


def redistributeMoles(self, keys, newClustMoles):
    clust=self.clusterNW
    def _func1(k, idC):
        for c in idC:
            deltaMoles = np.clip(
                self.cumNetMolesClusters[k],0.0,np.abs(self.cumNetMolesClusters[c]))
            newClustMoles[[k,c]] += (deltaMoles, -deltaMoles)
            self.cumNetMolesClusters[[k,c]] += (-deltaMoles, deltaMoles)

    def _func2(k, neigh):
        try:
            cond = (self.clusterNW_ID[neigh]>=0)
            assert cond.any()
            idC = np.unique(self.clusterNW_ID[neigh][cond])
            _func1(k, idC)
            neigh = neigh[~cond]          
        except AssertionError:
            pass
        return neigh

    for k in keys:
        notdone = np.ones(self.totElements, bool)
        neigh = clust[k].neighbours
        conc = self.gasConc[clust[k].members[0],0]
        neigh = neigh[(self.gasConc[neigh,0]>conc)]
        try:
            neighP = neigh[neigh<=self.nPores]
            assert neighP.any()
            neighT = self.PTConnections[neighP]
            neighT = neighT[(neighT>0)&
                            (self.gasConc[neighT,0]>self.gasConc[neighP,0][:,np.newaxis])&
                            notdone[neighT]]
            neigh = np.append(neigh[neigh>self.nPores], neighT)        
        except AssertionError:
            pass
        notdone[neigh] = False 
        while neigh.size>0:
            neigh = _func2(k, neigh)
            if self.cumNetMolesClusters[k]==0.0: break
           
            try:
                assert neigh.any()
                neighP = self.TPConnections[neigh-self.nPores]
                neigh = neighP[(neighP>0)&
                               (self.gasConc[neighP,0]>self.gasConc[neigh,0][:,np.newaxis])&
                               notdone[neighP]]
                notdone[neigh] = False
                neigh = _func2(k, neigh)
                if self.cumNetMolesClusters[k]==0.0: break
            except AssertionError:
                break

            try:
                assert neigh.any()
                neighT = self.PTConnections[neigh]
                neigh = neighT[(neighT>0)&
                               (self.gasConc[neighT,0]>self.gasConc[neigh,0][:,np.newaxis])&
                               notdone[neighT]]
                notdone[neigh] = False
                neigh = _func2(k, neigh)
                if self.cumNetMolesClusters[k]==0.0: break
            except AssertionError:
                break

    return newClustMoles


def __cornerApex__(self, arr, arrr, halfAng, Pc, m_exists, apexDist, initedApexDist, 
                   fact, fact_part, sinBeta, sinBetacosBeta):
    
    ''' return the apexDist and dimensionless corner area '''
    newDist = fact[:,arrr]/Pc[arrr]
    apexDist[:,arrr] = np.maximum(newDist, initedApexDist[:, arrr])

    # update contact angle
    part = np.clip(Pc[:,np.newaxis]*fact_part, -0.999999, 0.999999)
    newAng = np.clip(np.arccos(part)-halfAng.T, 0.0, np.pi)[arrr]
    cond1 = (newDist>initedApexDist[:, arrr]).T
    conAng = np.ones(apexDist.shape)
    conAng.T[arrr] = np.where(cond1, self.thetaAdvAng[arr[arrr],np.newaxis], newAng)
    
    # update corner area
    thetaBeta = conAng+halfAng
    cond3 = (np.abs(thetaBeta-np.pi/2) < 0.01)
    dimlessCornerA = np.select([m_exists&cond3, m_exists&~cond3],
        [sinBetacosBeta, 
        pow((sinBeta/np.cos(thetaBeta)), 2.0)*(
            np.cos(conAng)*np.cos(thetaBeta)/sinBeta+thetaBeta-np.pi/2)],
        default=0.0)
    return apexDist, dimlessCornerA


def cornerApex(
        self, arr, arrr, halfAng, Pc, m_exists, apexDist, initedApexDist, shape, _return=False):
    lkup = self.lookup        
    if shape==3:
        fact, fact_part = lkup.fact_triangle, lkup.part_triangle
        sinBeta, sinBetacosBeta = lkup.sinBetaTr, lkup.sinBetacosBetaTr
    else:
        fact, fact_part = lkup.fact_square, lkup.part_square
        sinBeta, sinBetacosBeta = lkup.sinBetaSq, lkup.sinBetacosBetaSq
    
    apexDist, dimlessCornerA = __cornerApex__(
        self, arr, arrr, halfAng, Pc, m_exists, apexDist, initedApexDist, 
        fact, fact_part, sinBeta, sinBetacosBeta)
        
    cornerArea = (apexDist*apexDist*dimlessCornerA).sum(axis=0)
    _cond4 = arrr & self.hasNWFluid[arr] & self.isPolygon[arr]
    cond4 = arr[_cond4]
    self._areaWP[cond4] = np.where(cornerArea[_cond4]<self.areaSPhase[cond4], 
                                   cornerArea[_cond4], self.areaSPhase[cond4])
    self._areaNWP[cond4] = self.areaSPhase[cond4]-self._areaWP[cond4]

    if _return:
        return apexDist, dimlessCornerA


def fitStretchedExponential(self, arrr):
    ''' fit the stretched exponential function to the data '''
    clust = self.clusterNW
    pcstar = 0.0
    def returnPcstar(arr, _arrr, Pc, apexDist, dimlessCornerA):
        ''' return the min. capillary pressure at which phase 2 starts to occupy 
            the center of the pore/throat'''
        return np.sqrt(
            np.sum(pow((apexDist*Pc).T[_arrr],2)*dimlessCornerA.T[_arrr],axis=1)/
            self.areaSPhase[arr[_arrr]])

    def stretchedExponential(pc, a, k, beta, b):
        ''' return the stretched exponential function '''
        return np.where(pc<=pcstar, 1.0, a*np.exp(-k*np.power(pc-pcstar,beta))+b)
    
    n = 10
    arrrS = arrr[self.isSquare]
    arrrT = arrr[self.isTriangle]
    arrrC = arrr[self.isCircle]
    arrPc = np.empty(self.totElements)
    #pcStar = np.empty(self.totElements)
    pc_data = np.empty([n,self.totElements])
    sat_data = np.empty([n,self.totElements])
    gasAvgPres = self.clusterNW.pc[self.valClust].mean()
    maxPc = np.empty(self.totElements)
    
    if arrrT.any():
        i, _return = 0, True
        while True:
            try:
                assert _return
                _arr = self.elemTriangle[arrrT]
                arrPc[_arr] = 1.0
                apexDist = np.empty_like(self.hingAngTr.T)
                apexDist, dimlessCornerA = cornerApex(
                    self, self.elemTriangle, arrrT, self.halfAnglesTr.T, arrPc[self.elemTriangle], 
                    self.cornExistsTr.T, apexDist, self.initedApexDistTr.T, 3, _return)
                self.fittedParams[_arr,4] = returnPcstar(
                    self.elemTriangle, arrrT, arrPc[self.elemTriangle], apexDist, dimlessCornerA)
                pc_data[0, _arr] = arrPc[_arr]
                pc_data[1,_arr] = np.clip(self.fittedParams[_arr,4]-1,2,None).astype(int)
                maxPc[_arr] = np.clip(self.fittedParams[_arr,4]+gasAvgPres*2,
                                      clust.pc[self.clusterNW_ID[_arr]],None).astype(int)
                pc_data[2:,_arr] = np.array([
                    np.logspace(np.log10(pc_data[1,j]+2), np.log10(maxPc[j]), n-2) for j in _arr]).T
                _return = False
            except AssertionError:
                arrPc[_arr] = pc_data[i, _arr]
                cornerApex(
                    self, self.elemTriangle, arrrT, self.halfAnglesTr.T, arrPc[self.elemTriangle], 
                    self.cornExistsTr.T, apexDist, self.initedApexDistTr.T, 3)
            
            if (self.areaWPhase<0.0).any():
                print('areaWPhase is negative TTT, check!!!')
                from IPython import embed; embed()
            sat_data[i, _arr] = np.nan_to_num(self.areaWPhase[_arr]/self.areaSPhase[_arr])
            i+=1
            if i==n: break
    
    if arrrS.any():
        i, _return = 0, True
        while True:
            try:
                assert _return
                _arr = self.elemSquare[arrrS]
                arrPc[_arr] = 1.0
                apexDist = np.empty_like(self.hingAngSq.T)
                apexDist, dimlessCornerA = cornerApex(
                    self, self.elemSquare, arrrS, self.halfAnglesSq[:,np.newaxis], 
                    arrPc[self.elemSquare], self.cornExistsSq.T, apexDist, self.initedApexDistSq.T, 4, _return)
                self.fittedParams[_arr,4] = returnPcstar(
                    self.elemSquare, arrrS, arrPc[self.elemSquare], apexDist, dimlessCornerA)
                pc_data[0, _arr] = arrPc[_arr]
                pc_data[1,_arr] = np.clip(self.fittedParams[_arr,4]-1,2,None).astype(int)
                maxPc[_arr] = np.clip(self.fittedParams[_arr,4]+gasAvgPres*2,
                                      clust.pc[self.clusterNW_ID[_arr]],None).astype(int)
                pc_data[2:,_arr] = np.array([
                    np.logspace(np.log10(pc_data[1,j]+2), np.log10(maxPc[j]), n-2) for j in _arr]).T
                _return = False
            except AssertionError:
                arrPc[_arr] = pc_data[i, _arr]
                cornerApex(
                    self, self.elemSquare, arrrS, self.halfAnglesSq[:,np.newaxis],
                    arrPc[self.elemSquare], self.cornExistsSq.T, apexDist, self.initedApexDistSq.T, 4)
    
            if (self.areaWPhase<0.0).any():
                print('areaWPhase is negative SSSS, check!!!')
                from IPython import embed; embed()
            sat_data[i, _arr] = np.nan_to_num(self.areaWPhase[_arr]/self.areaSPhase[_arr])
            i+=1
            if i==n: break
            
    # Fit the model
    _arr = self.elementListS[arrr&self.isPolygon]
    if _arr.any():
        m = n//2
        for j in _arr:
            pcstar=self.fittedParams[j,4]
            a,beta,b = sat_data[2,j]-sat_data[-1,j],0.5,sat_data[-1,j]
            try:
                assert a>9.9999e-9
                k = -np.log((sat_data[m,j]-b)/a)/(pc_data[m,j]-pcstar)**beta
                popt, pcov = curve_fit(stretchedExponential, pc_data[:,j], sat_data[:,j], 
                                    p0=[a, k, beta, b])
                popt = np.append(popt, pcstar)
            except:
                popt = np.array([sat_data[2,j], 0.0, 0.0, 0.0, pcstar])
            self.fittedParams[j] = popt

    if arrrC.any():
        _arr = self.elemCircle[arrrC]
        self.fittedParams[_arr] = np.array([1.0, 0.0, 0.0, 0.0, 0.0])


def updateClusterPcVolume(self, mem, keys, clust, clustID, targetMoles, 
    event=False, moles_tol=1e-23, pc_tol=1e-4):
    ''' update the cluster pc and volume according to the moles in each cluster '''
    st = time()
    lkup = self.lookup    
    if not event:
        keys = np.flatnonzero(keys)
        term1 = clust.volume[keys]/lkup.RT
        term2 = (clust.pc[keys]+self.imposedP)/lkup.RT*clust.dVg_dPc[keys]
        term3 = clust.pc[keys]*clust.dVg_dPc[keys]
        dn_dPc = term1 + term2 + self.H*(clust.totalVolume[keys] - clust.volume[keys] - term3)
        lkup.guessPc[keys] = np.clip(
            clust.pc[keys]+(targetMoles[keys]-clust.moles[keys])/dn_dPc, 1e-3, None)
        tPhaseImb.__CondTPImbibition__(self, mem, lkup.guessPc[clustID], True, True)
        vol = doClust.volume_bincount_numba(mem, clustID, self.areaNWPhase, self.areaSPhase, self.volarray)
        self.satList[mem] = self.areaWPhase[mem]/self.areaSPhase[mem]
        clust.pc[keys], clust.volume[keys] = lkup.guessPc[keys], vol[keys]
    elif keys.any():        
        nC = clust.pc.size
        if lkup.guessPc.size!=nC:
            lkup = self.lookup = LookupClass(self)
            
        guessPc = lkup.guessPc
        guessPc1, guessPc2 = lkup.guessPc1, lkup.guessPc2
        guessMoles = lkup.guessMoles
        guessMoles1 = lkup.guessMoles1
        guessVol, resMoles = lkup.guessVol, np.zeros_like(guessMoles)
        
        cond1 = keys.copy()
        arr = np.flatnonzero(cond1)
        guessPc[arr] = clust.pc[arr]
        mem = cond1[clustID] & self.hasNWFluid
        tPhaseImb.__CondTPImbibition__(self, mem, guessPc[clustID], True, True)
        try:
            resMoles[arr], guessMoles[arr], guessVol[arr] = doClust.return_moles_vol_numba(
                mem, cond1, clustID, self.areaNWPhase, self.areaSPhase,
                self.volarray, clust.totalVolume, targetMoles, guessPc, lkup.RT, self.H, self.imposedP)
        except:
            print('there is an error on line 979 updateClusterPcVolume!!!')
            from IPython import embed; embed()
        
        cond = cond1&(np.abs(resMoles)>moles_tol)
        molesGT = resMoles > 0.0
        conda, condb = cond & molesGT, cond & ~molesGT
        guessPc1[conda] = self.Pc
        guessPc1[condb] = clust.pc.max()
        arr = np.flatnonzero(cond)
        mem = cond[clustID] & self.hasNWFluid
        tPhaseImb.__CondTPImbibition__(self, mem, guessPc1[clustID], True, True)
        resMoles1, guessMoles1, guessVol1 = doClust.return_moles_vol_numba(
            mem, cond, clustID, self.areaNWPhase, self.areaSPhase,
            self.volarray, clust.totalVolume, targetMoles, guessPc1, lkup.RT, self.H, self.imposedP)
            
        done = np.abs(resMoles1)<=moles_tol
        if any(done):
            _done = arr[done]
            guessPc[_done] = guessPc1[_done]
            guessMoles[_done] = guessMoles1[done]
            guessVol[_done] = guessVol1[done]
            cond[_done] = False
            
        # Update guess values where guessPc2 is a better approximation
        closer = (np.abs(resMoles1)<np.abs(resMoles[arr]))
        if closer.any():
            _closer = arr[closer]
            guessPc[_closer], guessPc1[_closer] = guessPc1[_closer], guessPc[_closer]
            guessMoles[_closer] = guessMoles1[closer]
            guessVol[_closer] = guessVol1[closer]
            resMoles[_closer] = resMoles1[closer]
        
        i=0
        cond = cond & (np.abs(resMoles)>moles_tol)
        # you may have to swap the values of guessPc for guessPc1 if its a better guess 
        while cond.any():
            arr = np.flatnonzero(cond)
            guessPc2[arr] = 0.5*(guessPc[arr]+guessPc1[arr])
            mem = cond[clustID] & self.hasNWFluid
            tPhaseImb.__CondTPImbibition__(self, mem, guessPc2[clustID], True, True)
            resMoles2, guessMoles2, guessVol2 = doClust.return_moles_vol_numba(
                mem, cond, clustID, self.areaNWPhase, self.areaSPhase,
                self.volarray, clust.totalVolume, targetMoles, guessPc2, lkup.RT, self.H, self.imposedP)
            
            # Update guess values where guessPc2 is a better approximation
            abs_resMoles = np.abs(resMoles[arr])
            closer = np.abs(resMoles2)< abs_resMoles
            if closer.any():
                _closer = arr[closer]
                guessPc[_closer] = guessPc2[_closer]
                guessMoles[_closer] = guessMoles2[closer]
                guessVol[_closer] = guessVol2[closer]
                resMoles[_closer] = resMoles2[closer]
            
            # Update guessPc1 where guessPc2 is a worse approximation
            guessPc1[arr[~closer]] = guessPc2[arr[~closer]]
            
            cond[arr] = (abs_resMoles>moles_tol) & (np.abs(guessPc[arr]-guessPc1[arr])>pc_tol)
            i += 1
            if i > 50:
                print("Not converged after 50 iterations.")
                break
                
        mem = np.flatnonzero(cond1[clustID]&self.hasNWFluid)
        self.satList[mem] = self.areaWPhase[mem]/self.areaSPhase[mem]
        if (guessPc[cond1]<0.0).any():
            print('a new pc is negative!!!')
            from IPython import embed; embed()
        clust.pc[cond1], clust.volume[cond1] = guessPc[cond1], guessVol[cond1]
        #print('im done with updateClusterPcVolume!!!')
    

           
def updateClusterPcVolume1(self, mem, keys, clust, clustID, targetMoles):
    ''' update the cluster pc and volume according to the moles in each cluster '''
    st = time()
    cond = mem&(self.fittedParams[:,0]==0.0)
    if cond.any():
        fitStretchedExponential(self, cond)

    # identify clusters that need adjustment
    a,k,beta,b,pcstar = (self.fittedParams[:,0], self.fittedParams[:,1], 
                         self.fittedParams[:,2], self.fittedParams[:,3], 
                         self.fittedParams[:,4])
    lkup = self.lookup
    term1, term2, term3 = self.H*clust.totalVolume, lkup._impP_RT, lkup._RT_H
    
    def returnVol(arrr,arrkeys,arrrPc):
        _deltaPc = arrrPc[arrr]-pcstar[arrr]
        sat = np.where(
            _deltaPc<=0.0, 1.0, a[arrr]*np.exp(-k[arrr]*np.power(_deltaPc,beta[arrr]))+b[arrr])
        vol = np.bincount(clustID[arrr],(1-sat)*self.volarray[arrr])
        return vol[arrkeys[:vol.size]], sat
    
    def _func(arrr,arrkeys,arrPc):
        vol, sat = returnVol(arrr, arrkeys, arrPc[clustID])
        num = targetMoles[arrkeys]/arrPc[arrkeys] - term1[arrkeys]
        den = term2/arrPc[arrkeys] + term3
        return vol-num/den, vol, sat
    
    def returnVol1(arrr, arrkeys, arrPc):
        clust.computeCornerArea(arrr, arrPc[clustID], self)
        vol = np.bincount(clustID[arrr],
            self.areaNWPhase[arrr]/self.areaSPhase[arrr]*self.volarray[arrr])
        return vol[arrkeys[:vol.size]]
    
    def _func1(arrr,arrkeys,arrPc):
        vol = returnVol1(arrr, arrkeys, arrPc[clustID])
        num = targetMoles[arrkeys]/arrPc[arrkeys] - term1[arrkeys]
        den = term2/arrPc[arrkeys] + term3
        return vol-num/den, vol

    
    nC = clust.pc.size
    guessPc, guessPc1, guessPc2 = lkup.guessPc, lkup.guessPc1, lkup.guessPc2
    
    # identify clusters that need adjustment and define guess 1
    guessPc1[keys] = clust.pc[keys]
    #resVol, _, _ = _func(mem, keys, guessPc1)
    resVol, _ = _func1(mem, keys, guessPc1)
    cond, vol1GT, vol2GT = keys.copy(), np.empty(nC,bool), np.empty(nC,bool)
    vol1GT[cond] = (resVol>0.0)
    cond[cond] = (np.abs(resVol)>1e-23)
    
    # guess 2
    cond1, cond1a, cond1b = cond.copy(), cond&vol1GT, cond&~vol1GT
    i, fact = 0, 1.001
    print(':::>>>>>>>>>>>>')
    from IPython import embed; embed()
    while True:
        mem = cond[clustID]
        guessPc2[cond1a] = guessPc1[cond1a]*fact
        guessPc2[cond1b] = guessPc1[cond1b]/fact
        #resVol, _, _ = _func(mem, cond, guessPc2)
        resVol, _ = _func1(mem, cond, guessPc2)
        vol2GT[cond] = (resVol>0.0)
        try:
            cond[cond] = (vol1GT[cond]==vol2GT[cond])
            assert cond.any()
            guessPc1[cond] = guessPc2[cond]
            cond1a, cond1b = cond1a&cond, cond1b&cond
        except AssertionError:
            cond1b[cond1b] = False
            break
        i+=1
        fact *= 1.001
        print(i, cond1a.sum(), cond1b.sum(), cond.sum())
    
    print(':::>>>>>>>>>>>>')
    from IPython import embed; embed()
    i=0
    volGT, cond1c = np.empty(nC,bool), np.empty(nC,bool)
    while cond1.any():
        mem = cond1[clustID]
        guessPc[cond1] = (guessPc1[cond1]+guessPc2[cond1])/2.0
        resVol, vol, sat = _func(mem, cond1, guessPc)
        volGT[cond1] = (resVol>0.0)
        
        # identify those already converged
        cond1a[cond1a] = False
        cond1a[cond1] = ((np.abs(resVol)<1e-23)|(np.abs(guessPc1[cond1]-guessPc2[cond1])<1e-5))
        try:
            assert cond1a.any()
            clust.pc[cond1a] = guessPc[cond1a]
            self.satList[cond1a[clustID]] = sat[cond1a[clustID[mem]]]
            clust.volume[cond1a] = vol[cond1a[cond1]]
            cond1[cond1a] = False
        except AssertionError:
            pass
        
        # update guessPc1, guessPc2
        cond1b[cond1],cond1c[cond1] = ((volGT[cond1]==vol1GT[cond1]),(volGT[cond1]==vol2GT[cond1]))
        guessPc1[cond1b], guessPc2[cond1c] = guessPc[cond1b], guessPc[cond1c]
        i+=1

        if i>50:
            print('not converged')
            from IPython import embed; embed()

        if (np.abs(guessPc1[cond1]-guessPc2[cond1])<1).any():
            print('diff is 1 or less!!!!')
            from IPython import embed; embed()


def computeMolesPcConc1(self):
    ''' compute and update the moles, pc and concentration of gas in each element and cluster'''
    #print('Im in computeMolesPcConc!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    determineTimeStep(self)
    #self.dt=self._dt
    clust = self.clusterNW

    #print('&&&^^^%%$$$$£££!!!!!!!!!!!!!!')
    #from IPython import embed; embed()

    ''' update element filled with WP properties '''
    netMoles = np.clip(self.netFlux*self.dt, -self.aqueousMoles, None)
    try:
        assert abs(netMoles.sum())>1e-25
        condA, condB = (netMoles>0.0), (netMoles<0.0)
        sumA, sumB = netMoles[condA].sum(), abs(netMoles[condB].sum())
        netMoles[condA] = sumB/sumA*netMoles[condA]
    except AssertionError:
        pass
    netMolesClusters = np.bincount(
        self.clusterNW_ID[self._elemToUpdateNW], netMoles[self._elemToUpdateNW],
        len(self.clusterNW.keys))
        
    self.aqueousMoles[self._elemToUpdateW] += netMoles[self._elemToUpdateW]
    self.gasConc[self._elemToUpdateW, 1] = np.nan_to_num(
        self.aqueousMoles[self._elemToUpdateW]/self.volarray[self._elemToUpdateW])

    ''' update NWP cluster properties '''
    try:
        newClustMoles = (clust.moles+netMolesClusters)
        cond = (netMolesClusters[self.toUpdateNW]!=0.0)
        cond3 = cond & (self.initialVolume[self.toUpdateNW]==clust.volMax[self.toUpdateNW])
        cond1 = cond & ~cond3
        cond1a = cond1 & (newClustMoles[self.toUpdateNW]<self.initialMoles[self.toUpdateNW])
        cond1b = cond1 & (newClustMoles[self.toUpdateNW]>=self.initialMoles[self.toUpdateNW])
        toUpdateNW1 = self.toUpdateNW[cond1]

        pcL = (clust.pcShrink[self.toUpdateNW]*cond1a)+(self.initialPc[self.toUpdateNW]*cond1b)
        pcU = (self.initialPc[self.toUpdateNW]*cond1a)+(clust.pcMax[self.toUpdateNW]*cond1b)
        volL = (clust.volShrink[self.toUpdateNW]*cond1a)+(
            self.initialVolume[self.toUpdateNW]*cond1b)
        volU = (self.initialVolume[self.toUpdateNW]*cond1a)+(clust.volMax[self.toUpdateNW]*cond1b)

        newPc, newVol = determinePc_Vol(
            self, newClustMoles[toUpdateNW1], pcL[cond1], volL[cond1], pcU[cond1], volU[cond1], clust.totalVolume[toUpdateNW1])
        # update pc and vol only when the values are reasonable
        cond = (newPc<self.maxPc)&~np.isnan(newPc)
        toUpdateNW1 = toUpdateNW1[cond]
        clust.pc[toUpdateNW1] = newPc[cond]  
        clust.volume[toUpdateNW1] = np.clip(newVol[cond], 0.0, clust.totalVolume[toUpdateNW1])
      
        toUpdateNW3 = self.toUpdateNW[cond3]
        clust.pc[toUpdateNW3] = (
            newClustMoles[toUpdateNW3]*self.R*self.T - self.imposedP*clust.volume[toUpdateNW3])/(self.H*(clust.totalVolume[toUpdateNW3]-clust.volume[toUpdateNW3])*self.R*self.T + clust.volume[toUpdateNW3])
        
        clust.moles[:] = newClustMoles

        if (self.valClust &
            (newClustMoles<=self.clusterNW.molesShrink)|
            (newClustMoles>=self.clusterNW.molesMax)).any():
            self.updateClust = True
        
        assert np.any((clust.pc>self.maxPc) | np.isnan(clust.pc))
        print('there is an unrealistic pc, check line 427!!!')
        breakpoint()
    except AssertionError:
        pass
    
    # update conc and moles but check if the cluster gas becomes completely dissolved
    gasConcClusters = self.H*clust.pc
    elemToUpdateNW = self.elemToUpdateNW
    self.gasConc[elemToUpdateNW,1] = gasConcClusters[self.clusterNW_ID[elemToUpdateNW]]


def updateSat1(self):
    try:
        vol = ((1-self.satList[self.elemToUpdateNW])*self.volarray[self.elemToUpdateNW])
        clustVol = np.bincount(self.clusterNW_ID[self.elemToUpdateNW], vol,
                                self.clusterNW.volume.size)
        cond = ~np.isclose(self.clusterNW.volume, clustVol, atol=1e-25)
        assert cond.any()
        mem = self._elemToUpdateNW[cond[self.clusterNW_ID[self._elemToUpdateNW]]]
        memkeys = self.clusterNW_ID[mem]
        _centerArea = np.clip(self.areaSPhase[mem]/self.clusterNW.totalVolume[memkeys]*
                        self.clusterNW.volume[memkeys], 0.0, self.areaSPhase[mem])
        _cornArea = self._cornArea.copy()
        _cornArea[mem] = self.areaSPhase[mem] - _centerArea
        self.satList[mem] = (_cornArea[mem]/self.areaSPhase[mem])
    except AssertionError:
        pass
    
    if (self.satList[self.elemToUpdateW]!=1.0).any():
        print(':::@@@>>><<<<')
        from IPython import embed; embed()
    self.satW = (self.satList[self.isinsideBox]*self.volarray[
        self.isinsideBox]).sum()/self.totVoidVolume


def updateSat(self):
    try:
        vol = ((1-self.satList[self.elemToUpdateNW])*self.volarray[self.elemToUpdateNW])
        clustVol = np.bincount(self.clusterNW_ID[self.elemToUpdateNW], vol,
                                self.clusterNW.volume.size)
        cond = ~np.isclose(self.clusterNW.volume, clustVol, atol=1e-25)
        assert cond.any()
        mem = self._elemToUpdateNW[cond[self.clusterNW_ID[self._elemToUpdateNW]]]
        memkeys = self.clusterNW_ID[mem]
        
        _centerArea = np.clip(self.areaSPhase[mem]/self.clusterNW.totalVolume[memkeys]*
                        self.clusterNW.volume[memkeys], 0.0, self.areaSPhase[mem])
        _cornArea = self._cornArea.copy()
        _cornArea[mem] = self.areaSPhase[mem] - _centerArea
        self.satList[mem] = (_cornArea[mem]/self.areaSPhase[mem])
    except AssertionError:
        pass
    
    # if (self.satList[self.elemToUpdateW]!=1.0).any():
    #     print(':::@@@>>><<<<')
    #     from IPython import embed; embed()
    self.satW = (self.satList[self.isinsideBox]*self.volarray[
        self.isinsideBox]).sum()/self.totVoidVolume




def simulateOstRip(self, implicit=True, freshStart=True):
    duration = 3600*24  # 24-hours
    #duration = 3600
    #duration = 600
    #from IPython import embed; embed()
    #duration = 3600*144
    #duration = 3600*24*20
    #duration = 1800
    
    st = time()
    
    if not os.path.isfile('volarray_bent.dat'):
        np.savetxt('volarray_bent.dat', self.volarray)
    if not os.path.isfile('coordinates_bent.dat'):
        np.savetxt('coordinates_bent.dat', 
                    np.column_stack((self.x_array, self.y_array, self.z_array)))
        
    # records data in at most 10-mins intervals if no event occur
    toShrink_elements = np.loadtxt('bent_toShrink_elements.dat', delimiter=',')
    minTime = 600
    try:
        assert freshStart
        j , ii, totTime, totalTime = 1, 1, 0.0, 0.0
        writeData(self, totalTime)
        self.updateClust = False
        self.clusterNW.drainEvents, self.clusterNW.imbEvents = 0, 0
        totalMoles = (
            (self.volarray*self.gasConc[:,0])[~self.elemToUpdateNW].sum()+
            self.clusterNW.moles.sum())
        #updateSat(self)
        gasAvgPres = (
                (self.clusterNW.pc[self.valClust]*self.clusterNW.totalVolume[self.valClust]).sum()/
                self.clusterNW.totalVolume[self.valClust].sum())
        print(("#:%8.6g  \tSimulation Time:%12.6g  \tActual runtime:%6.6g\
                \tTotal cluster moles:%8.6e  \tTotal dissolved moles:%6.6e\
                \tTotal gas moles:%6.10e \tAvg Pressure:%8.6e \tNo Shrinkage:%8.6g\
                \tNo Growth:%8.6g \tSat:%6.6g \tAqAvgPres:%6.6g" % (
            ii, round(totalTime,3), round(time()-st,3), self.clusterNW.moles.sum(), 
            #self.dissolvedMoles.sum(), totalMoles,
            self.aqueousMoles.sum(), totalMoles,
            #self.clusterNW.pc[self.valClust].mean(),
            gasAvgPres,
            self.clusterNW.imbEvents, self.clusterNW.drainEvents, self.satW, self.aqAvgPres)))
        writeData(self, round(totalTime,3))
        
    except AssertionError:
        #with open(os.path.join(MEMORY_DIR, f"netsim_time_Dependent_2_hours.pkl"), "rb") as f:
        with open(os.path.join(MEMORY_DIR, f"netsim_time_Dependent_10505_seconds.pkl"), "rb") as f:
            loaded_obj = dill.load(f)
        updateObj(self, loaded_obj)
        j, ii, totTime, totalTime = self.j, self.ii, self.totTime, self.totalTime
        print(j, ii, totTime, totalTime)
        #from IPython import embed; embed()
    
    def _fff(ii, totTime, totalTime):
        #cond2 = (self.clusterNW.moles>=self.clusterNW.molesMax)
        #cond1 = ~cond2&(self.clusterNW.moles<=self.clusterNW.molesShrink)&self.valClust
        cond1 = (self.clusterNW.moles<=self.clusterNW.molesShrink)&self.valClust
        cond2 = ~cond1 & (self.clusterNW.moles>=self.clusterNW.molesMax)
        if not cond1[self.valClust].any() and not cond2[self.valClustD].any():
            computeFluxes(self)
            computeMolesPcConc(self)
            totTime += self.dt
        else:
            self.dt = 0.0
            self.updateClust = True
            self.gasConc[self.elemToUpdate,1] = self.gasConc[self.elemToUpdate,0]

        if self.updateClust or (totTime>minTime):     #check for every second simulation
            totalTime += totTime
            totTime = 0.0

            try:
                condShrink = cond1 #& self.valClust
            except ValueError:
                # a resizing had occured after the creation of cond1!!!
                condShrink = self.valClust&(self.clusterNW.moles<=self.clusterNW.molesShrink)
            
            try:
                if condShrink.any():
                    keys = np.array(self.clusterNW.keys)[condShrink]
                    toImbibe = self.clusterNW.toImbibe[keys]
                    print('@@:   ', keys, toImbibe, toImbibe in toShrink_elements)
                    # if ii==97:
                        # from IPython import embed; embed()
                    self.clusterNW.shrinkCluster(keys, self)
                    self.clusterNW.imbEvents += condShrink.sum()
                    updateLists(self, toImbibe)
                    self.updateClust = False
            except:
                print('there is an error in the simulation, check line 1390!!!')
                from IPython import embed; embed()

            self.valClustD = self.valClust & (self.clusterNW.toDrain<=self.totElements)
            try:
                condGrowth = cond2 & self.valClustD
            except ValueError:
                # an array had occured after the creation of cond1!!!
                condGrowth = self.valClust&(self.clusterNW.moles>=self.clusterNW.molesMax)
                
            try:
                if condGrowth.any():
                    keys = np.array(self.clusterNW.keys)[condGrowth]
                    toDrain = self.clusterNW.toDrain[keys]
                    print('$$:  ', keys, toDrain)
                    toDrain = self.clusterNW.growCluster(keys, self)
                    self.clusterNW.drainEvents += condGrowth.sum()
                    updateLists(self, toDrain, 1)
                    self.updateClust = False
            except:
                print('there is an error in the simulation, check line 1397!!!')
                from IPython import embed; embed()

            totalMoles = self.aqueousMoles[self._elemToUpdateW].sum()+self.clusterNW.moles.sum()
            if not condShrink.any() and not condGrowth.any():
                mem = self.hasNWFluid.copy()
                tPhaseImb.__CondTPImbibition__(self, mem, self.clusterNW.pc[self.clusterNW_ID], True, True)
                #self.clusterNW.computeCornerArea(mem, self.clusterNW.pc[self.clusterNW_ID], self)
                self.satList[mem] = self.areaWPhase[mem]/self.areaSPhase[mem]
            self.satW = ((self.satList[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()/
                         self.totVoidVolume)

            self.aqAvgPres = (
                self.aqueousMoles[self.elemToUpdateW].sum()/
                (self.H*self.volarray[self.elemToUpdateW].sum()))
            gasAvgPres = (
                (self.clusterNW.pc[self.valClust]*self.clusterNW.totalVolume[self.valClust]).sum()/
                self.clusterNW.totalVolume[self.valClust].sum())
            
            print(("#:%8.6g  \tSimulation Time:%12.6g  \tActual runtime:%6.6g\
                \tTotal cluster moles:%8.6e  \tTotal dissolved moles:%6.6e\
                \tTotal gas moles:%4.10e \tAvg Pressure:%8.6e \tNo Shrinkage:%8.6g\
                \tNo Growth:%8.6g \tSat:%6.6g \tAqAvgPres:%6.6g" % (
                ii, round(totalTime,3), round(time()-st,3), self.clusterNW.moles.sum(), 
                #self.dissolvedMoles.sum(), totalMoles,
                self.aqueousMoles.sum(), totalMoles,
                #self.clusterNW.pc[self.valClust].mean(),
                gasAvgPres,
                self.clusterNW.imbEvents, self.clusterNW.drainEvents, self.satW, self.aqAvgPres)))
                
            writeData(self, round(totalTime,3))
    
            ii += 1
                    
        self.gasConc[self.elemToUpdate,0] = self.gasConc[self.elemToUpdate,1]
        self.aqueousMoles[self._elemToUpdateNW] = (
            self.gasConc[self._elemToUpdateNW,1]*self.volarray[self._elemToUpdateNW]*
            self.satList[self._elemToUpdateNW])
            
            
        if (self.gasConc[:,1]<0.0).any():
            print('## a new element  concentration is negative!!!')
            from IPython import embed; embed()    
            
        return ii, totTime, totalTime
        

    # print('&&&^^^%%$$$$£££!!!!!!!!!!!!!!')
    # from IPython import embed; embed()
    #duration = 158.68
    timeToSave = 3600*j
    while totalTime < duration:
        try:
            ii, totTime, totalTime = _fff(ii, totTime, totalTime)
            if totalTime > timeToSave:
                try:
                    self.j, self.ii = j+1, ii
                    self.totTime, self.totalTime = totTime, totalTime
                    with open(os.path.join(MEMORY_DIR, f"netsim_time_Dependent_{j}_hours.pkl"),
                            "wb") as f:
                        dill.dump(self, f)
                    j += 1
                    timeToSave = 3600*j
                except:
                    print('there is an error in pickling!!!')
                    from IPython import embed; embed()
            
            # if ii==1997:
                # from IPython import embed; embed()
            #print(self.dt, totTime, totalTime)
            # if self.clusterNW[0].members.any():
            #     print(self.clusterNW[0].members.size, self.clusterNW.pc[0], self.clusterNW.moles[0], self.clusterNW.volume[0])
            # if self.clusterNW.imbEvents>733:
            #     print('imbEvents > 733!!!')
            #     from IPython import embed; embed()
        except:
            print('there is an error in the simulation, check line 1438!!!')
            from IPython import embed; embed()
            #breakpoint()
            #from IPython import embed; embed()
            #raise AssertionError('there is an error in the simulation, check line 603!!!')

    print('::::::::::::::::::::::::::::')
    from IPython import embed; embed()
    return self
    
  
def writeData(self, totalTime):
    with open(os.path.join(MEMORY_DIR, 'clustPcOstRipening_bent.dat'), 'a') as f1,\
            open(os.path.join(MEMORY_DIR, 'clustVolOstRipening_bent.dat'), 'a') as f2,\
            open(os.path.join(MEMORY_DIR, 'clustMolesOstRipening_bent.dat'), 'a') as f3,\
            open(os.path.join(MEMORY_DIR, 'clustIDOstRipening_bent.dat'), 'a') as f4,\
            open(os.path.join(MEMORY_DIR, 'dissolvedMolesOstRipening_bent.dat'), 'a') as f5,\
            open(os.path.join(MEMORY_DIR, 'timeArrayOstRipening_bent.dat'), 'a') as f6,\
            open(os.path.join(MEMORY_DIR, 'numberEventsOstRipening_bent.dat'), 'a') as f7,\
            open(os.path.join(MEMORY_DIR, 'saturationOstRipening_bent.dat'), 'a') as f8:
        np.savetxt(f1, [self.clusterNW.pc], delimiter=',', fmt='%g')
        np.savetxt(f2, [self.clusterNW.volume], delimiter=',', fmt='%g')
        np.savetxt(f3, [self.clusterNW.moles], delimiter=',', fmt='%g')
        np.savetxt(f4, [self.clusterNW_ID], delimiter=',', fmt='%g')
        np.savetxt(f5, [self.aqueousMoles], delimiter=',', fmt='%g')
        f6.write(str(totalTime)+',')
        np.savetxt(f7, [[self.clusterNW.imbEvents, self.clusterNW.drainEvents]], 
                    delimiter=',', fmt='%g')
        f8.write(str(self.satW)+',')
        

def recomputeClusterVolume(self):
    self.elementPc_area = self.clusterNW.pc[self.clusterNW_ID]
    self.elementPc_area[self.clusterNW_ID<0] = 0.0
    with open("clustVolOstRipening_bent_new_actual.dat", 'a') as f:
        clustPcData = np.loadtxt('clustPcOstRipening_bent_new.dat', delimiter=',', dtype=np.float64)
        clustIDData = np.loadtxt('clustIDOstRipening_bent_new.dat', delimiter=',', dtype=np.int64)
        previousPc = self.clusterNW.pc[self.clusterNW_ID]
        for i, (clustID, clustPc) in enumerate(zip(clustIDData, clustPcData)):
            try:
                elemPc = clustPc[clustID]
                elemPc[clustID<0] = 0.0
                mem = (clustID>= 0) & (elemPc != previousPc)
                self.clusterNW.volume[clustPc==0.0] = 0.0
                assert np.any(mem)
                keys = np.unique(clustID[mem])
                print(i, keys)
                _ = self.clusterNW.computeVolume(keys, mem, elemPc, clustID, clustID, self)
                np.savetxt(f, [self.clusterNW.volume], delimiter=',', fmt='%g')
                previousPc[mem] = elemPc[mem]
            except AssertionError:
                np.savetxt(f, [self.clusterNW.volume], delimiter=',', fmt='%g')
            except:
                print('there is another error here')
                from IPython import embed; embed()


def updateObj(self, obj):
    selfDict = self.__dict__
    selfKeys = selfDict.keys()
    objDict = obj.__dict__
    for key in objDict.keys():
        if key not in selfKeys:
            setattr(self, key, objDict[key])
            continue
        try:
            old_val = getattr(self, key)
            new_val = getattr(obj, key)
            if not (old_val==new_val).all():
                assert isinstance(old_val, np.ndarray)
                assert isinstance(new_val, np.ndarray)
                old_base = old_val.base
                new_base = new_val.base
                if (old_base is not None and new_base is not None and
                    isinstance(old_base, np.ndarray) and isinstance(new_base, np.ndarray)):
                    old_base[:] = new_base
                else:
                    old_val[:] = new_val
            continue
        except (AssertionError, AttributeError, TypeError):
            if old_val!=new_val:
                setattr(self, key, objDict[key])
        except ValueError:
            pass


def positionalSaturation(self, xDim, nBins=1000, fact=0.1):
    ''' compute the average saturation of the aqueous phase by position along the x-axis '''
    arrList = []
    isinsideBox = np.zeros(self.totElements, dtype='bool')
    x, deltaX, w = 0.0, xDim/(2*nBins), fact*xDim
    xstart, xend = x-w/2, x+w/2
    while x<=self.xDim+1e-10:
        isinsideBox[self.poreList] = (self.x_array[1:-1] >= xstart) & (
            self.x_array[1:-1] <= xend)
        isinsideBox[self.tList] = (
            isinsideBox[self.P1array] | isinsideBox[self.P2array])
        sat = (self.satList[isinsideBox]*self.volarray[isinsideBox]).sum()/(
            self.volarray[isinsideBox].sum())
        isinsideBox[isinsideBox] = False
        if not np.isnan(sat):
            arrList.append([x,sat])
            x += deltaX
            xstart += deltaX
            xend += deltaX
        else:
            xstart, xend = xstart-deltaX, xend+deltaX


    return np.array(arrList)
    



        
        









