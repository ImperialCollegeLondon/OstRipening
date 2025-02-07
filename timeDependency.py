import numpy as np
from time import time
from joblib import Parallel, delayed
from multiprocessing import shared_memory
from joblib import dump, load, Parallel, delayed
import os
import shutil
from numba import njit
import pnflowPy.utilities as do

from analytical_numerical_study import plotClass

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
        obj.imposedP = imposedP #imposedP    #1MPa
        obj.Pc = Pc
        obj._dt = dt
        obj.adjustTime = adjustTime
        print('@@@@@@@@@@@@@@@@@@@')

        #self.initialize()

# Define a memory-mapped directory (ensure it exists)
MEMORY_DIR = "./joblib_memmap"
os.makedirs(MEMORY_DIR, exist_ok=True)

def save_memmap(array, filename):
    """Saves array as a memory-mapped file and returns the filename."""
    filepath = os.path.join(MEMORY_DIR, filename)
    dump(array, filepath)
    return filepath


def initialize(self):
    # setting up the arrays for elements ...
    st=time()
    self.len_tij = LenEq1(self)
    self.gasConc = np.zeros([self.totElements, 2], dtype='float')
    self.dissolvedMoles = np.zeros(self.totElements, dtype='float')
    self.delta_nMoles = np.zeros(self.totElements, dtype='float')
    self.satList = np.zeros(self.totElements)
    self.satList[1:-1] = self.areaWPhase[1:-1]/self.areaSPhase[1:-1]
    self.volarrayW = self.satList*self.volarray
    self.volarrayNW = (1-self.satList)*self.volarray
    self.maxCornerArea = np.maximum(self.cornerArea, self.areaWPhase)
    self.initialPcArr = np.ones(self.totElements)*self.Pc
    self.initialPcArr[self.hasNWFluid] = self.clusterNW.pc[self.clusterNW_ID[self.hasNWFluid]]

    # resize the cluster arrays
    resizeClusters(self, self.clusterW, self.clusterNW)
    
    # setting up useful arrays
    settingUpArrays(self, self.clusterNW)

    # initializing moles, pc and concentration for NW cluster
    initializingPcMolesNWCluster(self, self.clusterNW)

    # initializing moles, pc and concentration for W/NW pores/throats
    self.netFlux = np.zeros(self.totElements)
    initializingMolesConcElements(self)

    # initialize the flow rates of the phases
    #initializeFlowrate(self)

    # initialize clusters' neighbouring clusters
    #from IPython import embed; embed()
    initializeNeighbouringClusters(self)

    # for writing data/results
    self.resultsP_str = "# Step,totalFlux,total_delta_nMoles,Sw, \
        #cluster_growth,#cluster_shrinkage,AvgPressure,totMoles_W,totMoles_NW,totMoles"
    # self.__fileName__()
    print('Im in timeDependency!')
    print('time spent:  ', time()-st)


def resizeClusters(self, clustW, clustNW):
    ''' resize the W/NW clusters by removing empty clusters id at the end '''

    id = np.setdiff1d(np.where(clustW.size==0)[0], clustW.availableID)
    clustW.availableID.update(id[id>0])
    clustW.neighbours = np.zeros(clustW.members.shape, dtype=bool)
    clustW.updateNeighMatrix()

    id = np.setdiff1d(np.where(clustNW.size==0)[0], clustNW.availableID)
    clustNW.availableID.update(id[id>0])
    clustNW.neighbours = np.zeros(clustNW.members.shape, dtype=bool)
    clustNW.updateNeighMatrix()


def settingUpArrays(self, clust):
    ''' sets up useful arrays for the time-dependent simulation '''

    # identify valid pore-throat pairs and clusters
    self.TPCond = (self.TPConnections>0) # location of valid pores connected to each throats
    self.TValid = np.dstack((self.tList, self.tList))[0][self.TPCond[1:]] # valid throats (Oren)
    self.tValid = self.TValid-self.nPores-1
    self.TPValid = self.TPConnections[self.TPCond] # valid pores connected to each throats
    self.valClust = (clust.members.any(axis=1)) # returns valid clusters
    self.len_tij_valid = self.len_tij[self.TPCond[1:]] # returns len_tij of valid throat-pore pairs
    self.toUpdateNW = np.array(clust.keys)[
        self.valClust & (~clust.connected)] # clusters that are to be updated

    # setting up arrays for clusters
    clust.satList = 1-self.satList
    clust.satList[[-1,0]] = 0.0  # pore indices -1 and 0 are not real pores
    self.initialCornerArea= self.cornerArea.copy()
    
    clust.volume = np.bincount(
        self.clusterNW_ID[self.hasNWFluid], self.volarrayNW[self.hasNWFluid], clust.pc.size)
    clust.totalVolume  = np.bincount(
        self.clusterNW_ID[self.hasNWFluid], self.volarray[self.hasNWFluid], clust.pc.size)
    
    # setting up arrays for elements
    self.elemToUpdateW = (self.fluid==0)
    self.elemToUpdateW[[-1,0]] = False                           # boolean
    self._elemToUpdateW = self.elementListS[self.elemToUpdateW]  # index
    self.elemToUpdateNW = self.hasNWFluid.copy()
    self._elemToUpdateNW = self.elementListS[self.elemToUpdateNW]           # index
    self.elemToUpdate = (self.elemToUpdateW|self.elemToUpdateNW)
    self._elemToUpdate = self.elementListS[self.elemToUpdate]   # index


def initializingPcMolesNWCluster(self, clust):
    ''' initialize moles, pc and gas conc for  NW cluster '''
    nClustNW = self.clusterNW.pc.size
    clust.toDrain = np.ones(nClustNW, dtype=int)*(self.totElements+1)
    clust.toImbibe = np.ones(nClustNW, dtype=int)*(self.totElements+1)
    
    #clust.volShrink = np.zeros(nClustNW)
    clust.pcMax = np.full(nClustNW, 1e30)           # maximum expected value
    clust.pcShrink = np.full(nClustNW, -1e30)       # least expected value
    clust.molesMax = np.full(nClustNW, 1e30)      # maximum expected value
    clust.molesShrink = np.zeros(nClustNW)      # least expected value
    clust.volMax = np.zeros(nClustNW)
    clust.volShrink = np.zeros(nClustNW)

    # initializing cluster moles and concentrations
    self.initialPc = clust.pc.copy()
    self.initialVolume = clust.volume.copy()
    clust.moles = (
        (self.imposedP+self.initialPc)*clust.volume/(self.R*self.T)
        + self.H*self.initialPc*(clust.totalVolume-clust.volume))
    self.initialMoles = clust.moles.copy()
    validKeys = np.where(self.valClust)[0]
    updateVolumeShrinkMax(self, clust, validKeys)
    clust.updateMolesShrinkGrowth(validKeys, self)


def updateVolumeShrinkMax(self, clust, validKeys):
    clust.elemCornerArea = np.zeros(self.totElements, dtype=bool)
    [clust.updateToDrainImbibe(k, self) for k in validKeys]

    mem = self.elementListS[clust.elemCornerArea]
    clust.computeCornerArea(validKeys, clust.elemCornerArea, clust.pcMax, self)
    elemID = self.clusterNW_ID[mem]
    conda = (clust.pcMax[elemID]>=self.initialPc[elemID])
    condb = (self.cornerArea[mem]>self.initialCornerArea[mem])
    _mem = mem[(conda==condb)]
    #_mem = mem[((conda&condb)|(~conda&~condb))]
    self.cornerArea[_mem] = self.initialCornerArea[_mem]
    vol = (1-self.cornerArea[mem]/self.areaSPhase[mem])*self.volarray[mem]
    clust.volMax[validKeys] = np.bincount(self.clusterNW_ID[mem], vol)[validKeys]

    clust.elemCornerArea[clust.toImbibe[validKeys]] = False
    mem = self.elementListS[clust.elemCornerArea]
    clust.computeCornerArea(validKeys, clust.elemCornerArea, clust.pcShrink, self)
    elemID = self.clusterNW_ID[mem]
    conda = (clust.pcShrink[elemID]<=self.initialPc[elemID])
    condb = (self.cornerArea[mem]<self.initialCornerArea[mem])
    _mem = mem[(conda==condb)]
    #_mem = mem[((conda&condb)|(~conda&~condb))]
    self.cornerArea[_mem] = self.initialCornerArea[_mem]
    vol = (1-self.cornerArea[mem]/self.areaSPhase[mem])*self.volarray[mem]
    clust.volShrink[validKeys] = np.bincount(
        self.clusterNW_ID[mem], vol, validKeys.max()+1)[validKeys]
    
    
def initializingMolesConcElements(self):
    # moles of gas in elements filled with WPhase are all updated
    # but only members of NWPhase clusters not connected across s
    # the network are updated.
    
    ''' initialize the NW elements according to their trapped Pc!!! '''
    self.gasConc[self._elemToUpdateNW,0] = self.H*self.clusterNW.pc[
        self.clusterNW_ID[self._elemToUpdateNW]]
    
    ''' initialize the brine to have conc equivalent to Pc!!! '''
    #self.computeAqAvgPres()
    self.aqAvgPres = self.Pc
    #self.aqAvgPres = 4000
    self.gasConc[self._elemToUpdateW,0] = self.H*self.aqAvgPres
    #self.assignConcToAqElements()
    self.dissolvedMoles[self._elemToUpdateW] = (
        self.gasConc[self._elemToUpdateW,0]*self.volarray[self._elemToUpdateW])
    

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


@njit(parallel=True)
def findNextT1(arrP, notdone, done, PTConnections, PTValid):
    result = []
    for p in arrP:
        valid_T = PTValid[p]
        arrT = PTConnections[p][valid_T]
        for t in arrT:
            if notdone[t]:
                result.append(t)
                done[t] = True
                notdone[t] = False
    return np.array(result, dtype=np.int32)

#@njit
def findNextP1(arrT, notdone, done, TPConnections, TPCond):
    mask = TPCond[arrT]
    arrP = TPConnections[arrT][mask]
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
    newClustID = np.unique(clusterID[nextT][clusterID[nextT]>=0])
    foundClustID.extend(newClustID)
    notdone[clusterNW_mem[newClustID].any(axis=0)] = False
    while True:
        try:
            assert arrT.size>0
            arrP = findNextP(arrT-nPores, notdone, done, TPConnections, TPCond)
            newClustID = np.unique(clusterID[arrP][clusterID[arrP]>=0])
            foundClustID.extend(newClustID)
            notdone[np.any(clusterNW_mem[newClustID], axis=0)] = False
            arrP = arrP[clusterID[arrP]<0]
            assert arrP.size>0
            arrT = findNextT(arrP, notdone, done, PTConnections, PTValid)
            newClustID = np.unique(clusterID[arrT][clusterID[arrT]>=0])
            foundClustID.extend(newClustID)
            notdone[np.any(clusterNW_mem[newClustID], axis=0)] = False
            arrT = arrT[clusterID[arrT]<0]
        except AssertionError:
            return np.array(foundClustID)


def findNeighbouringClusters_with_shm(c, totElements, nPores, shm_names, shm_shapes, elemToUpdate):
    """Worker function that retrieves shared arrays using their names."""
    
    # Attach to shared memory (reconstruct arrays)
    shm_members = shared_memory.SharedMemory(name=shm_names["members"])
    members = np.ndarray(shm_shapes["members"], dtype=bool, buffer=shm_members.buf)

    shm_neighbours = shared_memory.SharedMemory(name=shm_names["neighbours"])
    neighbours = np.ndarray(shm_shapes["neighbours"], dtype=bool, buffer=shm_neighbours.buf)
    
    shm_clusterNW_ID = shared_memory.SharedMemory(name=shm_names["clusterNW_ID"])
    clusterNW_ID = np.ndarray(shm_shapes["clusterNW_ID"], dtype=np.int64, 
                              buffer=shm_clusterNW_ID.buf)

    shm_PTConnections = shared_memory.SharedMemory(name=shm_names["PTConnections"])
    PTConnections = np.ndarray(shm_shapes["PTConnections"], dtype=np.int64, 
                               buffer=shm_PTConnections.buf)
    
    shm_PTValid = shared_memory.SharedMemory(name=shm_names["PTValid"])
    PTValid = np.ndarray(shm_shapes["PTValid"], dtype=bool, buffer=shm_PTValid.buf)
    
    shm_TPConnections = shared_memory.SharedMemory(name=shm_names["TPConnections"])
    TPConnections = np.ndarray(shm_shapes["TPConnections"], dtype=np.int32,
                               buffer=shm_TPConnections.buf)

    shm_TPCond = shared_memory.SharedMemory(name=shm_names["TPCond"])
    TPCond = np.ndarray(shm_shapes["TPCond"], dtype=bool, buffer=shm_TPCond.buf)

    notdone = elemToUpdate.astype(bool, copy=True)
    
    # Call the actual function
    return findNeighbouringClusters(c, totElements, nPores, members, neighbours, clusterNW_ID, 
                                    PTConnections, PTValid, TPConnections, TPCond, notdone)


def initializeNeighbouringClusters1(self):
    cluster_size = self.clusterNW.pc.size  # Extract required data
    # Create shared memory for large arrays
    shm_members, shm_members_arr = create_shared_array(self.clusterNW.members)
    shm_neighbours, shm_neighbours_arr = create_shared_array(self.clusterNW.neighbours)
    shm_clusterNW_ID, shm_clusterNW_ID_arr = create_shared_array(self.clusterNW_ID)
    shm_PTConnections, shm_PTConnections_arr = create_shared_array(self.PTConnections)
    shm_PTValid, shm_PTValid_arr = create_shared_array(self.PTValid)
    shm_TPConnections, shm_TPConnections_arr = create_shared_array(self.TPConnections)
    shm_TPCond, shm_TPCond_arr = create_shared_array(self.TPCond)

    shm_names = {
        "members": shm_members.name,
        "neighbours": shm_neighbours.name,
        "clusterNW_ID": shm_clusterNW_ID.name,
        "PTConnections": shm_PTConnections.name,
        "PTValid": shm_PTValid.name,
        "TPConnections": shm_TPConnections.name,
        "TPCond": shm_TPCond.name,
    }

    shm_shapes = {
        "members": shm_members_arr.shape,
        "neighbours": shm_neighbours_arr.shape,
        "clusterNW_ID": shm_clusterNW_ID_arr.shape,
        "PTConnections": shm_PTConnections_arr.shape,
        "PTValid": shm_PTValid_arr.shape,
        "TPConnections": shm_TPConnections_arr.shape,
        "TPCond": shm_TPCond_arr.shape,
    }
    
    args = (self.totElements, self.nPores)
    self.neighbouringClusters = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(findNeighbouringClusters_with_shm)(
            c, *args, shm_names, shm_shapes, self.elemToUpdate) for c in range(cluster_size))
    self.neighbouringClusters = np.array(self.neighbouringClusters, dtype="object")

    shm_members.close()
    shm_neighbours.close()
    shm_clusterNW_ID.close()
    shm_PTConnections.close()
    shm_PTValid.close()
    shm_TPConnections.close()
    shm_TPCond.close()

    shm_members.unlink()
    shm_neighbours.unlink()
    shm_clusterNW_ID.unlink()
    shm_PTConnections.unlink()
    shm_PTValid.unlink()
    shm_TPConnections.unlink()
    shm_TPCond.unlink()

def initializeNeighbouringClusters(self):
    cluster_size = self.clusterNW.pc.size  

    # Save arrays as memory-mapped files
    memmap_files = {
        "members": save_memmap(self.clusterNW.members, "members.pkl"),
        "neighbours": save_memmap(self.clusterNW.neighbours, "neighbours.pkl"),
        "clusterNW_ID": save_memmap(self.clusterNW_ID, "clusterNW_ID.pkl"),
        "PTConnections": save_memmap(self.PTConnections, "PTConnections.pkl"),
        "PTValid": save_memmap(self.PTValid, "PTValid.pkl"),
        "TPConnections": save_memmap(self.TPConnections, "TPConnections.pkl"),
        "TPCond": save_memmap(self.TPCond, "TPCond.pkl"),
    }

    args = (self.totElements, self.nPores, self.elemToUpdate)
    # Run parallel processing with joblib
    self.neighbouringClusters = Parallel(n_jobs=-1, batch_size=100)(
        delayed(findNeighbouringClusters_with_joblib)(
            c, *args, memmap_files) for c in range(cluster_size)
    )

    self.neighbouringClusters = np.array(self.neighbouringClusters, dtype="object")
    # Call this after processing if you don’t need the data anymore
    #cleanup_memory()

def findNeighbouringClusters_with_joblib(c, totElements, nPores, elemToUpdate, memmap_files):
    """Worker function that loads memory-mapped arrays."""

    # Load memory-mapped arrays
    members = load(memmap_files["members"], mmap_mode="r")
    neighbours = load(memmap_files["neighbours"], mmap_mode="r")
    clusterNW_ID = load(memmap_files["clusterNW_ID"], mmap_mode="r")
    PTConnections = load(memmap_files["PTConnections"], mmap_mode="r")
    PTValid = load(memmap_files["PTValid"], mmap_mode="r").astype(bool)  # Ensure boolean dtype
    TPConnections = load(memmap_files["TPConnections"], mmap_mode="r")
    TPCond = load(memmap_files["TPCond"], mmap_mode="r")
    notdone = elemToUpdate.astype(bool, copy=True)

    
    # Call the actual function
    return findNeighbouringClusters(c, totElements, nPores, members, neighbours, 
                                    clusterNW_ID, PTConnections, PTValid, 
                                    TPConnections, TPCond, notdone)

def cleanup_memory():
    """Deletes all memory-mapped files after processing."""
    shutil.rmtree(MEMORY_DIR)


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
    self.valClust = (self.clusterNW.size>0)
    self.toUpdateNW = np.array(self.clusterNW.keys)[self.valClust & (~self.clusterNW.connected)]


def LenEq(self):
    term1 = self.Rarray[self.TPConnections[1:]]/self.areaSPhase[self.TPConnections[1:]]
    term1[np.isnan(term1)] = 0.0
    term2 = self.LTarray/(2*self.areaSPhase[self.tList])
    return 1/(term1+term2[:,np.newaxis])

def LenEq1(self):
    term1 = np.nan_to_num(1/self.Rarray[self.TPConnections[1:]])
    term2 = self.LTarray/(2*self.areaSPhase[self.tList])
    return 1/(term1 + term2[:,np.newaxis])

def computeFluxes(self, advection=False):
    ''' compute flux for each element and cluster '''
    try:
        flux = self.D*self.len_tij_valid*(
            self.gasConc[self.TPValid, 0] - self.gasConc[self.TValid, 0])
        assert not advection
    except AssertionError:
        flux += (self.flowrateNW_TValid*self.dissolvedMoles[self.upstreamElement]/
                    self.volarrayW[self.upstreamElement])
    
    # compute the net flux for each element
    self.netFlux[self.tList] = np.bincount(self.tValid, flux, self.nThroats)
    self.netFlux[self.poreListS] = np.bincount(self.TPValid, -flux, self.nPores+2)

    # compute the net flux for each cluster
    self.netFluxClusters = np.bincount(
        self.clusterNW_ID[self._elemToUpdateNW], self.netFlux[self._elemToUpdateNW],
        len(self.clusterNW.keys))
    
    # if self.netFluxClusters.sum()>1e-20:
    #     print('net flux clusters is too high, check!!!')
    #     from IPython import embed; embed()
    
    # self.dissolvedMoles[self._elemToUpdateW] = (
    #     self.gasConc[self._elemToUpdateW,0]*self.volarray[self._elemToUpdateW])

def determineTimeStep(self):
    dissolvedMoles = (self.dissolvedMoles+self.netFlux*self._dt)[self._elemToUpdateW]
    ind = self._elemToUpdateW[dissolvedMoles<0.0]
    try:
        self.dt = np.min(-self.dissolvedMoles[ind]/self.netFlux[ind])
    except ValueError:
        self.dt = self._dt

    clusterMoles = (self.clusterNW.moles+self.netFluxClusters*self.dt)
    condC = (clusterMoles<self.clusterNW.molesShrink)&self.valClust
    try:
        self.dt = (-(self.clusterNW.moles[condC]-self.clusterNW.molesShrink[condC])/
                    self.netFluxClusters[condC]).min()
        clusterMoles = (self.clusterNW.moles+self.netFluxClusters*self.dt)
        self.updateClust = True
    except ValueError:
        pass

    condC = (clusterMoles>self.clusterNW.molesMax)&self.valClust
    try:
        self.dt = ((self.clusterNW.molesMax[condC]-self.clusterNW.moles[condC])/
                    self.netFluxClusters[condC]).min()
        self.updateClust = True
    except ValueError:
        pass

    if self.dt<1e-10:
        print('I am in dt:  {self.dt}')
        #from IPython import embed; embed()
        self.dt = 1e-10

    

def computeMolesPcConc(self):
    ''' compute and update the moles, pc and concentration of gas in each element and cluster'''
    
    determineTimeStep(self)
    clust = self.clusterNW

    ''' update element filled with WP properties '''
    self.dissolvedMoles[self._elemToUpdateW] = np.clip(
        (self.dissolvedMoles+self.netFlux*self.dt)[self._elemToUpdateW], 0.0, None)
    
    self.gasConc[self._elemToUpdateW, 1] = np.nan_to_num(
        self.dissolvedMoles[self._elemToUpdateW]/self.volarray[self._elemToUpdateW])       
    
    ''' update NWP cluster properties '''
    try:
        assert (self.netFluxClusters>0).any()
        _netFluxClusters = np.zeros_like(self.netFluxClusters)
        clustIDS = np.where(self.netFluxClusters>0)[0]
        for i in clustIDS:
            cIDS = self.neighbouringClusters[i]
            try:
                cIDS = cIDS[self.clusterNW.pc[cIDS]>self.clusterNW.pc[i]]
                cIDS = cIDS[self.netFluxClusters[cIDS]<0.0]
                nfC = np.cumsum(self.netFluxClusters[cIDS])+self.netFluxClusters[i]
                _netFluxClusters[cIDS[nfC>0]] += self.netFluxClusters[cIDS[nfC>0]]
                self.netFluxClusters[cIDS[nfC>0]] = 0.0
                remFlux = self.netFluxClusters[i]+_netFluxClusters[cIDS[nfC>0]].sum()
                _netFluxClusters[cIDS[nfC<0][0]] -= remFlux
                self.netFluxClusters[cIDS[nfC<0][0]] += remFlux
                _netFluxClusters[i] = self.netFluxClusters[i]
            except IndexError:
                continue
            
        clust.moles = (clust.moles+_netFluxClusters*self.dt)
        cond1 = (clust.moles[self.toUpdateNW]>self.initialMoles[self.toUpdateNW])
        cond2 = (clust.moles[self.toUpdateNW]<self.initialMoles[self.toUpdateNW])
        toUpdateNW1 = self.toUpdateNW[cond1]
        toUpdateNW2 = self.toUpdateNW[cond2]
    
        clust.pc[toUpdateNW1] = (clust.moles[toUpdateNW1]-self.initialMoles[toUpdateNW1])/(
            clust.molesMax[toUpdateNW1]-self.initialMoles[toUpdateNW1])*(
                clust.pcMax[toUpdateNW1]-self.initialPc[toUpdateNW1]) + self.initialPc[toUpdateNW1]
        clust.pc[toUpdateNW2] = (clust.moles[toUpdateNW2]-clust.molesShrink[toUpdateNW2])/(
            self.initialMoles[toUpdateNW2]-clust.molesShrink[toUpdateNW2])*(
                self.initialPc[toUpdateNW2]-clust.pcShrink[toUpdateNW2]) + clust.pcShrink[
                    toUpdateNW2]
    except AssertionError:
        pass
    except:
        print('there is an error on line 612, checkkkkk!!!!')
        from IPython import embed; embed()
    
    try:
        assert any((clust.pc<0.0) | (clust.pc>self.maxPc))
        print('there is an unrealistic pc, check line 427!!!')
        breakpoint()
    except AssertionError:
        pass

    # update conc and moles but check if the cluster gas becomes completely dissolved
    #cond = self.valClust & ((self.netFluxClusters<0.0)|(clust.moles>=clust.molesMax))
    gasConcClusters = self.H*clust.pc
    #elemToUpdateNW = self._elemToUpdateNW[cond[self.clusterNW_ID[self._elemToUpdateNW]]]
    elemToUpdateNW = self.elemToUpdateNW
    self.gasConc[elemToUpdateNW,1] = gasConcClusters[self.clusterNW_ID[elemToUpdateNW]]

    #print(self.gasConc[[8180,12082,2915]])

def create_shared_array(array):
    """Creates a shared memory array and returns the shared object and np.ndarray"""
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)  # Allocate shared memory
    dtype = np.uint8 if array.dtype == bool else array.dtype
    shm_arr = np.ndarray(array.shape, dtype=dtype, buffer=shm.buf)  # Create NumPy array view
    np.copyto(shm_arr, array)  # Copy original data to shared memory
    return shm, shm_arr


def simulateOstRip(self, implicit=True):
    duration = 3600*24*2  # 24-hours
    duration = 200
    st = time()
    totalTime, totTime = 0.0, 0.0
    self.updateClust = False

    if not os.path.isfile('volarray_bent.dat'):
        np.savetxt('volarray_bent.dat', self.volarray)
    if not os.path.isfile('coordinates_bent.dat'):
        np.savetxt('coordinates_bent.dat', 
                    np.column_stack((self.x_array, self.y_array, self.z_array)))
        
    # records data in at most 10-mins intervals if no event occur
    minTime = 600
    #self.writeData(totalTime)
    ii = 1
    self.clusterNW.drainEvents, self.clusterNW.imbEvents = 0, 0
    toShrink_elements = np.loadtxt('bent_toShrink_elements.dat', delimiter=',')
    
    def _fff(ii, totTime, totalTime):
        try:
            self.valClustD = self.valClust & (self.clusterNW.toDrain<=self.totElements)
        except ValueError:
            self.clusterNW.resizeArrays(self)
            self.valClustD = self.valClust & (self.clusterNW.toDrain<=self.totElements)
        try:
            # check for further shrinkage!!!
            assert (self.clusterNW.moles>self.clusterNW.molesShrink)[self.valClust].all()
            assert (self.clusterNW.moles<self.clusterNW.molesMax)[self.valClustD].all()
            #self.computeAqAvgPres()
            #self.gasConc[self._elemToUpdateW,0] = self.H*self.aqAvgPres
            #self.dissolvedMoles[self._elemToUpdateW] = (
                #  self.gasConc[self._elemToUpdateW,0]*self.volarray[self._elemToUpdateW])
            computeFluxes(self)
            computeMolesPcConc(self)
        except AssertionError:
            self.dt = 0.0
            self.updateClust = True
            self.gasConc[self.elemToUpdate,1] = self.gasConc[self.elemToUpdate,0]
            

        try:
            totTime += self.dt
            assert self.updateClust or (totTime>minTime)     #check for every second simulation
            condShrink = (self.clusterNW.moles<=self.clusterNW.molesShrink) & self.valClust
            condGrowth = (self.clusterNW.moles>=self.clusterNW.molesMax) & self.valClustD
            totalTime += totTime
            totTime = 0.0

            try:
                assert condShrink.any()
                keys = np.array(self.clusterNW.keys)[condShrink]
                toImbibe = self.clusterNW.toImbibe[keys]
                print('@@:   ', keys, toImbibe, toImbibe in toShrink_elements)
                #from IPython import embed; embed()
                self.clusterNW.shrinkCluster(keys, self)
                self.clusterNW.imbEvents += condShrink.sum()
                updateLists(self, toImbibe)
                self.updateClust = False
            except AssertionError:
                pass

            try:
                assert condGrowth.any()
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^')
                keys = np.array(self.clusterNW.keys)[condGrowth]
                toDrain = self.clusterNW.toDrain[keys]
                print('$$:  ', keys, toDrain)
                #from IPython import embed; embed()
                self.clusterNW.growCluster(keys, self)
                self.clusterNW.drainEvents += condGrowth.sum()
                updateLists(self, toDrain, 1)
                self.updateClust = False
            except AssertionError:
                pass

            totalMoles = (
                (self.volarray*self.gasConc[:,1])[self.elemToUpdateW].sum()+
                self.clusterNW.moles.sum())
            self.aqAvgPres = self.gasConc[self._elemToUpdateW,1].mean()/self.H
            
            sat = (self.satList[self.isinsideBox]*self.volarray[
                self.isinsideBox]).sum()/self.totVoidVolume
            print(("#:%8.6g  \tSimulation Time:%12.6g  \tActual runtime:%6.6g\
                \tTotal cluster moles:%8.6e  \tTotal dissolved moles:%6.6e\
                \tTotal gas moles:%6.6e \tAvg Pressure:%8.6e \tNo Shrinkage:%8.6g\
                \tNo Growth:%8.6g \tSat:%6.6g \tAqAvgPres:%6.6g" % (
            ii, round(totalTime,3), round(time()-st,3), self.clusterNW.moles.sum(), 
            self.dissolvedMoles.sum(), totalMoles,
            self.clusterNW.pc[self.valClust].mean(),
            self.clusterNW.imbEvents, self.clusterNW.drainEvents, sat, self.aqAvgPres)))
            #self.writeData(round(totalTime,3))
    
            ii += 1
        except AssertionError: 
            pass
                    
        self.gasConc[self.elemToUpdate,0] = self.gasConc[self.elemToUpdate,1]
        return ii, totTime, totalTime


    #print('::::::::::::::::::::::::::::')
    #from IPython import embed; embed()
    while totalTime < duration:
        ii, totTime, totalTime = _fff(ii, totTime, totalTime)
        #print(ii)

    print('::::::::::::::::::::::::::::')
    from IPython import embed; embed()
    
    
    # plot.plot_conc_t()
    # plot = plotClass('bent', (nRow, self.totElements), D=self.D)
    # plot.plot_conc_x()
    # plot.plot_conc_universal()
    # from IPython import embed; embed()










