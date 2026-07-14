import numpy as np
from time import time
import os
import joblib
from numba import prange, njit
from .cluster import *
from . import temp



#os.makedirs(MEMORY_DIR, exist_ok=True)
_temp = temp.TempArrays()


class TimeDependency:
    def __init__(self, network):
        
        print('----------------------------------------------------------------------------------')
        print('-----------------------------Time Dependent Ostwald Ripening ---------------------')
 
        # initialising the parameters
        
        network.R = 8.314  #J/mol.K
       
        network.HimposedP = network.H*network.imposedP
        
        network.RT = 8.314*network.T
        network.half_pi = np.pi/2.0
        

def LenEq(self):
    ''' returns the diffusion characteristic length for each P1-T-P2 pair '''
    return LenEq_numba(self.P1array, self.P2array, self.LP1array, self.LP2array,
    	self.LTarray, self.areaSPhase, self.Rarray, self.tList, self.totElements, self.nThroats)


def initialize(self):
    # setting up the arrays for elements ...
    print('Initializing timeDependency ...........................', end='  ', flush=True)
    st=time()   

    self.cNWP.resizeClusters(0, True)
    self.cWP.resizeClusters(0, True)

    # set up necessary arrays
    settingUpArrays(self, self.cNWP, False)

    # initialize the flow rates of the phases
    #initializeFlowrate(self)
    # for writing data/results
    self.resultsP_str = "# Step,totalFlux,total_delta_nMoles,Sw, \
        #cluster_growth,#cluster_shrinkage,AvgPressure,totMoles_W,totMoles_NW,totMoles"
    
    self.j , self.ii, self.totTime, self.totalTime = 0, 0, 0.0, 0.0
    self.initialized = True
    writeData(self, round(self.totalTime,3))
    print('Initializing timeDependency completed!!!', self.cNWP.moles.sum())
    print('time spent:  ', time()-st)


def settingUpArrays(self, cNWP, excludeCircles=False, 
	pc_min=1.0e-3, pc_max=1.0e30, moles_tol=1e-18):
	
    totElements = self.totElements
    
    self.len_tij_valid = LenEq1_numba(self.P1array, self.P2array, 
        self.LP1array, self.LP2array, self.LTarray, self.areaSPhase, self.Rarray, self.tList,
        self.TPValid, self.TValid, self.isCircle, totElements, self.nPores, self.nThroats)
    
    Pc = cNWP.pc[cNWP.clusterID].astype(np.float64)
    arrr = np.ones(self.totElements, dtype=bool)
    arrr[[-1,0]] = False
        
    hasFluid = cNWP.hasFluid
    self.satList = np.zeros(totElements, dtype=np.float64)
    self.satList[1:-1] = self.areaWPhase[1:-1]/self.areaSPhase[1:-1]
    
    do.update_areas_conductances(self, arrr, Pc, False, True, True)
    self.satList[hasFluid] = self._cornArea[hasFluid]/self.areaSPhase[hasFluid]
    cNWP.volume[:] = np.bincount(cNWP.clusterID[hasFluid], 
                                (1-self.satList[hasFluid])*self.volarray[hasFluid],
                                cNWP.nClusters)
    
    self.entryPress_freezed = np.zeros(totElements, dtype=np.bool_)
    self.oldSatW = self.satW 
    self.maxPc = 1e7
   
    nClust = cNWP.nClusters
    keys = np.arange(nClust).astype(np.int32)
    cNWP.minPcArray = np.full(nClust, self.maxPc, dtype=np.float32)
    cNWP.volTotal =  np.zeros(nClust, dtype=np.float64)
    
    cNWP.updateClusterProperties(keys, self, 1e-3, 1e30, False)
    cond = cNWP.valClustD
    cPc = cNWP.pc.copy()
    cMoles = (cNWP.pc+self.imposedP)*cNWP.volume/self.RT
    
    cNWP.pc[cond] = self.alpha*cNWP.pcMax[cond]+(1-self.alpha)*cPc[cond]
    Pc = cNWP.pc[cNWP.clusterID].astype(np.float64)
    do.update_areas_conductances(self, arrr, Pc, False, True, True)
        
    self.satList[hasFluid] = self._cornArea[hasFluid]/self.areaSPhase[hasFluid]
    cNWP.volume[cond] = np.bincount(cNWP.clusterID[hasFluid],
                                (1-self.satList[hasFluid])*self.volarray[hasFluid],
                                cNWP.nClusters)[cond]
    #updateClusterVolume(cNWP.hasFluid, cNWP.clusterID, cNWP.volume, 
    #    self.satList, self.volarray, totElements)
        
    cNWP.maxGasPc = cNWP.pc.max()
    cNWP.moles[cond] = (self.imposedP + cNWP.pc[cond])*cNWP.volume[cond]/self.RT
    
    self.initialAvgPres = ((cNWP.pc+self.imposedP)*cNWP.volume).sum()/cNWP.volume.sum() - self.imposedP
    self.initAqConc = self.H*(self.imposedP + self.initialAvgPres)
    self.minMoles = self.H * 1e-3 * self.volarray
    self.elemToUpdateW = np.zeros(totElements, dtype=np.bool_)
    self.elemToUpdateNW = np.zeros(totElements, dtype=np.bool_)
    self.validVolumes = np.zeros(totElements, dtype=np.bool_)
    self.gasConc = np.zeros(totElements, dtype=np.float64)
    self.aqueousMoles = np.zeros(totElements, dtype=np.float64)
   
    cNWP.validToGrow = cNWP.valClustD & (cNWP.pcMax < self.initialAvgPres)
    cNWP.validToShrink = cNWP.valClust & (cNWP.pcShrink > self.initialAvgPres)
    cNWP.toShrinkList = np.full(nClust, -5, dtype=np.int32)
    cNWP.toGrowList = np.full(nClust, -5, dtype=np.int32)
   
    updateArrays(cNWP.hasFluid, cNWP.validToShrink, cNWP.validToGrow, cNWP.pcShrink, 
        cNWP.pcMax, cNWP.clusterID, cNWP.pc, cNWP.moles,
        self.volarray, self.satList, self.elemToUpdateW, 
        self.elemToUpdateNW, self.validVolumes, self.gasConc, self.aqueousMoles, 
        self.initialAvgPres, self.initAqConc, self.imposedP, self.H, totElements)

    self.flux = np.zeros(self.nValid, dtype=np.float64)
    self.netFlux = np.zeros(self.totElements, dtype=np.float64)
    self.netFluxClusters = np.zeros(nClust, dtype=np.float64)
    self.netMoles = np.zeros(self.totElements, dtype=np.float64)
    
       

def compute_fluxes_moles_pc_conc(self, cNWP):
    self.dt = compute_fluxes_moles_pc_conc_numba(self.flux, self.netFlux, 
        self.len_tij_valid, self.gasConc, self.volarray, self.TPValid, self.TValid, 
        cNWP.clusterID, cNWP.moles, cNWP.volume, cNWP.pc, cNWP.sizes, cNWP.netClustMoles, 
        cNWP.netMolesAfterLastUpdate, cNWP.members, cNWP.mem_offsets, 
        self.elemToUpdateW, self.elemToUpdateNW, self.aqueousMoles, self.minMoles, cNWP.minPcArray,
        self.maxPc, _temp.filterNext, self.imposedP, self.D, self.H, self.RT, self._dt, self.moles_tol)
    cNWP.adjustClusterPcVolume(self, self.moles_tol, self.pc_tol)


def statistics(self, cNWP):
    return statistics_numba(self.elemToUpdateW, self.elemToUpdateNW, self.aqueousMoles, 
        self.satList, self.volarray, cNWP.moles, cNWP.prequilibratedMoles, cNWP.valClust, 
        cNWP.pc, cNWP.volume, self.totElements, self.isinsideBox, self.totVoidVolume, 
        self.H, self.imposedP)
        
        
def assignSaturation(self, cNWP):
    assignSaturation_numba(self.isCircle, self.satList, 
        cNWP.volume, cNWP.sizes, cNWP.members, cNWP.mem_offsets, self.volarray)



def simulate(self, ii, startTime, totTime, totalTime, minTime, waitTime):
    ''' main simulation loop for time-dependent Ostwald ripening '''

    cNWP = self.cNWP
    if waitTime > 1:
        nShrinkage, nGrowth = checkEvents(cNWP.sizes, cNWP.pc, cNWP.pcShrink, cNWP.pcMax, 
            cNWP.validToShrink, cNWP.validToGrow, cNWP.valClust, cNWP.toShrinkList, 
            cNWP.toGrowList, self.initialAvgPres)
        waitTime = 0.0
    else:
        nShrinkage, nGrowth = 0, 0

    if nShrinkage or nGrowth or totTime > minTime:
        if nShrinkage:
            keys = cNWP.toShrinkList[:nShrinkage]
            print("@@:   ", keys,   cNWP.toImbibe[keys])
            cNWP.shrinkCluster(keys, self)
            cNWP.imbEvents += nShrinkage
            
        if nGrowth:
            keys = cNWP.toGrowList[:nGrowth]
            print("££:   ", keys,   cNWP.toDrain[keys])
            cNWP.growCluster(keys, self)
            cNWP.drainEvents += nGrowth
           
        totalTime += totTime 
        totTime = 0.0
        waitTime = 0.0
        
        writeOnScreen(self, cNWP, ii, totTime, totalTime, startTime)
        writeData(self, round(totalTime,3))
        if totalTime > self.timeToSave:
            try:
                filename = os.path.join(self.MEMORY_DIR, f"netsim_time_Dependent_{self.j}.pkl")
                saveState(self, filename)
                self.j += 1
                self.timeToSave = minTime*self.j
            except:
                print('Could not pickle/save the state !!!')

        ii += 1
        
        if nShrinkage or nGrowth:
            cNWP.validToGrow = cNWP.valClustD & (cNWP.pcMax < self.initialAvgPres) 
            cNWP.validToShrink = cNWP.valClust & (cNWP.pcShrink > self.initialAvgPres)
           
    else:
        compute_fluxes_moles_pc_conc(self, cNWP)
        totTime += self.dt
        waitTime += self.dt
              

    return ii, totTime, totalTime, waitTime


def simulateOstRip(self):
    st = time()
    
    print(f"\nalpha={self.alpha}       imposedP={self.imposedP}        \
            aqAvgPres={self.initialAvgPres}      H={self.H}      D={self.D}   \n")

    cNWP = self.cNWP
    if not hasattr(_temp, 'nClusters'):
        temp.TempArrays.formTempClustArrays(cNWP.nClusters)
    
    
    writeOnScreen(self, cNWP, self.ii, self.totTime, self.totalTime, st)
    self.j += 1
    self.ii += 1
    minTime = 600
    waitTime = 0.0
    
    
    self.timeToSave = minTime*self.j
    
    while self.totalTime < self.duration:
        try:
            self.ii, self.totTime, self.totalTime, waitTime = simulate(
                self, self.ii, st, self.totTime, self.totalTime, minTime, waitTime)
                
            
        except:
            print('there is an error in the simulation, aborting!!!')
            return
            
    print('::::::::::::::::::::::::::::')


def writeOnScreen(self, clust, ii, totTime, totalTime, startTime):
    total_moles_in_gas, total_moles_in_aq, total_moles,\
            self.satW, avgGasPres, avgAqPres = statistics(self, clust)
    totalTime += totTime 
    totTime = 0.0
    
    print(("#:%8.6g  \tSimulation Time:%12.6g  \tActual runtime:%6.6g\
        \tTotal cluster moles:%8.6e  \tTotal dissolved moles:%6.6e\
        \tTotal gas moles:%4.10e \tAvg Pressure:%8.6e \tNo Shrinkage:%8.6g\
        \tNo Growth:%8.6g \tSat:%6.6g \tAqAvgPres:%6.6g" %(
        ii, round(totalTime,3), round(time()-startTime,3), total_moles_in_gas, 
        total_moles_in_aq, total_moles, avgGasPres, clust.imbEvents, clust.drainEvents, 
        self.satW, avgAqPres)))

def saveState(self, fname):
    state_attrs = ['satW', 'rng', 'fluid', 'cWP', 'cNWP', '_dt',  'satList', 'maxPc',
        'H', 'Pc', 'RT', 'imposedP', 'HimposedP', 'D', 'len_tij_valid', 
        'm_inited', 'm_initedApexDist', 'm_advPc', 'm_recPc',
        'elemToUpdateW', 'elemToUpdateNW', 'gasConc', 'aqueousMoles', 'minMoles', 'initAqConc',
       'initialAvgPres', 'flux', 'netFlux', 'j', 'ii', 'totTime', 'totalTime',
       'moles_tol', 'pc_tol', 'max_iter', 'entryPress_freezed']

    state = {attr: getattr(self, attr) for attr in state_attrs}  
    joblib.dump(state, fname, compress=3)
    


def writeData(self, totalTime):
    MEMORY_DIR = self.MEMORY_DIR
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
       
        cNWP = self.cNWP
        np.savetxt(f1, [cNWP.pc], delimiter=',', fmt='%g')
        np.savetxt(f2, [cNWP.volume], delimiter=',', fmt='%g')
        np.savetxt(f3, [cNWP.moles], delimiter=',', fmt='%g')
        np.savetxt(f4, [cNWP.clusterID], delimiter=',', fmt='%g')
        np.savetxt(f5, [self.aqueousMoles], delimiter=',', fmt='%g')
        f6.write(str(totalTime)+',')
        np.savetxt(f7, [[cNWP.imbEvents, cNWP.drainEvents]], delimiter=',', fmt='%g')
        f8.write(str(self.satW)+',')
        np.savetxt(f9, [cNWP.sizes], delimiter=',', fmt='%g')
        np.savetxt(f10, [self.satList], delimiter=',', fmt='%g')


    
@njit(parallel=True, cache=True)
def LenEq_numba(P1array, P2array, LP1array, LP2array, LTarray, areaSPhase,
	Rarray, tList, TPValid, TValid, isCircle, totElements, nPores, nThroats):
        
    lenEq_array = np.empty(nThroats, np.float32)
    nValid = TValid.size
    len_tij_valid = np.zeros(nValid, np.float32)
    for i in prange(nThroats):
        LP1 = LP1array[i]
        LP2 = LP2array[i]
        LT = LTarray[i]

        P1, P2, T = P1array[i], P2array[i], tList[i]
        if P1 < 0: P1 = totElements-1
        if P2 < 0: P2 = totElements-1
        areaP1 = areaSPhase[P1] 
        areaP2 = areaSPhase[P2]
        areaT = areaSPhase[T]

        if areaP1==0.0: areaP1 = np.inf
        if areaP2==0.0: areaP2 = np.inf
        if areaT==0.0: areaT = np.inf

        lenEq_array[i] = 1.0/(Rarray[P1]/areaP1 + LT/areaT + Rarray[P2]/areaP2)
        

    for i in prange(nValid):
        P, T = TPValid[i], TValid[i]
        if isCircle[P] or isCircle[T]: continue
        t = T - nPores - 1
        len_tij_valid[i] = lenEq_array[t] 

    return lenEq_array, len_tij_valid
    
    
@njit(parallel=True, cache=True)
def LenEq1_numba(P1array, P2array, LP1array, LP2array, LTarray, areaSPhase,
	Rarray, tList, TPValid, TValid, isCircle, totElements, nPores, nThroats):
        
    
    nValid = TValid.size
    len_tij_valid = np.zeros(nValid, np.float32)
    for i in prange(nValid):
        P, T = TPValid[i], TValid[i]
        if P < 0: P = totElements-1
        areaP = areaSPhase[P] 
        areaT = areaSPhase[T]
        if areaP==0.0: areaP = np.inf
        if areaT==0.0: areaT = np.inf
        t = T - nPores - 1
        lenEq = 1.0/(Rarray[P]/areaP + 0.5*LTarray[t]/areaT)
        len_tij_valid[i] = lenEq

    return len_tij_valid



