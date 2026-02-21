import numpy as np
from time import time
import math
import os
import joblib

import pnflowPy.tPhaseD as tPhaseD
import pnflowPy.tPhaseImb as tPhaseImb
from .mcluster import *

import OstRipening.temp as temp
from numba import prange, njit


#MEMORY_DIR = f"ostwald_ripening_results/300126_maxPc1e5/impP_1e6_D_4dot89minus9_H_7dot8minus6_clustPcalpha0dot56_minPc1eminus3_1swaiting_final_2"
#alpha = 0.56
MEMORY_DIR = f"ostwald_ripening_results/300126_maxPc7e4/impP_8e6_D_4dot75minus9_H_7dot8minus6_clustPcalpha0dot33_minPc1eminus3_1swaiting_final_2"
alpha = 0.33
os.makedirs(MEMORY_DIR, exist_ok=True)
_temp = temp.TempArrays()


class TimeDependency:
    def __init__(self, network, Pc, T=298, imposedP=1e6, H=7.8e-6, D=7.3e-9, 
                 steps=10, dt=0.0005, adjustTime=False, moles_tol=1e-18, pc_tol=1e-4, max_iter=50):
        
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
    return LenEq_numba(self.P1array, self.P2array, self.LP1array, self.LP2array, #
    	self.LTarray, self.areaSPhase, self.tList, self.totElements, self.nThroats)


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
    # self.__fileName__()
    print('Initializing timeDependency completed!!!', self.cNWP.moles.sum())
    print('time spent:  ', time()-st)
   


def settingUpArrays(self, cNWP, excludeCircles=False, 
	pc_min=1.0e-3, pc_max=1.0e30, moles_tol=1e-18):
	
    totElements = self.totElements
    
    self.len_tij, self.len_tij_valid = LenEq_numba(self.P1array, self.P2array, 
        self.LP1array, self.LP2array, self.LTarray, self.areaSPhase, self.tList,
        self.TPValid, self.TValid, self.isCircle, totElements, self.nPores, self.nThroats)

    self.satList = np.zeros(totElements, dtype=np.float64)
    self.satList[1:-1] = self.areaWPhase[1:-1]/self.areaSPhase[1:-1]
    self.entryPress_freezed = np.zeros(totElements, dtype=np.bool_)
    self.oldSatW = self.satW
    #from IPython import embed; embed()

    nClust = cNWP.nClusters
    keys = np.arange(nClust).astype(np.int32)
    cNWP.updateClusterProperties(keys, self, 1e-3, 1e30, False)
    
    updateClusterVolume(cNWP.hasFluid, cNWP.clusterID, cNWP.volume, 
        self.satList, self.volarray, totElements)
    
        
    #
    #_ = adjustPcUpdateMolesReturnAvgPc(keys, cNWP.pc, 
    #    cNWP.volume, cNWP.moles, cNWP.valClust, cNWP.valClustD, cNWP.pcMax, 
    #    self.imposedP, self.RT, alpha) 
    #self.initialAvgPres = cNWP.pcShrink[cNWP.valClust].max() - 1
   
    #test-run
    #self.initialAvgPres = ((cNWP.pc+self.imposedP)*cNWP.volume).sum()/cNWP.volume.sum() - self.imposedP
    
    #######################################################
    if self.title=='Bentheimer':
        a = 5070.0   #Bentheimer
        #a = 5925.0   #Bentheimer
    elif self.title=='BentSepi600':
        #a = 7640.0    #BentSepi600
        a = 8391.683
        
        
    
    
    #cond = (cNWP.sizes>0)
    #condG = (cond &(cNWP.pcMax<a))
    #condS = (cond &(cNWP.pcShrink>a))
    #self.initialAvgPres = a
    #cNWP.pc[condG] = alpha*cNWP.pcMax[condG]+(1-alpha)*cNWP.pc[condG]
    #cNWP.pc[condS] = alpha*cNWP.pcShrink[condS]+(1-alpha)*cNWP.pc[condS]
    cond = cNWP.valClustD
    cNWP.pc[cond] = alpha*cNWP.pcMax[cond]+(1-alpha)*cNWP.pc[cond]
    #cond = cNWP.valClust
    cNWP.moles[cond] = (self.imposedP + cNWP.pc[cond])*cNWP.volume[cond]/self.RT
    
    #from IPython import embed; embed()
    self.initialAvgPres = ((cNWP.pc+self.imposedP)*cNWP.volume).sum()/cNWP.volume.sum() - self.imposedP
    #self.initialAvgPres = a
    #self.initialAvgPres = cNWP.pc[cNWP.valClust].mean()
    #self.initialAvgPres = 7000
    #######################################################

    self.minConc = self.H*(self.imposedP + self.initialAvgPres)
    #self.minConc = self.H* 1e-3    
    #self.minConc = self.H*self.imposedP
    #self.minConc = self.H*(self.imposedP + avgPres)
    #self.minMoles = self.minConc*self.volarray.astype(np.float64)
    self.minMoles = np.zeros(self.totElements)
    self.elemToUpdateW = np.zeros(totElements, dtype=np.bool_)
    self.elemToUpdateNW = np.zeros(totElements, dtype=np.bool_)
    self.validVolumes = np.zeros(totElements, dtype=np.bool_)
    self.gasConc = np.zeros(totElements, dtype=np.float64)
    self.aqueousMoles = np.zeros(totElements, dtype=np.float64)
    #cNWP.validToShrink = cNWP.valClust.copy()
    #cNWP.validToGrow = cNWP.valClustD.copy()
    cNWP.validToGrow = cNWP.valClustD & (cNWP.pcMax < self.initialAvgPres)
    cNWP.validToShrink = cNWP.valClust & (cNWP.pcShrink > self.initialAvgPres)
    cNWP.toShrinkList = np.full(nClust, -5, dtype=np.int32)
    cNWP.toGrowList = np.full(nClust, -5, dtype=np.int32)
    #self.moles_tol = max(moles_tol, 10.0 * np.finfo(self.aqueousMoles.dtype).eps)
    
    #from IPython import embed; embed()
    
    updateArrays(cNWP.hasFluid, cNWP.validToShrink, cNWP.validToGrow, cNWP.pcShrink, 
        cNWP.pcMax, cNWP.clusterID, cNWP.pc, cNWP.moles, self.volarray, self.satList, self.elemToUpdateW, 
        self.elemToUpdateNW, self.validVolumes, self.gasConc, self.aqueousMoles, 
        self.initialAvgPres, self.minConc, self.imposedP, self.initialAvgPres, self.H, totElements)

    self.flux = np.zeros(self.nValid, dtype=np.float64)
    self.netFlux = np.zeros(self.totElements, dtype=np.float64)
    self.netFluxClusters = np.zeros(nClust, dtype=np.float64)
    self.netMoles = np.zeros(self.totElements, dtype=np.float64)
    
    
    
    #from IPython import embed; embed()


def compute_fluxes_moles_pc_conc(self, cNWP):
    self.dt = compute_fluxes_moles_pc_conc_numba(self.flux, self.netFlux, 
        self.len_tij_valid, self.gasConc, self.volarray, self.TPValid, self.TValid, 
        cNWP.clusterID, cNWP.moles, cNWP.volume, cNWP.pc, cNWP.sizes, cNWP.netClustMoles, 
        cNWP.netMolesAfterLastUpdate, cNWP.members, cNWP.mem_offsets, 
        self.elemToUpdateW, self.elemToUpdateNW, self.aqueousMoles, self.minMoles, self.minConc,
        self.imposedP, self.D, self.H, self.RT, self._dt, self.moles_tol)


def statistics(self, cNWP):
    return statistics_numba(self.elemToUpdateW, self.elemToUpdateNW, self.aqueousMoles, 
        self.satList, self.volarray, cNWP.moles, cNWP.valClust, cNWP.pc, cNWP.volume, self.minConc,
        self.totElements, self.isinsideBox, self.totVoidVolume, self.H, self.imposedP)
        
        
def assignSaturation(self, cNWP):
    assignSaturation_numba(self.isCircle, self.satList, 
        cNWP.volume, cNWP.sizes, cNWP.members, cNWP.mem_offsets, self.volarray)



def simulate(self, clust, ii, startTime, totTime, totalTime, minTime, waitTime,
    moles_tol, pc_tol, max_iter):
    ''' main simulation loop for time-dependent Ostwald ripening '''

    cNWP = self.cNWP
    if waitTime > 1.0:
        nShrinkage, nGrowth = checkEvents(cNWP.sizes, cNWP.pc, cNWP.pcShrink, cNWP.pcMax, cNWP.validToShrink,
            cNWP.validToGrow, cNWP.valClust, cNWP.toShrinkList, cNWP.toGrowList, self.initialAvgPres)
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
            
        cNWP.adjustClusterPcVolume(self, self.moles_tol, self.pc_tol)         

        totalTime += totTime 
        totTime = 0.0
        waitTime = 0.0
        
        
        #print('You need to fix the satList first !!!')
        #from IPython import embed; embed()
        #assignSaturation(self, cNWP)
        writeOnScreen(self, cNWP, ii, totTime, totalTime, startTime)
        writeData(self, round(totalTime,3))
        ii += 1
        
        if nShrinkage or nGrowth:
            #self.initialAvgPres = ((cNWP.pc+self.imposedP)*cNWP.volume).sum()/cNWP.volume.sum() - self.imposedP
            #self.initialAvgPres = cNWP.pc[cNWP.valClust].mean()
            cNWP.validToGrow = cNWP.valClustD & (cNWP.pcMax < self.initialAvgPres)
            cNWP.validToShrink = cNWP.valClust & (cNWP.pcShrink > self.initialAvgPres)
        elif self.entryPress_freezed.any():
            cids = np.unique(cNWP.clusterID[self.entryPress_freezed])
            for k in cids:
                mem = cNWP[k].members
                self.entryPress_freezed[mem] = False
                updateToDrainImbibe_single(k, mem, cNWP[k].neighbours, cNWP.toDrain, cNWP.toImbibe, 
                    cNWP.pcShrink, cNWP.pcMax, self.PcI, self.PcD, self.entryPress_freezed, self.totElements, 1e-3, 1e30)
        
    else:
        compute_fluxes_moles_pc_conc(self, cNWP)
        totTime += self.dt
        waitTime += self.dt
        
    if (cNWP.moles < 0.0).any():
        print('negative moles encountered !!!!')
        from IPython import embed; embed()
        

    return ii, totTime, totalTime, waitTime


def simulateOstRip(self, implicit=True, freshStart=True):
    print('Im in simulateOstRip  ', freshStart, self.cNWP.moles.sum())
    duration = 3600*24*2  # 24-hours
    #duration = 3600
    st = time()
    
    if not os.path.isfile('volarray_bent.dat'):
        np.savetxt('volarray_bent.dat', self.volarray)
    if not os.path.isfile('coordinates_bent.dat'):
        np.savetxt('coordinates_bent.dat', 
                    np.column_stack((self.x_array, self.y_array, self.z_array)))
        
    # records data in at most 10-mins intervals if no event occur
    #toShrink_elements = np.loadtxt('bent_toShrink_elements.dat', delimiter=',', dtype=np.int32)
    if hasattr(self, 'imposedP'):
        print(f"\nalpha={alpha}       imposedP={self.imposedP}        aqAvgPres={self.initialAvgPres}\
                H={self.H}      D={self.D}   \n")
    
    self.timesWithNoEvent = 0
    minTime = 600
    waitTime = 60.0
    print('$$$$$$$$$$$$$$$')
    if freshStart:
        self.j , self.ii, self.totTime, self.totalTime = 1, 1, 0.0, 0.0
        self.updateClust = False
        clust = self.cNWP
        clust.drainEvents, clust.imbEvents = 0, 0
        #from IPython import embed; embed()
        writeOnScreen(self, clust, self.ii, self.totTime, self.totalTime, st)
        writeData(self, round(self.totalTime,3))
        
    else:
        file_path = os.path.join(MEMORY_DIR, f"netsim_time_Dependent_61_mins.pkl")
        loaded_obj = joblib.load(file_path)
        do.updateObj(self, loaded_obj)
        clust = self.cNWP
        writeOnScreen(self, clust, self.ii, self.totTime, self.totalTime, st)
        temp.TempArrays.formTempClustArrays(clust.nClusters)
    
    timeToSave = minTime*self.j
    #self._factor = self.H*self.totElements*2000         # no cluster disappearance should raise avg aq pressure > 2000 Pa
    from IPython import embed; embed()
    #self.entryPress_freezed = np.zeros(self.totElements, dtype=np.bool_)
    #print('started')
    
    while self.totalTime < duration:
        #while totalTime < 25.33:
        try:
            self.ii, self.totTime, self.totalTime, waitTime = simulate(
                self, clust, self.ii, st, self.totTime, self.totalTime, minTime, waitTime, self.moles_tol, 
                self.pc_tol, self.max_iter)
            if self.totalTime > timeToSave:
                try:
                    filename = os.path.join(MEMORY_DIR, f"netsim_time_Dependent_{self.j}_mins.pkl")
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
    state_attrs = ['satW', 'rng', 'fluid', 'cWP', 'cNWP', '_dt',  'satList', 
        'H', 'Pc', 'RT', 'imposedP', 'D', 'len_tij_valid', 'elemToUpdateW', 
        'elemToUpdateNW', 'gasConc', 'aqueousMoles', 'minMoles', 'minConc', 
       'initialAvgPres', 'flux', 'netFlux', 'j', 'ii', 'totTime', 'totalTime',
       'moles_tol', 'pc_tol', 'max_iter', 'entryPress_freezed']

    state = {attr: getattr(self, attr) for attr in state_attrs}    
    joblib.dump(state, fname, compress=3)
    


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
	tList, TPValid, TValid, isCircle, totElements, nPores, nThroats):

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

		len_diff = 0.5*(LP1 + LT + LP2)
		area_min = min(areaP1, areaP2, areaT)
		lenEq = area_min/len_diff
		lenEq_array[i] = lenEq

	for i in prange(nValid):
		P, T = TPValid[i], TValid[i]
		if isCircle[P] or isCircle[T]: continue
		t = T - nPores - 1
		len_tij_valid[i] = lenEq_array[t] 

	return lenEq_array, len_tij_valid


