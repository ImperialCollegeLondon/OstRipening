import numpy as np
from time import time
import math
import os
import dill

import pnflowPy.tPhaseD as tPhaseD
import pnflowPy.tPhaseImb as tPhaseImb
import cluster as doClust
from network_numba import NetworkJit, ClusterJit
import OstRipening.temp as temp
from utilities_numba import *


MEMORY_DIR = "./ostrip_transient_avgPc_volweightedavgClustPc_impP_20000_D_1dot8minus9_H_6dot9minus6"
os.makedirs(MEMORY_DIR, exist_ok=True)
_temp = temp.TempArrays()

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
        network.imposedP = 17500 #imposedP    #1MPa
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
    
    # setting up arrays for cluster
    nClust = clust.nClusters
    nwID = np.flatnonzero(self.hasNWFluid)
    updateToDrainImbibe_numba(valkeys, clust.members, clust.neighbours, clust.toDrain, 
        clust.toImbibe, clust.pcShrink, clust.pcMax, self.PcI, self.PcD, self.Rarray,
        self.totElements, _temp.mem, True, pc_min, pc_max)
    clust.valClustD[:] = clust.valClust & (clust.toDrain<=self.totElements)

    # updating upper and lower values of volumes and moles for NW clusters
    #clust.updateVolumeNWCluster(nwID, valkeys, self)
    self.satList = np.zeros(self.totElements)
    self.satList[1:-1] = self.areaWPhase[1:-1]/self.areaSPhase[1:-1]
    clust.volume[valkeys] = np.bincount(
        clust.clusterID[self.hasNWFluid], (1-self.satList[self.hasNWFluid])*self.volarray[self.hasNWFluid])[valkeys]

    # print('Im on line 493!!!')
    # from IPython import embed; embed()
    # valkeysD = valkeys[clust.valClustD[valkeys]]
    # arrkeys = valkeysD[(self.PcI[clust.toDrain[valkeysD]]>clust.pcShrink[valkeysD]) & ()]
    # toDrain = clust.toDrain[arrkeys]
    # (self.PcI[toDrain]+self.imposedP)*(clust.volume[arrkeys]+self.volarray[toDrain])/clust.volume[arrkeys] - self.imposedP

    
    # adjust the cluster pc and updating moles of NW clusters
    #clust.pc[valkeys] = np.where(clust.valClustD[valkeys], clust.pcMax[valkeys], clust.pc[valkeys])
    clust.moles[valkeys] = ((self.imposedP + clust.pc[valkeys])*clust.volume[valkeys]/self.RT)
    #self.aqAvgPres = (clust.pc*clust.volume).sum()/clust.volume.sum() - 2000
    self.aqAvgPres = (clust.pc*clust.volume).sum()/clust.volume.sum() #self.Pc
    #self.aqAvgPres = 2000 #self.Pc
    self.oldaqAvgPres = self.aqAvgPres
    self.minConc = self.H*self.aqAvgPres
    self.minMoles = self.minConc*self.volarray
    #clust.updateMolesNWCluster(valkeys, self)
    
    # setting up arrays for elements
    self.elemToUpdateW = (self.fluid==0)
    self.elemToUpdateW[[-1,0]] = False                           # boolean
    self._elemToUpdateW = np.flatnonzero(self.elemToUpdateW).astype(np.int32)  # index
    self.elemToUpdateNW = self.hasNWFluid.copy()
    self._elemToUpdateNW = np.flatnonzero(self.elemToUpdateNW)           # index
    self.elemToUpdate = (self.elemToUpdateW|self.elemToUpdateNW)
    self._elemToUpdate = np.flatnonzero(self.elemToUpdate).astype(np.int32)   # index

    # initializing and updating moles and concentration for elements
    self.gasConc = np.zeros((self.totElements, 2))
    self.aqueousMoles = np.zeros(self.totElements)
    #from IPython import embed; embed()
    
    updateMolesConcElements(self, clust)
    self.flux = np.zeros(self.nValid)
    self.netFlux = np.zeros(self.totElements)
    self.netFluxClusters = np.zeros(nClust)
    self.netMoles = np.zeros(self.totElements)
    self.elemValidToLoseFlux = (self.gasConc[:,0] > self.minConc) & (self.aqueousMoles - self.minMoles > moles_tol)
    

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
    self.elemToUpdateW[arr] = cond
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



def computeFluxes(self, advection=False, moles_tol=1e-18):
    ''' compute flux for each element and cluster '''

    try:
        gasConc0 = self.gasConc[:, 0]
        self.flux = self.D*self.len_tij_valid*(gasConc0[self.TPValid] - gasConc0[self.TValid])
        indP, indT = np.flatnonzero(self.flux>0.0), np.flatnonzero(self.flux<0.0)
        arrP, arrT = self.TPValid[indP], self.TValid[indT]
        self.flux[indP] *= (self.aqueousMoles[arrP]-self.minMoles[arrP]>=moles_tol)
        self.flux[indT] *= (self.aqueousMoles[arrT]-self.minMoles[arrT]>=moles_tol)
        assert not advection
    except AssertionError:
        self.flux += (self.flowrateNW_TValid*self.aqueousMoles[self.upstreamElement]/
                    self.volarray[self.upstreamElement]*self.satList[self.upstreamElement])

    # # compute the net flux for each element
    self.netFlux[self.tList] = np.bincount(self.tValid, self.flux, self.nThroats)[1:]
    self.netFlux[self.poreListS] = np.bincount(self.TPValid, -self.flux, self.nPores+2)

    # compute the net flux for each cluster
    self.netFluxClusters = np.bincount(
        self.clusterNW_ID[self._elemToUpdateNW], self.netFlux[self._elemToUpdateNW],
        self.clusterNW.nClusters)


def computeMolesPcConc(self, moles_tol, pc_tol, max_iter):
    ''' compute and update the moles, pc and concentration of gas in each element and cluster'''

    determineTimeStep(self)
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
    clust.pc[keys] = np.maximum(
        (clust.moles[keys]*self.RT/clust.volume[keys]) - self.imposedP, 1e-3)
    self.gasConc[self._elemToUpdateNW,1] = self.H*clust.pc[
        clust.clusterID[self._elemToUpdateNW]]
    
    self.aqueousMoles[self._elemToUpdateW] += netMoles[self._elemToUpdateW]
    self.gasConc[self._elemToUpdateW, 1] = np.nan_to_num(
        self.aqueousMoles[self._elemToUpdateW]/self.volarray[self._elemToUpdateW])


def simulate(self, clust, ii, startTime, totTime, totalTime, minTime, moles_tol, pc_tol, max_iter):
    ''' main simulation loop for time-dependent Ostwald ripening '''
    self.updateClust = False
    cond1 = (clust.pc<clust.pcShrink)&clust.valClust
    cond2 = ~cond1 & (clust.pc>clust.pcMax)&clust.valClustD
    #cond2 = ~cond1 & (clust.moles>clust.molesMax)&clust.valClustD&(clust.pc>self.oldaqAvgPres)
    if not cond1[clust.valClust].any() and not cond2[clust.valClustD].any():
        computeFluxes(self)
        computeMolesPcConc(self, moles_tol, pc_tol, max_iter)
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
                print('$$:  ', keys, toDrain)
                #from IPython import embed; embed()
                toDrain = clust.growCluster(keys, self)
                clust.drainEvents += condGrowth.sum()
                updateLists(self, toDrain, 1)
                self.updateClust = False
        except:
            print('there is an error in the simulation, check line 1397!!!')
            from IPython import embed; embed()

        totalMoles = self.aqueousMoles[self._elemToUpdateW].sum()+clust.moles.sum()
        self.satW = ((self.satList[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()/
                        self.totVoidVolume)

        if not condShrink.any() and not condGrowth.any():
            self.timesWithNoEvent += 1
        else:
            self.timesWithNoEvent = 0

        self.aqAvgPres = (
            self.aqueousMoles[self.elemToUpdateW].sum()/
            (self.H*self.volarray[self.elemToUpdateW].sum()))
        gasAvgPres = (
            (clust.pc[clust.valClust]*clust.totalVolume[clust.valClust]).sum()/
            clust.totalVolume[clust.valClust].sum())
        
        print(("#:%8.6g  \tSimulation Time:%12.6g  \tActual runtime:%6.6g\
            \tTotal cluster moles:%8.6e  \tTotal dissolved moles:%6.6e\
            \tTotal gas moles:%4.10e \tAvg Pressure:%8.6e \tNo Shrinkage:%8.6g\
            \tNo Growth:%8.6g \tSat:%6.6g \tAqAvgPres:%6.6g" % (
            ii, round(totalTime,3), round(time()-startTime,3), clust.moles.sum(), 
            #self.dissolvedMoles.sum(), totalMoles,
            self.aqueousMoles.sum(), totalMoles,
            #clust.pc[self.valClust].mean(),
            gasAvgPres,
            clust.imbEvents, clust.drainEvents, self.satW, self.aqAvgPres)))
            
        writeData(self, round(totalTime,3))

        ii += 1
                
    self.gasConc[self.elemToUpdate,0] = self.gasConc[self.elemToUpdate,1]
        
    if (self.gasConc[:,1]<0.0).any():
        print('## a new element  concentration is negative!!!')
        from IPython import embed; embed()    
        
    return ii, totTime, totalTime


def simulateOstRip(self, implicit=True, freshStart=True):
    print('Im in simulateOstRip  ', freshStart)
    duration = 3600*24  # 24-hours
    st = time()
    
    if not os.path.isfile('volarray_bent.dat'):
        np.savetxt('volarray_bent.dat', self.volarray)
    if not os.path.isfile('coordinates_bent.dat'):
        np.savetxt('coordinates_bent.dat', 
                    np.column_stack((self.x_array, self.y_array, self.z_array)))
        
    # records data in at most 10-mins intervals if no event occur
    #toShrink_elements = np.loadtxt('bent_toShrink_elements.dat', delimiter=',', dtype=np.int32)
    self.timesWithNoEvent = 0
    minTime = 600
    print('$$$$$$$$$$$$$$$')
    if freshStart:
        j , ii, totTime, totalTime = 1, 1, 0.0, 0.0
        self.updateClust = False
        clust = self.clusterNW
        clust.drainEvents, clust.imbEvents = 0, 0
        totalMoles = (
            (self.volarray*self.gasConc[:,0])[~self.elemToUpdateNW].sum()+
            clust.moles.sum())
        gasAvgPres = (
                (clust.pc[clust.valClust]*clust.totalVolume[clust.valClust]).sum()/
                clust.totalVolume[clust.valClust].sum())
        print(("#:%8.6g  \tSimulation Time:%12.6g  \tActual runtime:%6.6g\
                \tTotal cluster moles:%8.6e  \tTotal dissolved moles:%6.6e\
                \tTotal gas moles:%6.10e \tAvg Pressure:%8.6e \tNo Shrinkage:%8.6g\
                \tNo Growth:%8.6g \tSat:%6.6g \tAqAvgPres:%6.6g" % (
            ii, round(totalTime,3), round(time()-st,3), clust.moles.sum(), 
            self.aqueousMoles.sum(), totalMoles, gasAvgPres,
            clust.imbEvents, clust.drainEvents, self.satW, self.aqAvgPres)))
        #from IPython import embed; embed()
        writeData(self, round(totalTime,3))
        
    else:
        #from IPython import embed; embed()
        with open(os.path.join(MEMORY_DIR, f"netsim_time_Dependent_15_hours.pkl"), "rb") as f:
            loaded_obj = dill.load(f)
            #self = dill.load(f)
        
        do.updateObj(self, loaded_obj)
        j, ii, totTime, totalTime = self.j, self.ii, self.totTime, self.totalTime
        print(j, ii, totTime, totalTime)
        clust = self.clusterNW
        temp.TempArrays.formTempClustArrays(clust.nClusters)

    #from IPython import embed; embed()
    #self.oldaqAvgPres = self.aqAvgPres
    timeToSave = 3600*j
    # while totalTime < duration:
    while totalTime < duration:
        try:
            ii, totTime, totalTime = simulate(
                self, clust, ii, st, totTime, totalTime, minTime, self.moles_tol, 
                self.pc_tol, self.max_iter)
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
            if self.timesWithNoEvent==3:
                print('times with no event = 3')
                break
                keys = np.where(clust.valClust)[0]
                pci = self.PcI[clust.toImbibe[keys]]
                keys = keys[(pci<self.oldaqAvgPres) & (pci > self.oldaqAvgPres-1000)]
                clust.pcShrink[keys] = self.PcI[clust.toImbibe[keys]]
                
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
        

            
            
 