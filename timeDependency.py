import numpy as np
from time import time
import math
import os
from scipy.optimize import bisect

from analytical_numerical_study import plotClass

class TimeDependency:
    
    def __init__(self, obj, Pc, T=298, imposedP=1e6, H=7.8e-6, D=7.3e-9, 
                 steps=10, dt=0.0005, adjustTime=False):
        global __clust
        self.obj = obj

        # initialising the parameters
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
        
        # setting up the arrays for elements ...
        self.len_tij = self.LenEq()
        self.gasConc = np.zeros([self.totElements, 2], dtype='float')
        self.dissolvedMoles = np.zeros(self.totElements, dtype='float')
        self.delta_nMoles = np.zeros(self.totElements, dtype='float')
        self.obj.satList = np.zeros(self.totElements)
        self.obj.satList[1:-1] = self.areaWPhase[1:-1]/self.areaSPhase[1:-1]
        self.obj.volarrayW = self.satList*self.volarray
        self.obj.volarrayNW = (1-self.satList)*self.volarray
        self.obj.satAtMaxPc = np.nan_to_num(self.minCornerArea/self.areaSPhase)

        # resize the cluster arrays
        self.resizeClusters(self.clusterW, self.clusterNW)
        
        # setting up useful arrays
        self.settingUpArrays(self.clusterNW)

        # initializing moles, pc and concentration for NW cluster
        self.initializingPcMolesNWCluster(self.clusterNW)
        
        # initializing moles, pc and concentration for W/NW pores/throats
        self.netFlux = np.zeros(self.totElements)
        self.initializingMolesConcElements()

        # initialize the flow rates of the phases
        self.initializeFlowrate()

        # for writing data/results
        self.resultsP_str = "# Step,totalFlux,total_delta_nMoles,Sw, \
            #cluster_growth,#cluster_shrinkage,AvgPressure,totMoles_W,totMoles_NW,totMoles"
        self.__fileName__()
        print('Im in timeDependency!')

        #from IPython import embed; embed()
    
    def __getattr__(self, name):
        return getattr(self.obj, name)
    
    def resizeClusters(self, clustW, clustNW):
        ''' resize the W/NW clusters by removing empty clusters id at the end '''

        nClustW = self.clusterW_ID.max()+1
        clustW.members = clustW.members[:nClustW]
        clustW.pc = clustW.pc[:nClustW]
        clustW.clustConToInlet = clustW.clustConToInlet[:nClustW]
        clustW.connected= clustW.connected[:nClustW]
        clustW.keys = clustW.keys[:nClustW]
        clustW.availableID.clear()

        nClustNW = self.clusterNW_ID.max()+1
        clustNW.members = clustNW.members[:nClustNW]
        clustNW.pc = clustNW.pc[:nClustNW]
        clustNW.clustConToInlet = clustNW.clustConToInlet[:nClustNW]
        clustNW.connected = clustNW.connected[:nClustNW]
        clustNW.keys = clustNW.keys[:nClustNW]
        clustNW.availableID.clear()

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
        clust.satList[[-1,0]] = 0.0
        clust.cornerArea = self.cornerArea.copy()
        clust.neighbours = np.zeros(
            [self.clusterNW_ID.max()+1, self.totElements], dtype=bool)
        
        clust.volume = np.bincount(
            self.clusterNW_ID[self.hasNWFluid], self.volarrayNW[self.hasNWFluid])
        self.totalVolClusters  = np.bincount(
            self.clusterNW_ID[self.hasNWFluid], self.volarray[self.hasNWFluid])
        self.maxNWVolarray = np.nan_to_num(
            (self.areaSPhase-self.minCornerArea)/self.areaSPhase*self.volarray)
        clust.updateNeighMatrix()
        
        # setting up arrays for elements
        self.elemToUpdateW = (self.fluid==0)
        self.elemToUpdateW[[-1,0]] = False                           # boolean
        self._elemToUpdateW = self.elementListS[self.elemToUpdateW]  # index
        self.elemToUpdateNW = self.hasNWFluid.copy()
        self._elemToUpdateNW = self.elementListS[self.elemToUpdateNW]           # index
        self.elemToUpdate = self.elementListS[self.elemToUpdateW|self.elemToUpdateNW]   # index


    def initializingPcMolesNWCluster(self, clust):
        ''' initialize moles, pc and gas conc for  NW cluster '''
        nClustNW = self.clusterNW_ID.max()+1
        clust.toDrain = np.ones(nClustNW, dtype=int)*(self.totElements+1)
        clust.toImbibe = np.ones(nClustNW, dtype=int)*(self.totElements+1)
        
        #clust.volShrink = np.zeros(nClustNW)
        clust.pcMax = np.full(nClustNW, 1e30)
        clust.pcShrink = np.full(nClustNW, -1e30)
        clust.molesMax = np.full(nClustNW, 1e30)      # maximum expected value
        clust.molesMin = np.zeros(nClustNW)      # least expected value
        clust.volMax = np.zeros(nClustNW)
        validKeys = np.where(self.valClust)[0]
        [clust.updateToDrainImbibe(k, self) for k in validKeys]

        # initializing cluster moles and concentrations
        self.initialPc = clust.pc.copy()
        self.initialVolume = clust.volume.copy()
        #print('Im on line 149!!!')
        #from IPython import embed; embed()
        #clust.avgSatList = self._cornArea/self.areaSPhase
        clust.avgSatList = np.nan_to_num(
            (self._cornArea+self.areaWPhase)/(2*self.areaSPhase)) #take an average for now
        clust.moles = (
            (self.imposedP+self.initialPc)*clust.volume/(self.R*self.T)
            + self.H*self.initialPc*(self.totalVolClusters-clust.volume)
            )
        self.initialMoles = clust.moles.copy()
        #self.clusterNW.molesShrink = (self.H*self.imposedP*self.totalVolClusters)
        self.clusterNW.molesShrink = (self.H*self.initialPc*self.totalVolClusters)
        self.betaLow = np.zeros(nClustNW)
        self.betaHigh = np.zeros(nClustNW)
        clust.updateHighLow(validKeys, self, False)


    def initializingMolesConcElements1(self):
        # moles of gas in elements filled with WPhase are all updated
        # but only members of NWPhase clusters not connected across 
        # the network are updated.
        
        ''' initialize the brine to have conc equivalent to Pc!!! '''
        from IPython import embed; embed()
        #avgPres = 5000
        avgPres = 3000
        #avgPres = 0
        self.gasConc[self._elemToUpdateW,0] = self.H*avgPres
        #self.assignConcToAqElements()
        self.dissolvedMoles[self._elemToUpdateW] = (
           self.gasConc[self._elemToUpdateW,0]*self.volarray[self._elemToUpdateW])
        
        ''' initialize the NW elements according to their trapped Pc!!! '''
        self.gasConc[self._elemToUpdateNW,0] = self.H*self.clusterNW.pc[
            self.clusterNW_ID[self._elemToUpdateNW]]
        
        # self.gasConc[self._elemToUpdateNW,0] = (
        #     (self.clusterNW.moles/self.totalVolClusters)[self.clusterNW_ID[self._elemToUpdateNW]])

    def initializingMolesConcElements(self):
        # moles of gas in elements filled with WPhase are all updated
        # but only members of NWPhase clusters not connected across 
        # the network are updated.
        
        ''' initialize the NW elements according to their trapped Pc!!! '''
        self.gasConc[self._elemToUpdateNW,0] = self.H*self.clusterNW.pc[
            self.clusterNW_ID[self._elemToUpdateNW]]
        
        ''' initialize the brine to have conc equivalent to Pc!!! '''
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@')
        # from IPython import embed; embed()
        # self.aqAvgPres = bisect(self._funcPc, 0, self.clusterNW.pc[self.valClust].mean())
        self.computeAqAvgPres()
        self.gasConc[self._elemToUpdateW,0] = self.H*self.aqAvgPres
        #self.assignConcToAqElements()
        self.dissolvedMoles[self._elemToUpdateW] = (
            self.gasConc[self._elemToUpdateW,0]*self.volarray[self._elemToUpdateW])
        

    def _funcPc(self, avgPres):
        self.gasConc[self._elemToUpdateW,0] = self.H*avgPres
        flux = self.len_tij_valid*(
                self.gasConc[self.TPValid, 0] - self.gasConc[self.TValid, 0])
        return (flux[self.elemToUpdateNW[self.TValid]].sum()-
            flux[self.elemToUpdateNW[self.TPValid]].sum())
    
    def computeAqAvgPres(self):
        cond1 = (self.elemToUpdateNW[self.TPValid]&self.elemToUpdateW[self.TValid])
        cond2 = (self.elemToUpdateW[self.TPValid]&self.elemToUpdateNW[self.TValid])

        a=(self.len_tij_valid*self.clusterNW.pc[self.clusterNW_ID[self.TPValid]])[cond1].sum()
        b=(self.len_tij_valid*self.clusterNW.pc[self.clusterNW_ID[self.TValid]])[cond2].sum()
        c=self.len_tij_valid[cond1].sum()
        d=self.len_tij_valid[cond2].sum()
        self.aqAvgPres = ((a+b)/(c+d))
    
    def findNextT(self, arrP, notdone, done):
        arrT = self.PTConnections[arrP][self.PTValid[arrP]]
        arrT = arrT[notdone[arrT]]
        done[arrT] = True
        notdone[arrT] = False
        return arrT
    
    def findNextP(self, arrT, notdone, done):
        arrP = self.TPConnections[arrT][self.TPCond[arrT]]
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
        #gwL = self.do.computegL(self.gWPhase)
        gnwL = self.do.computegL(self.gNWPhase)
        #self.flowrateW, self.flowDirectionW = self.do.computeFlowrate(gwL, 0, self.Pc, True)
        self.flowrateNW, self.flowDirectionNW = self.do.computeFlowrate(gnwL, 1, self.Pc, True)
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
                     self.volarrayW[self.upstreamElement])
        
        # compute the net flux for each element
        self.netFlux[self.tList] = np.bincount(self.tValid, flux, self.nThroats)
        self.netFlux[self.poreListS] = np.bincount(self.TPValid, -flux, self.nPores+2)

        # compute the net flux for each cluster
        self.netFluxClusters = np.bincount(
            self.clusterNW_ID[self._elemToUpdateNW], self.netFlux[self._elemToUpdateNW],
            len(self.clusterNW.keys))
        
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
            self.updateClust = True
        except ValueError:
            pass
        condC = (clusterMoles>self.clusterNW.molesMax)&self.valClust
        try:
            self.dt = (-(self.clusterNW.molesMax[condC]-self.clusterNW.moles[condC])/
                        self.netFluxClusters[condC]).min()
            self.updateClust = True
        except ValueError:
            pass
        if self.dt<0:
            print('I am in dt:  {self.d}')
            from IPython import embed; embed()

       

    def computeMolesPcConc(self):
        ''' compute and update the moles, pc and concentration of gas in each element and cluster'''
        # try:
        #     clusterMoles = (self.clusterNW.moles+self.netFluxClusters*self._dt)
        #     assert (clusterMoles>self.clusterNW.molesShrink)[self.valClust].all()
        #     self.dt = self._dt
        #     try:
        #         assert not (clusterMoles>=self.clusterNW.molesMax)[self.valClust].any()
        #     except AssertionError:
        #         self.updateClust = True
        # except AssertionError:
        #     condC = (clusterMoles<=self.clusterNW.molesShrink)&self.valClust
        #     if (self.clusterNW.moles[condC]<=self.clusterNW.molesShrink[condC]).any():
        #         self.dt = 0.0
        #     else:
        #         self.dt = max(
        #             1e-6, (-(self.clusterNW.moles[condC]-self.clusterNW.molesShrink[condC])/
        #                 self.netFluxClusters[condC]).min())
        #         # if self.dt==1e-6:
        #         #     print(self.dt)
        #         #     from IPython import embed; embed()
            
        #     clusterMoles = (self.clusterNW.moles+self.netFluxClusters*self.dt)
        #     self.updateClust = True
            
        
        # try:
        #     dissolvedMoles = (self.dissolvedMoles[self._elemToUpdateW]+
        #         self.netFlux[self._elemToUpdateW]*self.dt)
        #     assert (dissolvedMoles>=0.0).all()
        # except AssertionError:
        #     print('Mole is negative at element level,  try a smaller time step!!!')
        #     from IPython import embed; embed()

        self.determineTimeStep()

        ''' update element filled with WP properties '''
        self.dissolvedMoles[self._elemToUpdateW] = (
            self.dissolvedMoles+self.netFlux*self.dt)[self._elemToUpdateW]
        #self.dissolvedMoles[self._elemToUpdateW] = dissolvedMoles

        # self.gasConc[self._elemToUpdateW, 1] = (
        #     dissolvedMoles/self.volarray[self._elemToUpdateW])       
        
        ''' update NWP cluster properties '''
        self.clusterNW.moles = (self.clusterNW.moles+self.netFluxClusters*self.dt)
        #self.clusterNW.moles = clusterMoles
        toUpdateNW1 = self.toUpdateNW[self.clusterNW.moles[self.toUpdateNW]>
                                        self.initialMoles[self.toUpdateNW]]
        toUpdateNW2 = self.toUpdateNW[self.clusterNW.moles[self.toUpdateNW]<
                                        self.initialMoles[self.toUpdateNW]]
        
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
        from IPython import embed; embed()
        
        newPc = (self.betaHigh[toUpdateNW1]*(
            self.clusterNW.moles[toUpdateNW1]-self.initialMoles[toUpdateNW1]) +
            self.initialPc[toUpdateNW1])
        self.clusterNW.pc[toUpdateNW1] = newPc
        try:
            assert not any(np.isinf(newPc))
        except AssertionError:
            toUpdateNW1b = toUpdateNW1[np.isinf(newPc)]
            self.clusterNW.pc[toUpdateNW1b] = (
                (self.initialPc[toUpdateNW1b]/self.initialMoles[toUpdateNW1b])*
                 self.clusterNW.moles[toUpdateNW1b])
        
        self.clusterNW.volume[toUpdateNW1] = np.minimum(
            (self.clusterNW.moles[toUpdateNW1]-self.initialMoles[toUpdateNW1])/
            (self.clusterNW.molesMax[toUpdateNW1]-self.initialMoles[toUpdateNW1])*
            (self.clusterNW.volMax[toUpdateNW1]-self.initialVolume[toUpdateNW1]) +
            self.initialVolume[toUpdateNW1], self.clusterNW.volMax[toUpdateNW1])
        
        self.clusterNW.pc[toUpdateNW2] = (self.betaLow[toUpdateNW2]*(
            self.clusterNW.moles[toUpdateNW2]-self.clusterNW.molesShrink[toUpdateNW2]) +
            self.clusterNW.pcShrink[toUpdateNW2])
        
        self.clusterNW.volume[toUpdateNW2] = np.maximum(
            (self.clusterNW.moles[toUpdateNW2]-self.clusterNW.molesMin[toUpdateNW2])/
            (self.initialMoles[toUpdateNW2]-self.clusterNW.molesMin[toUpdateNW2])*
            self.initialVolume[toUpdateNW2], 0.0)
        
        try:
            assert (self.clusterNW.pc<1e6).all()
        except AssertionError:
            print('there is an unrealistic pc, check line 427!!!')
            from IPython import embed; embed()

        
         # update conc and moles but check if the cluster gas becomes completely dissolved
        gasConcClusters = self.H*self.clusterNW.pc
        self.gasConc[self._elemToUpdateNW,1] = gasConcClusters[
            self.clusterNW_ID[self._elemToUpdateNW]]

    
    def simulateOstRip(self, implicit=True):
        duration = 3600*24*2  # 24-hours
        st = time()
        totalTime, totTime = 0.0, 0.0
        self.updateClust = False

        if not os.path.isfile('volarray_bent.dat'):
            np.savetxt('volarray_bent.dat', self.volarray)
        if not os.path.isfile('coordinates_bent.dat'):
            np.savetxt('coordinates_bent.dat', 
                       np.column_stack((self.x_array, self.y_array, self.z_array)))
            
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
        ind = np.array([813,1103,1197,1213,1245,1448,1581,1711,1716,1720,1749,1791,1792,1803,1835,1877,1899,1904,1906,1910,1924,1951,1979,1998,2002,2003,2011,2015,2052,2057,2059,2061,2076,2144,2147,2148,2160,2162,2182,2198,2203,2213,2220,2229,2233,2260,2271,2272,2273,2275,2277,2280,2282,2283,2292,2304,2306,2307,2310,2322,2327,2330,2332,2334,2338,2355,2356,2357,2362])
        totalVolume = self.volarray[self.elemToUpdate].sum()
        
        def _fff(ii, totTime, totalTime):
            self.valClustD = self.valClust & (self.clusterNW.toDrain<=self.totElements)
            try:
                # check for further shrinkage!!!
                assert (self.clusterNW.moles>self.clusterNW.molesShrink)[self.valClust].all()
                self.computeAqAvgPres()
                self.gasConc[self._elemToUpdateW,0] = self.H*self.aqAvgPres
                self.dissolvedMoles[self._elemToUpdateW] = (
                    self.gasConc[self._elemToUpdateW,0]*self.volarray[self._elemToUpdateW])
                self.computeFluxes()
                self.computeMolesPcConc()
            except AssertionError:
                self.dt = 0.0
                self.updateClust = True
            try:
                totTime += self.dt
                #print(self.dt, totTime)
                assert self.updateClust or (totTime>minTime)     #check for every second simulation
                condShrink = (self.clusterNW.moles<=self.clusterNW.molesShrink) & self.valClust
                condGrowth = (self.clusterNW.moles>=self.clusterNW.molesMax) & self.valClustD
                totalTime += totTime
                totTime = 0.0

                try:
                    assert condShrink.any()
                    keys = np.array(self.clusterNW.keys)[condShrink]
                    toImbibe = self.clusterNW.toImbibe[keys]
                    print('@@:   ', keys, toImbibe)
                    self.clusterNW.shrinkCluster(keys, self)
                    self.clusterNW.imbEvents += condShrink.sum()
                    self.updateLists(toImbibe)
                    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    # from IPython import embed; embed()
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
                    self.updateLists(toDrain, 1)
                    self.updateClust = False
                except AssertionError:
                    pass

                totalMoles = (self.volarray*self.gasConc[:,1]).sum()
                
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

                with open('cluster_data_pc_ini_3000.csv', 'a') as f:
                    np.savetxt(f, [self.clusterNW.pc], delimiter=',', fmt='%g')

                #gasConc[ii] = self.gasConc[:, 1] 
                #clustPc[ii] = self.clusterNW.pc
                #timeArray[ii] = totalTime
                ii += 1
            except AssertionError: 
                pass
                        
            
            self.gasConc[self.elemToUpdate,0] = self.gasConc[self.elemToUpdate,1]
            return ii, totTime, totalTime

        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        from IPython import embed; embed()
        with open('cluster_data_pc_ini_3000.csv', 'a') as f:
            np.savetxt(f, [ind], delimiter=',', fmt='%g')
            np.savetxt(f, [self.clusterNW.pc[ind]], delimiter=',', fmt='%g' )

        
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
        

            
            
            

    

    



            

        
    
        



