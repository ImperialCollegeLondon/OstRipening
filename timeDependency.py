import numpy as np
from time import time
import math
import os

#from clustering import Cluster
from analytical_numerical_study import plotClass


class TimeDependency:
    def __init__(self, obj, Pc, T=298, imposedP=1e6, H=7.8e-6, D=7.3e-9, 
                 steps=10, dt=0.0005, adjustTime=False):
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
        
    
    def simulateOstRip(self, implicit=True):
        duration = 3600*24*2  # 24-hours
        st = time()
        totalTime, totTime = 0.0, 0.0
        self.netFlux = np.zeros(self.totElements)
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

        from IPython import embed; embed()
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
        

            
            
            

    

    



            

        
    
        



