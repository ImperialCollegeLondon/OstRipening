import numpy as np
from time import time
import math

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
        self.adjustTime = adjustTime

        # for test5D
        initialize_test5D(self)

        # setting up the arrays for elements ...
        self.len_tij = self.LenEq()
        self.gasConc = np.zeros([self.totElements, 2], dtype='float')
        self.dissolvedMoles = np.zeros(self.totElements, dtype='float')
        self.delta_nMoles = np.zeros(self.totElements, dtype='float')
        self.obj.satList = np.zeros(self.totElements)
        self.obj.satList[1:-1] = self.areaWPhase[1:-1]/self.areaSPhase[1:-1]

        # resize the cluster arrays
        nClustW = self.clusterW_ID.max()+1
        self.clusterW.members = self.clusterW.members[:nClustW]
        self.clusterW.pc = self.clusterW.pc[:nClustW]
        self.clusterW.clustConToInlet = self.clusterW.clustConToInlet[:nClustW]
        self.clusterW.keys = self.clusterW.keys[:nClustW]
        self.clusterW.availableID.clear()

        nClustNW = self.clusterNW_ID.max()+1
        self.clusterNW.members = self.clusterNW.members[:nClustNW]
        self.clusterNW.pc = self.clusterNW.pc[:nClustNW]
        self.clusterNW.clustConToInlet = self.clusterNW.clustConToInlet[:nClustNW]
        self.clusterNW.keys = self.clusterNW.keys[:nClustNW]
        self.clusterNW.availableID.clear()
        
        # setting up useful arrays
        self.TPCond = (self.TPConnections[1:]>0) # location of valid pores connected to each throats
        self.TPValid = self.TPConnections[1:][self.TPCond] # valid pores connected to each throats
        self.valClust = (self.clusterNW.members.any(axis=1)) # returns valid clusters

        # setting up arrays for clusters
        self.clusterNW.satList = 1-self.satList
        self.clusterNW.satList[[-1,0]] = 0.0
        self.clusterNW.neighbours = np.zeros([nClustNW, self.totElements], dtype=bool)
        self.clusterNW.volume = np.bincount(
            self.clusterNW_ID[self.hasNWFluid], 
            self.volarray[self.hasNWFluid]) #*self.clusterNW.satList[self.hasNWFluid])
        self.clusterNW.moles = (self.imposedP+self.clusterNW.pc)*self.clusterNW.volume/(
            self.R*self.T)
        #from IPython import embed; embed()
        self.clusterNW.updateNeighMatrix()

        # update the toDrain and toImbibe arrays
        self.clusterNW.toDrain = np.ones(nClustNW, dtype=int)*(self.totElements+1)
        self.clusterNW.toImbibe = np.ones(nClustNW, dtype=int)*(self.totElements+1)
        self.clusterNW.pcHigh = np.full(nClustNW, np.inf)
        self.clusterNW.pcLow = np.full(nClustNW, -np.inf)
        validKeys = np.where(self.valClust)[0]
        [updateToDrainImbibe(k, self) for k in validKeys]
        condD = (self.clusterNW.toDrain<self.totElements)
        condI = (self.clusterNW.toImbibe<self.totElements)
        self.clusterNW.pcHigh[condD] = self.PcD[self.clusterNW.toDrain[condD]]
        self.clusterNW.pcLow[condI] = self.PcI[self.clusterNW.toImbibe[condI]]
        
        #nitial and boundary condition
        self.gasConc[self.hasNWFluid,0] = self.H*self.clusterNW.pc[
            self.clusterNW_ID[self.hasNWFluid]]
        self.dissolvedMoles[self.hasNWFluid] = self.gasConc[self.hasNWFluid,0]*self.volarray[
            self.hasNWFluid] #*self.clusterNW.satList[self.hasNWFluid]
        
        # determine the time step value
        try:
            assert adjustTime
            term = self.D*self.len_tij
            volP1P2 = (term/self.volarray[self.TPConnections[1:]])[self.TPCond] # using pore volume
            volT = (term/self.volarray[self.tList][:,np.newaxis])[self.TPCond] # using throat volume
            dt1, dt2 = 1/volP1P2.max(), 1/volT.max()
            _dt = round_down_to_n_sf((dt1+dt2)/2, 1)
            self.dt = _dt
        except AssertionError:
            self.dt = dt
    
        '''moles of gas in elements filled with WPhase are all updated
        but only members of NWPhase clusters not connected to the inlet are updated.'''
        self.elemToUpdateW = (self.fluid==0)
        self.elemToUpdateW[[-1,0]] = False
        clustToUpdateNW = self.valClust & (~self.clusterNW.clustConToInlet)
        self.elemToUpdateNW = np.zeros(self.totElements, dtype=bool)
        self.elemToUpdateNW[self.hasNWFluid] = clustToUpdateNW[self.clusterNW_ID[self.hasNWFluid]]
        self.clustToUpdateNW = np.array(self.clusterNW.keys)[clustToUpdateNW]
        self.elemToUpdate = self.elemToUpdateW|self.elemToUpdateNW

        # for writing data/results
        self.resultsP_str = "# Step,totalFlux,total_delta_nMoles,Sw, \
            #cluster_growth,#cluster_shrinkage,AvgPressure,totMoles_W,totMoles_NW,totMoles"
        self.__fileName__()
        print('Im in timeDependency!')
    
    def __getattr__(self, name):
        return getattr(self.obj, name)
    
    def LenEq(self):
        term1 = self.Rarray[self.TPConnections[1:]]/self.areaSPhase[self.TPConnections[1:]]
        term1[np.isnan(term1)] = 0.0
        term2 = self.LTarray/(2*self.areaSPhase[self.tList])
        return 1/(term1+term2[:,np.newaxis]) 

    def computeMoles(self):
        # compute flux and number of moles
        flux = self.D*self.len_tij*(self.gasConc[self.TPConnections[1:],0] - self.gasConc[
            self.tList,0][:,np.newaxis])
        
        # comput the net flux for each element
        netFlux = np.zeros(self.totElements)
        netFlux[self.tList] = flux.sum(axis=1, where=self.TPCond)
        netFlux[:self.nPores+1] = np.bincount(
            self.TPValid, -flux[self.TPCond], self.nPores+1)
        
        # determine a suitable time step so as to avoid a negative updated moles in any element
        try:
            assert self.adjustTime
            cond = (netFlux<0.0) & self.elemToUpdate
            assert cond.any()
            self._dt = min(self.dt, min(-self.dissolvedMoles[cond]/netFlux[cond]))
        except AssertionError:
            self._dt = self.dt
    
        self.delta_nMoles = netFlux*self._dt
        self.delta_nMoles_clusters = np.bincount(
            self.clusterNW_ID[self.elemToUpdateNW], self.delta_nMoles[self.elemToUpdateNW],
            len(self.clusterNW.keys))        

    def updateConcentration(self):
        ''' update element properties '''
        self.dissolvedMoles[self.elemToUpdateW] += self.delta_nMoles[self.elemToUpdateW]
       
        self.gasConc[self.elemToUpdateW, 1] = self.dissolvedMoles[
            self.elemToUpdateW]/self.volarray[self.elemToUpdateW]

        ''' update cluster properties '''
        newMoles = (self.clusterNW.moles[self.clustToUpdateNW] + 
                    self.delta_nMoles_clusters[self.clustToUpdateNW])
        self.clusterNW.moles[self.clustToUpdateNW] = newMoles
        try:
            cond = (self.dissolvedMoles[self.elemToUpdate]<0.0).any() or (newMoles<0.0).any()
            assert cond
            print('mole is negative,  try a smaller time step!!!!')
            exit()
        except AssertionError:
            pass

        self.clusterNW.pc[self.clustToUpdateNW] = (
            newMoles*self.R*self.T/self.clusterNW.volume[self.clustToUpdateNW] - self.imposedP)
        self.gasConc[self.elemToUpdateNW, 1] = (
            self.H*self.clusterNW.pc[self.clusterNW_ID[self.elemToUpdateNW]])
        self.dissolvedMoles[self.elemToUpdateNW] = (
            self.gasConc[self.elemToUpdateNW,1]*
            self.volarray[self.elemToUpdateNW]) #*self.clusterNW.satList[self.elemToUpdateNW])
        self.gasConc[~self.elemToUpdate,1] = self.gasConc[~self.elemToUpdate,0]
        
    
    def simulateOstRip(self, implicit=True):
        steps = 200000 #4000000
        st = time()
        totalTime = 0.0
        initialClusterMoles = self.clusterNW.moles
        sat = (self.satList[self.isinsideBox]*self.volarray[
                self.isinsideBox]).sum()/self.totVoidVolume
        
        gasConc = np.memmap('gasConcOstRipening_test5D.dat', dtype='float32', 
                            mode='w+', shape=(steps,self.totElements))
        cnt=0
        for i in range(1,steps+1):
            self.clusterNW.drainEvents, self.clusterNW.imbEvents = 0, 0
            self.computeMoles()            
            self.updateConcentration()
            condShrink = ((self.clusterNW.moles<initialClusterMoles) & 
                              (self.clusterNW.pc<self.clusterNW.pcLow))
            condGrowth = ((self.clusterNW.moles>initialClusterMoles) & 
                              (self.clusterNW.pc>self.clusterNW.pcHigh))
            try:
                assert condShrink.any() or condGrowth.any()
                print('There is a shrinkage and/or growth!!!')
                from IPython import embed; embed()
            except:
                pass
            #print(self.gasConc)
            gasConc[i-1] = self.gasConc[:, 1]
            totalTime += self._dt
            try:
                cnt += 1
                assert cnt==500
                print('#Iterations: %10.6g  \tSimulation Time: %12.6g  \tActual runtime: %12.6g  \tTotal gas moles: %8.6e  \tTotal dissolved moles:%12.6e \tAvg Pressure:%12.6e' % (
                i, round(totalTime,3), round(time()-st,3), self.clusterNW.moles.sum(), 
                self.dissolvedMoles[self.hasWFluid].sum(), self.clusterNW.pc.mean()))
                cnt=0
            except AssertionError:
                pass
                
            self.gasConc[:,0] = self.gasConc[:,1]
            #input('waittt')
    
        
        from IPython import embed; embed()


def round_down_to_n_sf(x, n):
    ''' round down x to n significant figures '''
    if x == 0:
        return 0
    factor = 10 ** (math.floor(math.log10(abs(x))) - (n - 1))
    return math.floor(x / factor) * factor

def updateToDrainImbibe(k, self):
    ''' updates the toDrain and toImbibe arrays'''
    condD, condI = self.clusterNW.neighbours[k], self.clusterNW.members[k]
    try:
        self.clusterNW.toImbibe[k] = self.elementListS[condI][np.argmax(self.PcI[condI])]
        self.clusterNW.toDrain[k] = self.elementListS[condD][np.argmin(self.PcD[condD])]
    except ValueError:
        pass
    return

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
    self.clusterNW.clustConToInlet[1] = True
    self.clusterW_ID[1:-1] = 1
    self.clusterW_ID[self.fluid==1] = -5
    self.clusterNW_ID[:] = -5
    self.clusterNW_ID[self.fluid==1] = 1
    dt = 1
        

            
            
            

    

    



            

        
    
        



