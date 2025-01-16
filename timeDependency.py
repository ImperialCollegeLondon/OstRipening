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
        obj.H = H  #mol m-3 pa-1
        self.D = D  #m2/s
        self.num = 1
        obj.R = 8.314  #J/mol.K
        obj.T = T    #K
        obj.imposedP = imposedP #imposedP    #1MPa
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
        self.maxCornerArea = np.maximum(self.cornerArea, self.areaWPhase)
        self.initialPcArr = np.ones(self.totElements)*self.Pc
        self.initialPcArr[self.hasNWFluid] = self.clusterNW.pc[self.clusterNW_ID[self.hasNWFluid]]

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
        clust.satList[[-1,0]] = 0.0
        self.initialCornerArea= self.cornerArea.copy()
        
        clust.volume = np.bincount(
            self.clusterNW_ID[self.hasNWFluid], self.volarrayNW[self.hasNWFluid], clust.pc.size)
        clust.totalVolume  = np.bincount(
            self.clusterNW_ID[self.hasNWFluid], self.volarray[self.hasNWFluid], clust.pc.size)
        
        # print(':::::::::::@@@@@>>>>>><<<<<<<')
        # from IPython import embed; embed()
        # self.obj.satAtMaxPc = np.nan_to_num(self.minCornerArea/self.areaSPhase)
        # # clust.avgSatList = np.nan_to_num(self.minCornerArea/self.areaSPhase)
        # cornArea =  np.minimum(
        #     np.mean((self.minCornerArea, self.cornerArea), axis=0), self.areaWPhase)
        # clust.avgSatList = np.nan_to_num(cornArea/self.areaSPhase)
        # clust.avgSatList = np.nan_to_num(self.areaWPhase/self.areaSPhase)
        # clust.avgSatList[clust.avgSatList==0.0] = np.nan_to_num(
        #     self.minCornerArea/self.areaSPhase)[clust.avgSatList==0]

        # clust.avgSatList = np.zeros(self.totElements)
        # clust.avgSatList[self.fluid==0] =  np.nan_to_num(
        #     (self._cornArea+self.minCornerArea)/(2*self.areaSPhase))[self.fluid==0]
        # clust.avgSatList[self.fluid==1] = np.nan_to_num(
        #     (self.minCornerArea+self.areaWPhase)/(2*self.areaSPhase))[self.fluid==1] 
        # self.maxNWVolarray = (1-clust.avgSatList)*self.volarray
        
        
        # setting up arrays for elements
        self.elemToUpdateW = (self.fluid==0)
        self.elemToUpdateW[[-1,0]] = False                           # boolean
        self._elemToUpdateW = self.elementListS[self.elemToUpdateW]  # index
        self.elemToUpdateNW = self.hasNWFluid.copy()
        self._elemToUpdateNW = self.elementListS[self.elemToUpdateNW]           # index
        self.elemToUpdate = self.elementListS[self.elemToUpdateW|self.elemToUpdateNW]   # index


    def initializingPcMolesNWCluster(self, clust):
        ''' initialize moles, pc and gas conc for  NW cluster '''
        nClustNW = self.clusterNW.pc.size
        clust.toDrain = np.ones(nClustNW, dtype=int)*(self.totElements+1)
        clust.toImbibe = np.ones(nClustNW, dtype=int)*(self.totElements+1)
        
        #clust.volShrink = np.zeros(nClustNW)
        clust.pcMax = np.full(nClustNW, 1e30)
        clust.pcShrink = np.full(nClustNW, -1e30)
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
        self.updateVolumeShrinkMax(clust, validKeys)
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
        
        # if (clust.volShrink[validKeys]>clust.volume[validKeys]).any():
        #     print('::::::::::::::@@@@@@@@@@::::::::::::::::')
        #     from IPython import embed; embed()
        
        
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
        self.valClust = (self.clusterNW.size>0)
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

        if self.dt<=0:
            print('I am in dt:  {self.dt}')
            from IPython import embed; embed()

       

    def computeMolesPcConc(self):
        ''' compute and update the moles, pc and concentration of gas in each element and cluster'''
        
        self.determineTimeStep()
        clust = self.clusterNW

        ''' update element filled with WP properties '''
        self.dissolvedMoles[self._elemToUpdateW] = np.clip(
            (self.dissolvedMoles+self.netFlux*self.dt)[self._elemToUpdateW], 0.0, None)
        
        # try:
        #     assert (self.dissolvedMoles[self._elemToUpdateW]>=0.0).all()
        # except AssertionError:
        #     print('some dissolved moles are less than zero!!!')
        #     from IPython import embed; embed
        #self.dissolvedMoles[self._elemToUpdateW] = dissolvedMoles

        self.gasConc[self._elemToUpdateW, 1] = np.nan_to_num(
            self.dissolvedMoles[self._elemToUpdateW]/self.volarray[self._elemToUpdateW])       
        
        ''' update NWP cluster properties '''
        clust.moles = (clust.moles+self.netFluxClusters*self.dt)
        cond1 = (clust.moles[self.toUpdateNW]>self.initialMoles[self.toUpdateNW])
        cond2 = (clust.moles[self.toUpdateNW]<self.initialMoles[self.toUpdateNW])
        toUpdateNW1 = self.toUpdateNW[cond1]
        toUpdateNW2 = self.toUpdateNW[cond2]
        #toUpdateNW = self.toUpdateNW #[cond1|cond2]

        clust.pc[toUpdateNW1] = (clust.moles[toUpdateNW1]-self.initialMoles[toUpdateNW1])/(
            clust.molesMax[toUpdateNW1]-self.initialMoles[toUpdateNW1])*(
                clust.pcMax[toUpdateNW1]-self.initialPc[toUpdateNW1]) + self.initialPc[toUpdateNW1]
        clust.pc[toUpdateNW2] = (clust.moles[toUpdateNW2]-clust.molesShrink[toUpdateNW2])/(
            self.initialMoles[toUpdateNW2]-clust.molesShrink[toUpdateNW2])*(
                self.initialPc[toUpdateNW2]-clust.pcShrink[toUpdateNW2]) + clust.pcShrink[
                    toUpdateNW2]
        if any(clust.pc<0.0):
            print('::::::::::>>>>>>>>>>>>>>>>>')
            from IPython import embed; embed()
        
        # clust.volume[toUpdateNW1] = np.minimum(
        #     (clust.moles[toUpdateNW1]-self.initialMoles[toUpdateNW1])/
        #     (clust.molesMax[toUpdateNW1]-self.initialMoles[toUpdateNW1])*
        #     (clust.volMax[toUpdateNW1]-self.initialVolume[toUpdateNW1]) +
        #     self.initialVolume[toUpdateNW1], clust.volMax[toUpdateNW1])
        
        # clust.volume[toUpdateNW2] = np.maximum(
        #     (clust.moles[toUpdateNW2]-clust.molesMin[toUpdateNW2])/
        #     (self.initialMoles[toUpdateNW2]-clust.molesMin[toUpdateNW2])*
        #     self.initialVolume[toUpdateNW2], 0.0)
        #print('::::::::::::*****&&&&&&&&&£££££££')
        #from IPython import embed; embed()
        # num = (clust.moles[toUpdateNW]-(
        #     self.imposedP*clust.volume[toUpdateNW]/(self.R*self.T)))
        # den = ((clust.volume[toUpdateNW]/(self.R*self.T)) + 
        #        (self.H*(clust.totalVolume[toUpdateNW]-clust.volume[toUpdateNW])))
        # clust.pc[toUpdateNW] = num/den
        
        try:
            assert (clust.pc<self.maxPc).all()
        except AssertionError:
            print('there is an unrealistic pc, check line 427!!!')
            from IPython import embed; embed()

        

        # update conc and moles but check if the cluster gas becomes completely dissolved
        #cond = self.valClust & ((self.netFluxClusters<0.0)|(clust.moles>=clust.molesMax))
        gasConcClusters = self.H*clust.pc
        #elemToUpdateNW = self._elemToUpdateNW[cond[self.clusterNW_ID[self._elemToUpdateNW]]]
        elemToUpdateNW = self.elemToUpdateNW
        self.gasConc[elemToUpdateNW,1] = gasConcClusters[self.clusterNW_ID[elemToUpdateNW]]
        # self.gasConc[self.elemToUpdateNW,1] = gasConcClusters[
        #     self.clusterNW_ID[self.elemToUpdateNW]]

    
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
        #nRow = int(duration/minTime*1.5)+1
        # gasConc = np.memmap('gasConcOstRipening_bent.dat', dtype='float32', 
        #                     mode='w+', shape=(nRow,self.totElements))
        # clustPc = np.memmap('clustPcOstRipening_bent.dat', dtype=float,
        #                     mode='w+', shape=(nRow, self.clusterNW.pc.size))
        # clustVol = np.memmap('clustVolOstRipening_bent.dat', dtype=float,
        #                     mode='w+', shape=(nRow, self.clusterNW.pc.size))
        # clustMoles = np.memmap('clustMolesOstRipening_bent.dat', dtype=float,
        #                     mode='w+', shape=(nRow, self.clusterNW.pc.size))
        # clustID = np.memmap('clustIDOstRipening_bent.dat', dtype=float,
        #                     mode='w+', shape=(nRow, self.clusterNW_ID.size))
        # clustPcString = ','.join(map(str, self.clusterNW.pc))
        # clustVolString = ','.join(map(str, self.clusterNW.volume))
        # clustMolesString = ','.join(map(str, self.clusterNW.moles))
        # clustIDString = ','.join(map(str, self.clusterNW_ID))
        #timeArray = str(0.0)+',\n'
        # timeArray = np.zeros(nRow)

        
        #gasConc[0] = self.gasConc[:, 0]
        # clustPc[0] = self.clusterNW.pc
        # clustVol[0] = self.clusterNW.volume
        # clustMoles[0] = self.clusterNW.moles
        # with open('clustPcOstRipening_bent.dat', 'a') as f:
        #     np.savetxt(f, [self.clusterNW.pc], delimiter=',', fmt='%g')
        # with open('clustVolOstRipening_bent.dat', 'a') as f:
        #     np.savetxt(f, [self.clusterNW.volume], delimiter=',', fmt='%g')
        # with open('clustMolesOstRipening_bent.dat', 'a') as f:
        #     np.savetxt(f, [self.clusterNW.moles], delimiter=',', fmt='%g')
        # with open('clustIDOstRipening_bent.dat', 'a') as f:
        #     np.savetxt(f, [self.clusterNW_ID], delimiter=',', fmt='%g')
        #self.writeData(totalTime)
        #clustID[0] = self.clusterNW_ID
        ii = 1
        self.clusterNW.drainEvents, self.clusterNW.imbEvents = 0, 0
        
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
                self.computeFluxes()
                self.computeMolesPcConc()
            except AssertionError:
                self.dt = 0.0
                self.updateClust = True
                self.gasConc[self.elemToUpdate,1] = self.gasConc[self.elemToUpdate,0]
                

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
                    # if any((self.clusterNW.molesShrink>self.clusterNW.moles)&self.valClust):
                    #     print('a cluster may need further shrinking, check line 515!!!')
                    #     input('wattttt!')
                        #from IPython import embed; embed()
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

                # with open('cluster_data_pc_ini_3000.csv', 'a') as f:
                #     np.savetxt(f, [self.clusterNW.pc], delimiter=',', fmt='%g')
                # clustPc[ii] = self.clusterNW.pc
                # clustVol[ii] = self.clusterNW.volume
                # clustMoles[ii] = self.clusterNW.moles
                # clustID[ii] = self.clusterNW_ID
                # clustPcString += ('\n'+','.join(map(str, self.clusterNW.pc)))
                # clustVolString += ('\n'+','.join(map(str, self.clusterNW.volume)))
                # clustMolesString += ('\n'+','.join(map(str, self.clusterNW.moles)))
                # clustIDString += ('\n'+','.join(map(str, self.clusterNW_ID)))
                #self.writeData(round(totalTime,3))
                #timeArray += (str(totalTime)+',\n')
                
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

    def writeData(self, totalTime):
        with open('clustPcOstRipening_bent.dat', 'a') as f:
            np.savetxt(f, [self.clusterNW.pc], delimiter=',', fmt='%g')
        with open('clustVolOstRipening_bent.dat', 'a') as f:
            np.savetxt(f, [self.clusterNW.volume], delimiter=',', fmt='%g')
        with open('clustMolesOstRipening_bent.dat', 'a') as f:
            np.savetxt(f, [self.clusterNW.moles], delimiter=',', fmt='%g')
        with open('clustIDOstRipening_bent.dat', 'a') as f:
            np.savetxt(f, [self.clusterNW_ID], delimiter=',', fmt='%g')
        with open('dissolvedMolesOstRipening_bent.dat', 'a') as f:
            np.savetxt(f, [self.dissolvedMoles], delimiter=',', fmt='%g')
        with open('timeArrayOstRipening_bent.dat', 'a') as f:
            f.write(str(totalTime)+',')
        with open('numberEventsOstRipening_bent.dat', 'a') as f:
            np.savetxt(f, [[self.clusterNW.imbEvents, self.clusterNW.drainEvents]], 
                       delimiter=',', fmt='%g')
        


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
        

            
            
            

    

    



            

        
    
        



