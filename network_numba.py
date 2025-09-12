from numba import njit, int32, float64, boolean, prange
from numba.experimental import jitclass
import numpy as np


network_spec = [
    ('fluid', int32[:]),
    ('PcI', float64[:]),
    ('PcD', float64[:]),
    ('isSquare', boolean[:]),
    ('isTriangle', boolean[:]),
    ('isCircle', boolean[:]),
    ('areaSPhase', float64[:]),
    ('volarray', float64[:]),
    ('toInBdr', boolean[:]),
    ('toOutBdr', boolean[:]),
    ('isinsideBox', boolean[:]),
    ('connected', boolean[:]),
    ('poreList', int32[:]),
    ('throatList', int32[:]),
    ('tList', int32[:]),
    ('conTToIn', int32[:]),
    ('conTToOut', int32[:]), 
    ('connectivity_graph', int32[:]),
    ('cg_offsets', int32[:]),
    ('m_halfAngles', float64[:, :]),
    ('totElements', int32),
    ('nPores', int32),
    ('nThroats', int32),
    ('sigma', float64),
    ('_delta', float64),
    ('MOLECULAR_LENGTH', float64),
    ('m_cornExists', boolean[:, :]),
    ('m_initOrMaxPcHist', float64[:, :]),
     ('m_initOrMinApexDistHist', float64[:, :]),
    ('m_advPc', float64[:, :]),
    ('m_recPc', float64[:, :]),
    ('m_initedApexDist', float64[:, :]),
    ('PistonPcAdv', float64[:]),
    ('PistonPcRec', float64[:]),
    ('maxCornerArea', float64[:]),
    ('maxGasVolume', float64[:]),
    ('thetaAdvAng', float64[:]),
    ('thetaRecAng', float64[:]),
    ('contactAng', float64[:]),
    ('LTarray', float64[:]),
    ('TPConnections', int32[:, :]),
    ('isPore', boolean[:]),
    ('trappedW', boolean[:]),
    ('trappedNW', boolean[:]),
    ('clusterW_ID', int32[:]),
    ('clusterNW_ID', int32[:]),
    ('clusterW_pc', float64[:]),
    ('clusterNW_pc', float64[:]),
    ('imposedP', float64),
    ('muw', float64),
    ('satList', float64[:]),
    ('gasConc', float64[:, :]),
    ('aqueousMoles', float64[:]),
    ('criticalVolume', float64[:]),
    ('H', float64),
    ('R', float64),
    ('T', float64),
    ('Pc', float64),
    ('tempArea', float64[:]),
    ('tempVol', float64[:]),
    ('tempMem', boolean[:]),
    ('tempMemP', boolean[:]),
    ('tempID', int32[:]),
    ('RT', float64),
    ('_RT_H', float64),
    ('_impP_RT', float64),
    ('sinBeta', float64[:, :]),
    ('cosBeta', float64[:, :]),
    ('sinBetacosBeta', float64[:, :]),
    ('_factor', float64[:, :]),
    ('_part', float64[:, :]),
]


@jitclass(network_spec)
class NetworkJit:
    def __init__(self, isSquare, isTriangle, isCircle, areaSPhase, volarray,
        toInBdr, toOutBdr, isinsideBox, connected,  poreList, throatList, tList, 
        conTToIn, conTToOut, connectivity_graph, cg_offsets, m_halfAngles, 
        totElements, nPores, nThroats, sigma, _delta, MOLECULAR_LENGTH, 
        PistonPcAdv, PistonPcRec, thetaAdvAng, thetaRecAng, contactAng, LTarray,
        TPConnections, isPore, 
        trappedW, trappedNW, clusterW_ID, clusterNW_ID, clusterW_pc, clusterNW_pc, 
        imposedP, muw, H, R, T, Pc):
        
        self.fluid = np.zeros(totElements, dtype=int32)
        self.PcI = np.zeros(totElements, dtype=float64)
        self.PcD = np.zeros(totElements, dtype=float64)
        self.isSquare = isSquare
        self.isTriangle = isTriangle
        self.isCircle = isCircle
        self.areaSPhase = areaSPhase
        self.volarray = volarray
        self.toInBdr = toInBdr
        self.toOutBdr = toOutBdr
        self.isinsideBox = isinsideBox
        self.connected = connected
        self.poreList = poreList
        self.throatList = throatList
        self.tList = tList
        self.conTToIn = conTToIn
        self.conTToOut = conTToOut
        self.connectivity_graph = connectivity_graph
        self.cg_offsets = cg_offsets
        self.m_halfAngles = m_halfAngles
        self.totElements = totElements
        self.nPores = nPores
        self.nThroats = nThroats
        self.sigma = sigma
        self._delta = _delta
        self.MOLECULAR_LENGTH = MOLECULAR_LENGTH

        self.m_cornExists = np.zeros(self.m_halfAngles.shape, dtype=boolean)
        self.m_initOrMaxPcHist = np.zeros_like(self.m_halfAngles)
        self.m_initOrMinApexDistHist = np.zeros_like(self.m_halfAngles)
        self.m_advPc = np.zeros_like(self.m_halfAngles)
        self.m_recPc = np.zeros_like(self.m_halfAngles)
        self.m_initedApexDist = np.zeros_like(self.m_halfAngles)
        self.PistonPcAdv = PistonPcAdv
        self.PistonPcRec = PistonPcRec
        self.maxCornerArea = np.zeros(totElements, dtype=float64)
        self.maxGasVolume = np.zeros(totElements, dtype=float64)
        self.thetaAdvAng = thetaAdvAng
        self.thetaRecAng = thetaRecAng
        self.contactAng = contactAng
        self.LTarray = LTarray
        self.TPConnections = TPConnections
        self.isPore = isPore
        self.trappedW = trappedW.view(boolean)
        self.trappedNW = trappedNW.view(boolean)
        self.clusterW_ID = clusterW_ID.view(int32)
        self.clusterNW_ID = clusterNW_ID.view(int32)
        self.clusterW_pc = clusterW_pc.view(float64)
        self.clusterNW_pc = clusterNW_pc.view(float64)
        self.imposedP = imposedP
        self.muw = muw
        self.satList = np.zeros(totElements, dtype=float64)
        self.gasConc = np.zeros((totElements, 2), dtype=float64)
        self.aqueousMoles = np.zeros(totElements, dtype=float64)
        self.criticalVolume = np.zeros(self.totElements, dtype=float64)
        self.H = H
        self.R = R
        self.T = T
        self.Pc = Pc

        self.tempArea = np.zeros(totElements, dtype=float64)
        self.tempVol = np.zeros(totElements, dtype=float64)
        self.tempMem = np.zeros(totElements, dtype=boolean)
        self.tempMemP = np.zeros(nPores, dtype=boolean)
        self.tempID = np.zeros(nPores, dtype=int32)
        self.RT = R*T
        self._impP_RT = imposedP/self.RT
        self._RT_H = (1/self.RT - H)
        self.sinBeta = np.sin(self.m_halfAngles)
        self.cosBeta = np.cos(self.m_halfAngles)
        self.sinBetacosBeta = self.sinBeta*self.cosBeta
        numerator = (sigma*np.cos(thetaAdvAng.reshape(-1,1)+m_halfAngles)).ravel()
        sinBeta = self.sinBeta.ravel()
        mask = (sinBeta != 0.0)
        self._factor = np.zeros_like(m_halfAngles)
        self._factor.ravel()[mask] = numerator[mask]/sinBeta[mask]
        self._part = np.zeros_like(self.m_halfAngles)

        
       

cluster_spec = [
    ('members', boolean[:, :]),
    ('neighbours', boolean[:, :]),
    ('toImbibe', int32[:]),
    ('toDrain', int32[:]),
    ('pc', float64[:]), 
    ('pcShrink', float64[:]),
    ('pcMax', float64[:]), 
    ('volume', float64[:]), 
    ('volShrink', float64[:]), 
    ('volMax', float64[:]), 
    ('totalVolume', float64[:]),
    ('criticalPc', float64[:]),
    ('criticalVolume', float64[:]),
    ('criticalMoles', float64[:]),
    ('moles', float64[:]), 
    ('molesShrink', float64[:]), 
    ('molesMax', float64[:]), 
    ('clustSize', int32[:]),
    ('clusterID', int32[:]),
    ('valClust', boolean[:]),
    ('valClustD', boolean[:]),
    ('n_exponents', float64[:]),
    ('hasFluid', boolean[:]),
    ('nClusters', int32),
    ('dVg_dPc', float64[:]),
    ('drainEvents', int32),
    ('imbEvents', int32),
    ('trappedStatus', boolean[:]),
    ('connected', boolean[:]),
    ('trapped', boolean[:]),
    ('conn', boolean[:]),
]


@jitclass(cluster_spec)
class ClusterJit:
    def __init__(self, nClusters, totElements):
        self.members = np.zeros((nClusters, totElements), dtype=boolean)
        self.neighbours = np.zeros((nClusters, totElements), dtype=boolean)
        self.toImbibe = np.full(nClusters, totElements+1, dtype=int32)
        self.toDrain = np.full(nClusters, totElements+1, dtype=int32)
        self.pc = np.zeros(nClusters, dtype=float64)
        self.pcShrink = np.full(nClusters, 1.0e-30, dtype=float64)
        self.pcMax = np.full(nClusters, 1.0e30, dtype=float64)
        self.volume = np.zeros(nClusters, dtype=float64)
        self.volShrink = np.full(nClusters, 1.0e-30, dtype=float64)
        self.volMax = np.full(nClusters, 1.0e30, dtype=float64)
        self.totalVolume = np.full(nClusters, 1.0e30, dtype=float64)
        self.criticalPc = np.zeros(nClusters, dtype=float64)
        self.criticalVolume = np.zeros(nClusters, dtype=float64)
        self.criticalMoles = np.zeros(nClusters, dtype=float64)
        self.moles = np.zeros(nClusters, dtype=float64)
        self.molesShrink = np.full(nClusters, 1.0e-30, dtype=float64)
        self.molesMax = np.full(nClusters, 1.0e30, dtype=float64)
        self.clustSize = np.zeros(nClusters, dtype=int32)
        self.clusterID = np.full(totElements, -5, dtype=int32)
        self.valClust = np.zeros(nClusters, dtype=boolean)
        self.valClustD = np.zeros(nClusters, dtype=boolean)
        self.n_exponents = np.zeros(nClusters, dtype=float64)
        self.hasFluid = np.zeros(totElements, dtype=boolean)
        self.nClusters = nClusters
        self.dVg_dPc = np.zeros(nClusters, dtype=float64)

        self.drainEvents = 0
        self.imbEvents = 0
        self.trappedStatus = np.zeros(nClusters, dtype=boolean)
        self.connected = np.zeros(nClusters, dtype=boolean)
        self.trapped = np.zeros(totElements, dtype=boolean)
        self.conn = np.zeros(totElements, dtype=boolean)
        
       

        
