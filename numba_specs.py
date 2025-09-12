from numba import types
from numba.types import int32, float64, boolean, void, Tuple


updateToDrainImbibe_spec = void(
    int32[:],        # arrkeys
    boolean[:, :],   # members
    boolean[:, :],   # neighbours
    int32[:],        # toDrainArray
    int32[:],        # toImbibeArray
    float64[:],      # pcShrinkArray
    float64[:],      # pcMaxArray
    float64[:],      # PcI
    float64[:],      # PcD
    int32,           # totElements
    boolean[:],      # tempMemArray
    boolean,         # parallel
    float64,         # pc_min
    float64          # pc_max
)

computeCornerArea_spec = void(
    int32[:],         # arr
    float64[:],       # newPc
    float64[:],       # cornA
    float64[:, :],    # m_halfAngles
    boolean[:, :],    # m_cornExists
    float64[:, :],    # m_initOrMaxPcHist
    float64[:, :],    # m_initOrMinApexDistHist
    float64[:, :],    # m_advPc
    float64[:, :],    # m_recPc
    float64[:, :],    # m_initedApexDist
    boolean[:],       # trappedW
    boolean[:],       # trappedNW
    float64[:],       # clusterW_pc
    float64[:],       # clusterNW_pc
    int32[:],         # clusterW_ID
    int32[:],         # clusterNW_ID
    float64[:],       # areaSPhase
    float64[:],       # maxCornerArea
    float64,          # sigma
    float64[:],       # thetaAdvAng
    float64[:],       # thetaRecAng
    float64,          # _delta
    boolean[:],       # isSquare
    boolean[:],       # isTriangle
    float64,          # muw
    float64,          # MOLECULAR_LENGTH
    boolean           # overideTrapping
)

computeCornerArea1_spec = void(
    int32[:],       # arr
    float64[:],     # newPc
    int32[:],       # clusterID
    float64[:],     # cornA
    float64[:, :],  # m_halfAngles
    float64[:, :],  # sinHalfAng
    float64[:, :],  # sinHalfAngcosHalfAng
    boolean[:, :],  # m_cornExists
    float64[:, :],  # m_initOrMaxPcHist
    float64[:, :],  # m_advPc
    float64[:, :],  # m_initedApexDist
    float64[:],     # areaSPhase
    float64[:],     # maxCornerArea
    float64,        # sigma
    float64[:],     # thetaAdvAng
    float64[:],     # thetaRecAng
    boolean[:],     # isSquare
    boolean[:],     # isTriangle
    float64         # half_pi
)

computeVolume_spec = float64[:](
    int32[:],          # keys
    int32[:],          # memkeys
    int32[:],          # memID
    float64[:],        # newPc
    float64[:],        # cornA
    float64[:, :],     # m_halfAngles
    boolean[:, :],     # m_cornExists
    float64[:, :],     # m_initOrMaxPcHist
    float64[:, :],     # m_initOrMinApexDistHist
    float64[:, :],     # m_advPc
    float64[:, :],     # m_recPc
    float64[:, :],     # m_initedApexDist
    boolean[:],        # trappedW
    boolean[:],        # trappedNW
    float64[:],        # clusterW_pc
    float64[:],        # clusterNW_pc
    int32[:],          # clusterW_ID
    int32[:],          # clusterNW_ID
    float64[:],        # areaSPhase
    float64[:],        # maxCornerArea
    float64[:],        # volarray (output array)
    float64,           # sigma
    float64[:],        # thetaAdvAng
    float64[:],        # thetaRecAng
    float64[:],        # PcD (capillary pressure for drainage)
    float64[:],        # criticalVolume
    float64[:],        # tempClustVol
    float64[:],        # tempMemVol
    float64[:],        # satList
    float64,           # _delta
    boolean[:],        # isSquare
    boolean[:],        # isTriangle
    float64,           # muw
    float64,           # MOLECULAR_LENGTH
    boolean,           # overidetrapping
    boolean            # updateSat
)

computeVolume_spec1 = float64[:](   # return type
    int32[:],          # keys
    int32[:],          # memkeys
    int32[:],          # memID
    float64[:],        # newPc
    float64[:],        # cornA
    float64[:, :],     # m_halfAngles
    float64[:, :],     # sinHalfAng
    float64[:, :],     # sinHalfAngcosHalfAng
    boolean[:, :],     # m_cornExists
    float64[:, :],     # m_initOrMaxPcHist
    float64[:, :],     # m_advPc
    float64[:, :],     # m_initedApexDist
    int32[:],          # clusterNW_ID
    float64[:],        # areaSPhase
    float64[:],        # maxCornerArea
    float64[:],        # volarray
    float64,           # sigma
    float64[:],        # thetaAdvAng
    float64[:],        # thetaRecAng
    float64[:],        # PcD
    float64[:],        # criticalVolume
    float64[:],        # tempClustVol
    float64[:],        # tempMemVol
    float64[:],        # satList
    boolean[:],        # isSquare
    boolean[:],        # isTriangle
    boolean,           # updateSat
    float64            # half_pi
)


computeVolume_single_spec = float64(
    int32[:],          # memID
    float64[:],        # newPc
    float64[:],        # cornA
    float64[:, :],     # m_halfAngles
    boolean[:, :],     # m_cornExists
    float64[:, :],     # m_initOrMaxPcHist
    float64[:, :],     # m_initOrMinApexDistHist
    float64[:, :],     # m_advPc
    float64[:, :],     # m_recPc
    float64[:, :],     # m_initedApexDist
    boolean[:],        # trappedW
    boolean[:],        # trappedNW
    float64[:],        # clusterW_pc
    float64[:],        # clusterNW_pc
    int32[:],          # clusterW_ID
    int32[:],          # clusterNW_ID
    float64[:],        # areaSPhase
    float64[:],        # maxCornerArea
    float64[:],        # volarray (output array)
    float64,           # sigma
    float64[:],        # thetaAdvAng
    float64[:],        # thetaRecAng
    float64[:],        # PcD (capillary pressure for drainage)
    float64[:],        # criticalVolume
    float64[:],        # tempMemVol
    float64[:],        # satList
    float64,           # _delta
    boolean[:],        # isSquare
    boolean[:],        # isTriangle
    float64,           # muw
    float64,           # MOLECULAR_LENGTH
    boolean,           # overidetrapping
    boolean            # updateSat
)


updateVolumeShrinkMax_spec = void(
    int32[:],        # arrkeys
    boolean[:, :],   # members
    boolean[:, :],   # neighbours
    int32[:],        # toDrainArray
    int32[:],        # toImbibeArray
    float64[:],      # pcShrinkArray
    float64[:],      # pcMaxArray
    boolean[:],      # valClust
    boolean[:],      # valClustD
    int32[:],        # clustSize
    int32[:],        # clusterID
    float64[:],      # volShrinkArray
    float64[:],      # volMaxArray
    float64[:],      # PcI
    float64[:],      # PcD
    float64[:],      # maxGasVolume
    float64[:],      # criticalVolume
    int32,           # totElements
    boolean[:],      # tempMemArray
    boolean,         # parallel
    float64,         # pc_min
    float64          # pc_max
)


updateEventsThreshold_spec = void(
    int32[:],        # arrkeys
    boolean[:, :],   # members
    boolean[:, :],   # neighbours
    int32[:],        # toDrainArray
    int32[:],        # toImbibeArray
    float64[:],      # pcShrinkArray
    float64[:],      # pcMaxArray
    boolean[:],      # valClust
    boolean[:],      # valClustD
    int32[:],        # clustSize
    int32[:],        # clusterID
    float64[:],      # volShrinkArray
    float64[:],      # volMaxArray
    float64[:],      # molesShrinkArray
    float64[:],      # molesMaxArray
    float64[:],      # molesArray
    float64[:],      # totalVolume
    float64[:],      # volarray
    float64[:],      # PcI
    float64[:],      # PcD
    float64[:],      # maxGasVolume
    float64[:],      # criticalVolume
    int32,           # totElements
    float64,         # imposedP
    float64,         # RT
    float64,         # H
    boolean[:],      # tempMemArray
    boolean,         # parallel
    float64,         # pc_min
    float64          # pc_max
)


updateMolesShrinkGrowth_spec = void(
    int32[:],     # arrkeys
    float64[:],   # pcMaxArray
    float64[:],   # pcShrinkArray
    float64[:],   # volMaxArray
    float64[:],   # volShrinkArray
    float64[:],   # molesShrinkArray
    float64[:],   # molesMaxArray
    int32[:],     # toImbibeArray
    float64[:],   # totalVolume
    float64[:],   # volarray
    float64,      # imposedP
    float64,      # RT
    float64       # H
)

findPc_spec = void(
    int32[:],          # keys
    int32[:],          # memkeys
    int32[:],          # memID
    float64[:],        # targetMoles
    float64[:, :],     # m_halfAngles
    boolean[:, :],     # m_cornExists
    float64[:, :],     # m_initOrMaxPcHist
    float64[:, :],     # m_initOrMinApexDistHist
    float64[:, :],     # m_advPc
    float64[:, :],     # m_recPc
    float64[:, :],     # m_initedApexDist
    boolean[:],      # trappedW
    boolean[:],      # trappedNW
    float64[:],      # clusterW_pc
    float64[:],      # clusterNW_pc
    int32[:],        # clusterW_ID
    int32[:],        # clusterNW_ID
    float64[:],      # clusterVolume
    float64[:],      # areaSPhase
    float64[:],      # maxCornerArea
    float64[:],      # volarray
    float64,         # sigma
    float64[:],      # thetaAdvAng
    float64[:],      # thetaRecAng
    float64[:],      # PcD
    float64[:],      # criticalVolume
    float64[:],      # satList
    float64[:],      # tempClustVol
    float64[:],      # tempMemVol
    float64,         # _delta
    boolean[:],      # isSquare
    boolean[:],      # isTriangle
    float64[:],      # totalVolume
    float64[:],      # cornA
    float64,         # muw
    float64,         # MOLECULAR_LENGTH
    float64,         # imposedP
    float64,         # RT
    float64,         # H
    float64,         # moles_tol
    float64,         # pc_tol
    int32,           # max_iter
    boolean[:],      # notdone
    float64[:],      # pc1
    float64[:],      # pc2
    float64[:],      # pcNext
    float64          # eps
)


findPc1_spec = void(
    int32[:],        # keys
    int32[:],        # memkeys
    int32[:],        # memID
    float64[:],      # targetMoles
    float64[:, :],   # m_halfAngles
    float64[:, :],   # sinHalfAng
    float64[:, :],   # sinHalfAngcosHalfAng
    boolean[:, :],   # m_cornExists
    float64[:, :],   # m_initOrMaxPcHist
    float64[:, :],   # m_advPc
    float64[:, :],   # m_initedApexDist
    float64[:],      # clusterVolume
    int32[:],        # clusterNW_ID
    float64[:],      # clusterNW_pc
    float64[:],      # areaSPhase
    float64[:],      # maxCornerArea
    float64[:],      # volarray
    float64,         # sigma
    float64[:],      # thetaAdvAng
    float64[:],      # thetaRecAng
    float64[:],      # PcD
    float64[:],      # criticalVolume
    float64[:],      # satList
    float64[:],      # tempClustVol
    float64[:],      # tempMemVol
    boolean[:],      # isSquare
    boolean[:],      # isTriangle
    float64[:],      # totalVolume
    float64[:],      # cornA
    float64,         # imposedP
    float64,         # RT
    float64,         # H
    float64,         # moles_tol
    float64,         # pc_tol
    int32,           # max_iter
    boolean[:],      # notdone
    float64[:],      # pc1
    float64[:],      # pc2
    float64[:],      # pcNext
    float64,         # eps
    float64          # half_pi
)

findPc_single_spec = void(
    int32,            # key
    int32[:],         # memID
    float64[:],       # targetMoles
    float64[:, :],    # m_halfAngles
    boolean[:, :],    # m_cornExists
    float64[:, :],    # m_initOrMaxPcHist
    float64[:, :],    # m_initOrMinApexDistHist
    float64[:, :],    # m_advPc
    float64[:, :],    # m_recPc
    float64[:, :],    # m_initedApexDist
    boolean[:],       # trappedW
    boolean[:],       # trappedNW
    float64[:],       # clusterW_pc
    float64[:],       # clusterNW_pc
    int32[:],         # clusterW_ID
    int32[:],         # clusterNW_ID
    float64[:],       # clusterVolume
    float64[:],       # areaSPhase
    float64[:],       # maxCornerArea
    float64[:],       # volarray
    float64,          # sigma
    float64[:],       # thetaAdvAng
    float64[:],       # thetaRecAng
    float64[:],       # PcD
    float64[:],       # criticalVolume
    float64[:],       # satList
    float64[:],       # tempMemVol
    float64,          # _delta
    boolean[:],       # isSquare
    boolean[:],       # isTriangle
    float64[:],       # totalVolume
    float64[:],       # cornA
    float64,          # muw
    float64,          # MOLECULAR_LENGTH
    float64,          # imposedP
    float64,          # RT
    float64,          # H
    float64,          # moles_tol
    float64,          # pc_tol
    int32,            # max_iter
    float64[:],       # pc1
    float64[:],       # pc2
    float64[:],       # pcNext
    float64           # eps
)


updateClusterPcVolume_spec = void(
	int32[:],	   # memID
    int32[:],      # keys
    float64[:],    # targetMoles
    float64[:],    # cornA
    float64[:, :], # m_halfAngles
    boolean[:, :], # m_cornExists
    float64[:, :], # m_initOrMaxPcHist
    float64[:, :], # m_initOrMinApexDistHist
    float64[:, :], # m_advPc
    float64[:, :], # m_recPc
    float64[:, :], # m_initedApexDist
    boolean[:],    # trappedW
    boolean[:],    # trappedNW
    float64[:],    # clusterW_pc
    float64[:],    # clusterNW_pc
    float64[:],    # clusterVolume
    int32[:],      # clusterW_ID
    int32[:],      # clusterNW_ID
    float64[:],    # totalVolume
    float64[:],    # areaSPhase
    float64[:],    # maxCornerArea
    float64[:],    # volarray
    float64,       # sigma
    float64[:],    # thetaAdvAng
    float64[:],    # thetaRecAng
    float64[:],    # PcD
    float64[:],    # criticalVolume
    float64[:],    # satList
    float64[:],    # tempClustVol
    float64[:],    # tempMemVol
    float64,       # _delta
    boolean[:],    # isSquare
    boolean[:],    # isTriangle
    float64,       # muw
    float64,       # MOLECULAR_LENGTH
    float64,       # imposedP
    float64,       # RT
    float64,       # H
    float64,       # moles_tol
    float64,       # pc_tol
    int32,         # max_iter
    boolean[:],    # notdone
    float64[:],    # pc1
    float64[:],    # pc2
    float64[:],     # pcNext
    float64          # eps
)


updateClusterPcVolume1_spec = void(
    int32[:],        # memID
    int32[:],        # keys
    float64[:],      # targetMoles
    float64[:],      # cornA
    float64[:, :],   # m_halfAngles
    float64[:, :],   # sinHalfAng
    float64[:, :],   # sinHalfAngcosHalfAng
    boolean[:, :],   # m_cornExists
    float64[:, :],   # m_initOrMaxPcHist
    float64[:, :],   # m_advPc
    float64[:, :],   # m_initedApexDist
    float64[:],      # clusterVolume
    int32[:],        # clusterNW_ID
    float64[:],      # clusterNW_pc
    float64[:],      # totalVolume
    float64[:],      # areaSPhase
    float64[:],      # maxCornerArea
    float64[:],      # volarray
    float64,         # sigma
    float64[:],      # thetaAdvAng
    float64[:],      # thetaRecAng
    float64[:],      # PcD
    float64[:],      # criticalVolume
    float64[:],      # satList
    float64[:],      # tempClustVol
    float64[:],      # tempMemVol
    boolean[:],      # isSquare
    boolean[:],      # isTriangle
    float64,         # imposedP
    float64,         # RT
    float64,         # H
    float64,         # moles_tol
    float64,         # pc_tol
    int32,           # max_iter
    boolean[:],      # notdone
    float64[:],      # pc1
    float64[:],      # pc2
    float64[:],      # pcNext
    float64,         # eps
    float64          # half_pi
)

fillAdjoiningPores_spec = Tuple((int32[:], int32[:]))(
    int32[:],       # keys
    int32[:],       # toDrain
    boolean[:],     # isPore
    int32[:, :],    # TPConnections
    int32,          # nPores
    boolean[:],     # done
    int32[:],       # tempID
    boolean[:]      # hasFluid
)

volume_bincount_spec = float64[:](
    int32[:],   # memID
    int32[:],     # keys
    int32[:],     # clustID
    float64[:],   # areaPhase
    float64[:],   # areaSPhase
    float64[:]    # volarray
)

mapOldNewKeys_spec = Tuple((int32[:], int32[:]))(
    int32[:],       # oldkeys
    int32[:],       # newkeys
    boolean[:]      # notdone
)