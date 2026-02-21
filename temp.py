import numpy as np
from pnflowPy.temp import TempArrays as BaseTempArrays


class TempArrays(BaseTempArrays):
	
	@classmethod
	def formTempClustArrays(cls, nClusters):
		cls.nClusters = nClusters
		cls.guessPc = np.zeros(nClusters, dtype=np.float32)
		cls.guessPcL = np.zeros(nClusters, dtype=np.float32)
		cls.guessPcH = np.zeros(nClusters, dtype=np.float32)
		cls.guessMoles = np.zeros(nClusters, dtype=np.float64)
		cls.guessMolesL = np.zeros(nClusters, dtype=np.float64)
		cls.guessMolesH = np.zeros(nClusters, dtype=np.float64)
		cls.guessVol = np.zeros(nClusters, dtype=np.float64)
		cls.guessVolL = np.zeros(nClusters, dtype=np.float64)
		cls.guessVolH = np.zeros(nClusters, dtype=np.float64)
		cls.resMoles = np.zeros(nClusters, dtype=np.float64)
		cls.resMolesL = np.zeros(nClusters, dtype=np.float64)
		cls.resMolesH = np.zeros(nClusters, dtype=np.float64)
		cls.resCond = np.zeros(nClusters, dtype=np.bool_)
		cls.tempCID = np.zeros(nClusters, dtype=np.int32)

	@classmethod
	def updateTempClustArrays(cls, nClusters):
		size = nClusters - cls.nClusters
		cls.nClusters = nClusters
		cls.guessPc = np.concatenate((cls.guessPc,	 np.zeros(size)))
		cls.guessPcL = np.concatenate((cls.guessPcL,  np.zeros(size)))
		cls.guessPcH = np.concatenate((cls.guessPcH,  np.zeros(size)))
		cls.guessMoles = np.concatenate((cls.guessMoles,  np.zeros(size)))
		cls.guessMolesL = np.concatenate((cls.guessMolesL, np.zeros(size)))
		cls.guessMolesH = np.concatenate((cls.guessMolesH, np.zeros(size)))
		cls.guessVol = np.concatenate((cls.guessVol,  np.zeros(size)))
		cls.guessVolL = np.concatenate((cls.guessVolL, np.zeros(size)))
		cls.guessVolH = np.concatenate((cls.guessVolH, np.zeros(size)))
		cls.resMoles = np.concatenate((cls.resMoles,  np.zeros(size)))
		cls.resMolesL = np.concatenate((cls.resMolesL, np.zeros(size)))
		cls.resMolesH = np.concatenate((cls.resMolesH, np.zeros(size)))
		cls.resCond = np.concatenate((cls.resCond,	 np.zeros(size, dtype=np.bool_)))
		cls.tempCID =  np.concatenate((cls.tempCID, np.zeros(size, dtype=np.int32)))
	
	@classmethod
	def formTempNetworkArrays(cls, nPores, nThroats, totElements, conn_graph_size, halfAng):
		BaseTempArrays.formTempNetworkArrays(nPores, totElements, conn_graph_size)
		cls.done = np.zeros(totElements, dtype=np.bool_)
		cls.filterNext = np.zeros(totElements, dtype=np.bool_)
		cls.mList = np.zeros(nPores+2, dtype=np.int32)
		cls.visited = np.empty(conn_graph_size, dtype=np.int32)
		cls.mem = np.zeros(totElements, dtype=np.bool_)
		cls.area = np.zeros(totElements, dtype=np.float32)
		cls.volume = np.zeros(totElements, dtype=np.float64)
		cls.memPore = np.zeros(nPores, dtype=np.bool_)
		cls.memThroat = np.zeros(nThroats, dtype=np.bool_)
		cls.ID = np.zeros(totElements, dtype=np.int32)
		cls.m_sinHalfAng = np.sin(halfAng, dtype=np.float32)
		cls.m_cosHalfAng = np.cos(halfAng, dtype=np.float32)
		cls.m_sinHalfAngcosHalfAng = cls.m_sinHalfAng*cls.m_cosHalfAng
		cls.tempID = np.full(totElements, -5, dtype=np.int32)
