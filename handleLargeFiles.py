import numpy as np
import pandas as pd


# Read and split large CSV file
chunk_size = 50  # Number of rows per chunk
col_index = np.arange(27329)
volarray = pd.read_csv('volarray_bent.dat', header=None)
print(volarray)
with open('avgDissolvedConcOstRipening_bent.dat', 'a') as f:
    for i, chunk in enumerate(pd.read_csv(
        'dissolvedMolesOstRipening_bent.dat',chunksize=chunk_size, header=None, usecols=col_index)):
        chunk = chunk/volarray.values.T
        chunk[np.isnan(chunk)] = 0
        np.savetxt(f, chunk[chunk!=0].mean(axis=1), delimiter=',', fmt='%g')


# Select the every-hour data from the raw data
chunk_size = 1000  # Number of rows per chunk
col_index = np.arange(27329)
linspace_values = np.linspace(0, 168*3600, 169)
timeArray = np.genfromtxt('timeArrayOstRipening_bent.dat', delimiter=',', invalid_raise=False)
timeArray = np.nan_to_num(timeArray)
indices = np.argmin(np.abs(timeArray[:, None] - linspace_values), axis=0)
with open('clustID_OstRipening_bent_every_hour.dat', 'a') as f:
    for i, chunk in enumerate(
        pd.read_csv(
            'clustIDOstRipening_bent.dat', header=None, chunksize=chunk_size, usecols=col_index)
        ):
        checkIndex = indices[(indices>=chunk.index.min())&(indices<=chunk.index.max())]
        for j in checkIndex:
            j = j - chunk.index.min()
            np.savetxt(f, [chunk.iloc[j].to_numpy()], delimiter=',', fmt='%g')
with open('timeArrayOstRipening_bent_every_hour.dat', 'a') as f:
    np.savetxt(f, [timeArray[indices]], delimiter=',', fmt='%g')

chunk_size = 1000  # Number of rows per chunk
col_index = np.arange(27329)
linspace_values = np.linspace(0, 168*3600, 169)
timeArray = np.genfromtxt('timeArrayOstRipening_bent.dat', delimiter=',', invalid_raise=False)
timeArray = np.nan_to_num(timeArray)
indices = np.argmin(np.abs(timeArray[:, None] - linspace_values), axis=0)
with open('clustPc_OstRipening_bent_every_hour.dat', 'a') as f:
    with open('clustPcOstRipening_bent.dat', 'r') as file:
        clustPc = csv.reader(file)
        j = 0
        for i, row in enumerate(clustPc):
            if i==indices[j]:
                gasPc = np.array([*map(float, row)])
                np.savetxt(f, [gasPc], delimiter=',', fmt='%g')
                j += 1

chunk_size = 1000  # Number of rows per chunk
col_index = np.arange(27329)
linspace_values = np.linspace(0, 168*3600, 169)
timeArray = np.genfromtxt('timeArrayOstRipening_bent.dat', delimiter=',', invalid_raise=False)
timeArray = np.nan_to_num(timeArray)
indices = np.argmin(np.abs(timeArray[:, None] - linspace_values), axis=0)
with open('clustVol_OstRipening_bent_every_hour.dat', 'a') as f:
    with open('clustVolOstRipening_bent.dat', 'r') as file:
        clustVol = csv.reader(file)
        j = 0
        for i, row in enumerate(clustVol):
            if i==indices[j]:
                gasVol = np.array([*map(float, row)])
                np.savetxt(f, [gasVol], delimiter=',', fmt='%g')
                j += 1
    
chunk_size = 1000  # Number of rows per chunk
col_index = np.arange(27329)
linspace_values = np.linspace(0, 168*3600, 169)
timeArray = np.genfromtxt('timeArrayOstRipening_bent.dat', delimiter=',', invalid_raise=False)
timeArray = np.nan_to_num(timeArray)
indices = np.argmin(np.abs(timeArray[:, None] - linspace_values), axis=0)
with open('clustMoles_OstRipening_bent_every_hour.dat', 'a') as f:
    with open('clustMolesOstRipening_bent.dat', 'r') as file:
        clustMol = csv.reader(file)
        j = 0
        for i, row in enumerate(clustMol):
            if i==indices[j]:
                gasMol = np.array([*map(float, row)])
                np.savetxt(f, [gasMol], delimiter=',', fmt='%g')
                j += 1
    
        
# The average pc file
import csv
col_index = np.arange(27329)
clustIDArray = pd.read_csv(
    'clustID_OstRipening_bent_every_hour.dat', header=None, usecols=col_index)
dissolvedMolesArray = pd.read_csv(
    'dissolvedMoles_OstRipening_bent_every_hour.dat', header=None, usecols=col_index)
volarray = np.loadtxt('volarray_bent.dat', delimiter=',')
timeArray = np.genfromtxt('timeArrayOstRipening_bent.dat', delimiter=',', invalid_raise=False)
timeArray = np.nan_to_num(timeArray)
indices = np.argmin(np.abs(timeArray[:, None] - linspace_values), axis=0)
H = 6.9e-6
with open('AvgPc_time_OstRipening_bent_every_hour.dat', 'a') as f:
    with open('clustPcOstRipening_bent.dat', 'r') as filePc,\
        open('clustVolOstRipening_bent.dat', 'r') as fileVol:
        clustPc = csv.reader(filePc)
        clustVol = csv.reader(fileVol)
        j = 0
        for i, (rowPc, rowVol) in enumerate(zip(clustPc, clustVol)):
            if i==indices[j]:
                clustID = clustIDArray.iloc[j].to_numpy()
                dissolvedMoles = dissolvedMolesArray.iloc[j].to_numpy()
                #aqPc = np.mean(np.nan_to_num(dissolvedMoles[clustID<0]/volarray[clustID<0])/H)
                aqPc = dissolvedMoles[clustID<0].sum()/volarray[clustID<0].sum()/H
                clustID = clustID[clustID>=0]
                #gasPc = np.array([*map(float, row)])[clustIDNW].mean()
                #PcV = np.array([*map(float, rowPc)])[clustIDNW]*volarray[clustIDNW]
                rowPc = np.array([*map(float, rowPc)])
                rowVol = np.array([*map(float, rowVol)])
                #PcV = np.array([*map(float, rowPc)])[clustID]*volarray[clustID]
                #gasPc = PcV.sum()/volarray[clustID].sum()
                gasPc = (rowPc*rowVol).sum()/rowVol.sum()
                np.savetxt(f, [[timeArray[i], gasPc, aqPc]], delimiter=',', fmt='%g')
                j += 1


# Frequency distribution
chunk_size = 1  # Number of rows per chunk
num_bins=50
col_index = np.arange(27329)
rarray = np.loadtxt('radiiArray_bent.dat', delimiter=',')
with open('freqDistOstRipening_bent_every_hour_50bins.dat', 'a') as f:
    bin_edges = np.linspace(rarray.min(), rarray.max(), num_bins + 1)
    bin_indices = np.digitize(rarray, bin_edges, right=True)
    data = pd.read_csv('clustID_OstRipening_bent_every_hour.dat', header=None, usecols=col_index)
    for i in data.index:
        counts = np.bincount(bin_indices[(data.iloc[i].to_numpy()>=0)], minlength=num_bins+1)
        np.savetxt(f, [counts], delimiter=',', fmt='%g')

# cumulative volume distribution
chunk_size = 1  # Number of rows per chunk
num_bins=50
col_index = np.arange(27329)
with open('volDistOstRipening_bent_every_hour_50bins_1.dat', 'a') as f,\
    open('clustVol_OstRipening_bent_every_hour.dat', 'r') as fileVol:
    clustVol = csv.reader(fileVol)
    maxVol = 0.0
    for rowVol in clustVol:
        maxVol = max(maxVol, max([*map(float, rowVol)]))
    bin_edges = np.linspace(0.0, maxVol, num_bins + 1)
    fileVol.seek(0)
    for i, rowVol in enumerate(clustVol):
        rowVol = np.array([*map(float, rowVol)])
        rowVol = rowVol[rowVol>0.0]
        bin_indices = np.digitize(rowVol, bin_edges, right=True)
        counts = np.bincount(bin_indices, rowVol, minlength=num_bins+1)
        np.savetxt(f, [counts], delimiter=',', fmt='%g')


# case study
import csv
timeArray = np.genfromtxt('timeArrayOstRipening_bent.dat', delimiter=',', invalid_raise=False)
timeArray = np.nan_to_num(timeArray)
linspace_values = np.linspace(0, 168*3600, 169)
indices = np.argmin(np.abs(timeArray[:, None] - linspace_values), axis=0)
col_index = np.arange(27329)
clustID = pd.read_csv('clustID_OstRipening_bent_every_hour.dat', header=None, usecols=col_index)
# for pore ID 1
with open('pore_1_case_study_OstRipening_bent_every_hour.dat', 'a') as f,\
    open('clustPc_OstRipening_bent_every_hour.dat', 'r') as filePc,\
    open('clustVol_OstRipening_bent_every_hour.dat', 'r') as fileVol,\
    open('clustMoles_OstRipening_bent_every_hour.dat', 'r') as fileMol:
        clustPc = csv.reader(filePc)
        clustVol = csv.reader(fileVol)
        clustMol = csv.reader(fileMol)
        clustID_1 = clustID[1].to_numpy()
        for i, (rowPc, rowVol, rowMol) in enumerate(zip(clustPc, clustVol, clustMol)):
            if clustID_1[i]>=0:
                _size = (clustID.iloc[i]==clustID_1[i]).sum()
            else:
                _size = 0
            _pc = np.array([*map(float, rowPc)])[clustID_1[i]]
            _vol = np.array([*map(float, rowVol)])[clustID_1[i]]
            _mol = np.array([*map(float, rowMol)])[clustID_1[i]]
            print(clustID_1[i], _size, _pc, _vol, _mol)
            np.savetxt(f, [[clustID_1[i], _size, _pc, _vol, _mol]], delimiter=',', fmt='%g')

# for pore ID 6243
with open('pore_6243_case_study_OstRipening_bent_every_hour.dat', 'a') as f,\
    open('clustPc_OstRipening_bent_every_hour.dat', 'r') as filePc,\
    open('clustVol_OstRipening_bent_every_hour.dat', 'r') as fileVol,\
    open('clustMoles_OstRipening_bent_every_hour.dat', 'r') as fileMol:
    clustPc = csv.reader(filePc)
    clustVol = csv.reader(fileVol)
    clustMol = csv.reader(fileMol)
    clustID_6243 = clustID[6243].to_numpy()
    for i, (rowPc, rowVol, rowMol) in enumerate(zip(clustPc, clustVol, clustMol)):
        if clustID_6243[i]>=0:
            _size = (clustID.iloc[i]==clustID_6243[i]).sum()
        else:
            _size = 0
        _pc = np.array([*map(float, rowPc)])[clustID_6243[i]]
        _vol = np.array([*map(float, rowVol)])[clustID_6243[i]]
        _mol = np.array([*map(float, rowMol)])[clustID_6243[i]]
        print(clustID_6243[i], _size, _pc, _vol, _mol)
        np.savetxt(f, [[clustID_6243[i], _size, _pc, _vol, _mol]], delimiter=',', fmt='%g')


# for pore ID 8421
with open('pore_8421_case_study_OstRipening_bent_every_hour.dat', 'a') as f,\
    open('clustPc_OstRipening_bent_every_hour.dat', 'r') as filePc,\
    open('clustVol_OstRipening_bent_every_hour.dat', 'r') as fileVol,\
    open('clustMoles_OstRipening_bent_every_hour.dat', 'r') as fileMol:
    clustPc = csv.reader(filePc)
    clustVol = csv.reader(fileVol)
    clustMol = csv.reader(fileMol)
    clustID_8421 = clustID[8421].to_numpy()
    for i, (rowPc, rowVol, rowMol) in enumerate(zip(clustPc, clustVol, clustMol)):
        if clustID_8421[i]>=0:
            _size = (clustID.iloc[i]==clustID_8421[i]).sum()
        else:
            _size = 0
        _pc = np.array([*map(float, rowPc)])[clustID_8421[i]]
        _vol = np.array([*map(float, rowVol)])[clustID_8421[i]]
        _mol = np.array([*map(float, rowMol)])[clustID_8421[i]]
        print(clustID_8421[i], _size, _pc, _vol, _mol)
        np.savetxt(f, [[clustID_8421[i], _size, _pc, _vol, _mol]], delimiter=',', fmt='%g')



# compute the actual volume of gas in cluster from ID and Pc
# this has to be run while running the ostripening code
previousPc = 0 #defined in the code
previousVol = 0 #defined in the code
clustVol = 0 #defined in the code
self = 0 #object of the class
with open("clustVolOstRipening_bent_new_actual.dat", 'a') as f:
    clustPcData = np.loadtxt('clustPcOstRipening_bent_new.dat', delimiter=',', dtype=float)
    clustIDData = np.loadtxt('clustIDOstRipening_bent_new.dat', delimiter=',', dtype=int)
    elementList = np.arange(23728)
    for i, (clustID, clustPc) in enumerate(zip(clustIDData, clustPcData)):
        elemPc = clustPc[clustID]
        elemPc[clustID<0] = 0.0
        mem = (elemPc != previousPc)
        keys = np.unique(clustID[mem])
        self.clusterNW.computeVolume(keys, mem, clustID, self)
        np.savetxt(f, [self.clusterNW.volume], delimiter=',', fmt='%g')



#**********************************************************************************#
from matplotlib import pyplot as plt
import os

os.makedirs("./scatterplot", exist_ok=True)
clustPc = np.loadtxt('clustPcOstRipening_bent_new.dat',delimiter=',')
clustVol = np.loadtxt('clustVolOstRipening_bent_new.dat',delimiter=',')
clustID = np.loadtxt('clustIDOstRipening_bent_new.dat',delimiter=',')
timeArray = np.genfromtxt('timeArrayOstRipening_bent_new.dat',delimiter=',', dtype=float)

clustPc = clustPc[~np.isnan(timeArray)]

# select clusters whose membership did not change
selectedIDS = []
for c in range(clustVol.shape[1]):
    try:
        if all(np.where(clustID[0]==c)[0]==np.where(clustID[-1]==c)[0]) and\
            ((clustVol[0,c] != clustVol[-1,c])) and\
                ((clustVol[0,c] != 0.0) or (clustVol[-1,c] != 0.0)):
            selectedIDS.append(c)
    except:
        pass
selectedIDS = np.array(selectedIDS)

for i, t in enumerate(3600*24*np.arange(6)):
    c = np.argmin(np.abs(timeArray[:-1]-t))
    plt.scatter(clustVol[c, selectedIDS], clustPc[c, selectedIDS],
                alpha=0.7,
                edgecolors='blue')
    plt.xlabel(r'Cluster volume ($m^3$)')
    plt.ylabel('Capillary pressure (Pa)')
    plt.xscale('log')
    plt.savefig('./scatterplot/scatterplot_pc_vol_{i}_days.png', dpi=500)
    plt.close()


import dill
satList = [0.64333]
for i in [24,48,72,96,120,144]:
    with open(f"netsim_time_Dependent_{i}_hours.pkl", "rb") as f:
        self = dill.load(f)
        sat = (self.satList[self.isinsideBox]*self.volarray[
                    self.isinsideBox]).sum()/self.totVoidVolume
        satList.append(sat)


import re
import numpy as np

with open(os.path.join(MEMORY_DIR,'saturationOstRipening_bent.dat'), "r") as f:
    line = f.read().strip()

print(line.count('.'))
line = re.sub(r'(?<=\d)(?=0\.)', ' ', line)
numbers = numbers = re.findall(r'0\.\d+', line)
saturation0 = np.array([float(num) for num in numbers])  # Convert to float
print(saturation0.size)
print(saturation0)

with open(os.path.join(MEMORY_DIR,'saturationOstRipening_bent_6.dat'), "r") as f:
    line = f.read().strip()

print(line.count('.'))
line = re.sub(r'(?<=\d)(?=0\.)', ' ', line)
numbers = numbers = re.findall(r'0\.\d+', line)
saturation6 = np.array([float(num) for num in numbers])  # Convert to float
print(saturation6.size)
print(saturation6)

saturation = np.concatenate((saturation0, saturation6))
print(saturation.size, saturation)
np.savetxt(os.path.join(MEMORY_DIR,'saturationOstRipening_bent_new.dat'), saturation, delimiter=',', fmt='%g')



xarray = np.linspace(0, self.xDim, 1001)
satListArray = np.zeros([9,1000], dtype=float)
for i, t in enumerate([12,24,48,120,240,360,480,600,720]):
     with open(os.path.join(MEMORY_DIR, f"netsim_time_Dependent_{t}_hours.pkl"), "rb") as f:
        loaded_obj = dill.load(f)
    updateObj(self, loaded_obj)
satListArray[i] = positionalSaturation(self, xarray)
