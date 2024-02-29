import pandas as pd
from plot import makePlot



# with_trapping: 218; without_trapping: 70
#num = 70
#numDict = {'wt': 218, 'nt': 70}
numDict = {'wt': 241, 'nt': 92}
#numDict = {'wt': 3, 'nt': 4}
#numDict = {'wt': 1, 'nt': 1}
title = 'Bentheimer'
#title = 'estaillades_sublabel1'
drain = False
imbibe = False
probable = True
hysteresis = True
includeTrapping = True
scaled = True
plotDist = True


results = {'drainage':{}, 'imbibition':{}}
cycle=3
for i in range(1,cycle+1):
    label = 'wt'
    cycleLabel = 'cycle'+str(i)
    if includeTrapping or scaled:
        results['drainage'][label+'_'+cycleLabel] = pd.read_csv(
            './results_csv/Flowmodel_{}_Drainage_{}_wt_{}.csv'.format(
                title, cycleLabel, numDict['wt']), names=[
            'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
            sep=',', skiprows=17, index_col=False)
        
        results['imbibition'][label+'_'+cycleLabel] = pd.read_csv(
            './results_csv/Flowmodel_{}_Imbibition_{}_wt_{}.csv'.format(
                title, cycleLabel, numDict['wt']), names=[
            'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
            sep=',', skiprows=17, index_col=False)
        
        if plotDist:
            '''results['drainage'][label+'_'+cycleLabel+'_trDi'] = pd.read_csv(
                './results_csv/Flowmodel_{}_Drainage_{}_wt_{}_trappedDist.csv'.format(
                    title, cycleLabel, numDict['wt']), names=[
                'rad', 'volume', 'fluid', 'trappedW', 'trappedNW'],
                sep=',', skiprows=1, index_col=False)'''
            
            results['imbibition'][label+'_'+cycleLabel+'_trDi'] = pd.read_csv(
                './results_csv/Flowmodel_{}_Imbibition_{}_wt_{}_trappedDist.csv'.format(
                    title, cycleLabel, numDict['wt']), names=[
                'rad', 'volume', 'fluid', 'trappedW', 'trappedNW'],
                sep=',', skiprows=1, index_col=False)
    if not includeTrapping or scaled:
        label = 'nt'
        results['drainage'][label+'_'+cycleLabel] = pd.read_csv(
            './results_csv/Flowmodel_{}_Drainage_{}_nt_{}.csv'.format(
                title, cycleLabel, numDict['nt']), names=[
            'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
            sep=',', skiprows=17, index_col=False)
        
        results['imbibition'][label+'_'+cycleLabel] = pd.read_csv(
            './results_csv/Flowmodel_{}_Imbibition_{}_nt_{}.csv'.format(
                title, cycleLabel, numDict['nt']), names=[
            'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
            sep=',', skiprows=17, index_col=False)
        if plotDist:
            '''results['drainage'][label+'_'+cycleLabel+'_trDi'] = pd.read_csv(
                './results_csv/Flowmodel_{}_Drainage_{}_nt_{}_trappedDist.csv'.format(
                    title, cycleLabel, numDict['nt']), names=[
                'rad', 'volume', 'fluid', 'trappedW', 'trappedNW'],
                sep=',', skiprows=1, index_col=False)'''
            
            results['imbibition'][label+'_'+cycleLabel+'_trDi'] = pd.read_csv(
                './results_csv/Flowmodel_{}_Imbibition_{}_nt_{}_trappedDist.csv'.format(
                    title, cycleLabel, numDict['nt']), names=[
                'rad', 'volume', 'fluid', 'trappedW', 'trappedNW'],
                sep=',', skiprows=1, index_col=False)
        

#print(results)
#from IPython import embed; embed()


if drain:
    mkD = makePlot(numDict[label], title, drainage_results, True, True, True, False, include=None)
    mkD.pcSw()
    mkD.krSw()
if imbibe:
    mkI = makePlot(numDict[label], title, imbibition_results, True, True, False, True, include=None)
    mkI.pcSw()
    mkI.krSw()
if hysteresis:
    compWithLitData = True
    if not scaled:
        if not compWithLitData:
            mkH = makePlot(numDict[label], title, results, includeTrapping=includeTrapping)
            mkH.pcSw1()
            #mkH.krSw()
        else:
            #compWithLitData = False
            mkH = makePlot(numDict[label], title, results, includeTrapping=includeTrapping,
                           drain=True, imbibe=True, compWithLitData=compWithLitData)
            #mkH.pcSw1()
            print(mkH.compWithLitData)
            mkH.krSw1()
            #mkH.krSw2('drainage')
            #mkH.krSw2('imbibition')
            mkH.krSw3()
    else:
        mkH = makePlot(numDict[label], title, results, includeTrapping=includeTrapping)
        #mkH.pcSwScaled()
        #mkH.krSwScaled()
        #mkH.krSwProposed()
        mkH.plotDistribution()
    
    