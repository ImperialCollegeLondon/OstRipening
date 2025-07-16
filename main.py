from datetime import date
import sys
import os
import numpy as np

sys.path.append("./pnflowPy")
from pnflowPy.inputData import InputData
from pnflowPy.network import Network
import pnflowPy.sPhase as sPhase
import pnflowPy.tPhaseD as tPhaseD
import pnflowPy.tPhaseImb as tPhaseImb
import pnflowPy.utilities as do
import dill



# __DATE__ = "Jul 25 , 2023"
__DATE__ = date.today().strftime("%b") + " " + str(date.today().day) + ", " +\
      str(date.today().year)


def main():
    try:
        input_file_name = ""

        print("\nNetwork Model Code version 2 alpha, built: ", __DATE__, "\n")

        if len(sys.argv) > 1:
            input_file_name = sys.argv[1]
        else:
            input_file_name = input("Please input data file : ")

        input_data = InputData(input_file_name)
        netsim = Network(input_file_name)

        # Single Phase computation 
        sPhase.initialize(netsim)
        sPhase.singlephase(netsim)
        
        writeData = False
        writeTrappedData = False
        fillTillNWDisconnected = True
        timeDependent = True
        saveDrainage = True
        saveImbibition = False
        #timeDependent = False
        skip_drainage_imbibition = False
        skip_drainage = False
        skip_imbibition = False

        if timeDependent:
            from pnflowPy.tPhaseD import TwoPhaseDrainage as PDrainage
            from pnflowPy.tPhaseImb import TwoPhaseImbibition as PImbibition
            from pnflowPy.SecondaryProcesses import SecDrainage, SecImbibition 
            from timeDependency import TimeDependency
            import timeDependency as tDependency
        else:
            from percolation_without_trapping import PDrainage, PImbibition, SecDrainage, SecImbibition

        # two Phase simulations
        if not skip_drainage_imbibition and input_data.satControl():
            firstDrainCycle = True
            firstImbCycle = True
            netsim.cycle = 0
            netsim.saveDrainage = saveDrainage
            netsim.saveImbibition = saveImbibition
            for j in range(len(input_data.satControl())):
                netsim.finalSat, Pc, netsim.dSw, netsim.minDeltaPc,\
                 netsim.deltaPcFraction, netsim.calcKr, netsim.calcI,\
                 netsim.InjectFromLeft, netsim.InjectFromRight,\
                 netsim.EscapeFromLeft, netsim.EscapeFromRight =\
                 input_data.satControl()[j]
                netsim.filling = True

                if netsim.finalSat < netsim.satW:
                    # Drainage process
                    if skip_drainage:
                        print('11111111111111')
                        with open(os.path.join(f'./saved_simulation_{netsim.title}', 
                                               f"drainage_999999.pkl"), "rb") as f:
                            loaded_obj = dill.load(f)
                            #netsim = dill.load(f)
                        print('333333333333333333')
                        do.updateObj(netsim, loaded_obj)
                        print('2222222222222222')
                        #from IPython import embed; embed()
                        
                        netsim.areaWPhase = netsim._areaWP.view()
                        netsim.areaNWPhase = netsim._areaNWP.view()
                        netsim.gWPhase = netsim._condWP.view()
                        netsim.gNWPhase = netsim._condNWP.view()
                        
                    else:
                        netsim.is_oil_inj = True
                        netsim.maxPc = Pc
                        if firstDrainCycle:
                            (netsim.wettClass, netsim.minthetai, netsim.maxthetai, netsim.delta,
                                netsim.eta, netsim.distModel, netsim.sepAng) = input_data.initConAng('INIT_CONT_ANG')
                            PDrainage(netsim, writeData=writeData, writeTrappedData=writeTrappedData)
                            tPhaseD.initialize(netsim)
                            netsim.prop_drainage = {}
                            netsim.prop_drainage['contactAng'] = netsim.contactAng.copy()
                            netsim.prop_drainage['thetaRecAng'] = netsim.thetaRecAng.copy()
                            netsim.prop_drainage['thetaAdvAng'] = netsim.thetaAdvAng.copy()
                        else:
                            SecDrainage(netsim, writeData=writeData, writeTrappedData=writeTrappedData)
                            SecDrainage.initialize(netsim)
                            
                        tPhaseD.drainage(netsim)
                    firstDrainCycle = False
                        
                        
                else:
                    # Imbibition process
                    if skip_imbibition:
                        with open(os.path.join(f'./saved_simulation_{netsim.title}', 
                                               f"imbibition_1365_67.pkl"), "rb") as f:
                            loaded_obj = dill.load(f)
                        do.updateObj(netsim, loaded_obj)
                    else:
                        netsim.is_oil_inj = False
                        netsim.minPc = Pc
                        netsim.fillTillNWDisconnected = fillTillNWDisconnected
                        
                        if firstImbCycle:
                            (netsim.wettClass, netsim.minthetai, netsim.maxthetai, netsim.delta,
                                netsim.eta, netsim.distModel, netsim.sepAng) = input_data.initConAng(
                                    'EQUIL_CON_ANG')
                                    
                            PImbibition(netsim, writeData=writeData, writeTrappedData=writeTrappedData)
                            tPhaseImb.initialize(netsim)
                            netsim.prop_imbibition = {}
                            netsim.prop_imbibition['contactAng'] = netsim.contactAng.copy()
                            netsim.prop_imbibition['thetaRecAng'] = netsim.thetaRecAng.copy()
                            netsim.prop_imbibition['thetaAdvAng'] = netsim.thetaAdvAng.copy()
                            firstImbCycle = False
                        else:
                            SecImbibition(netsim, writeData=writeData,writeTrappedData=writeTrappedData)
                            SecImbibition.initialize(netsim)
                        
                        tPhaseImb.imbibition(netsim)
                       
                            

        #timeDependent = False
        
        with open(os.path.join(f'./saved_simulation_{netsim.title}', 
                                           f"imbibition_1365.pkl"), "rb") as f:
            loaded_obj = dill.load(f)
        do.updateObj(netsim, loaded_obj)
        try:
            assert timeDependent
            TimeDependency(
                netsim, netsim.capPresMin, steps=40000, dt=0.054, 
                D=1.8e-9,
                #D=5e-9,
                H=6.9e-6,
                #H=1.2e-7,
                imposedP=1e6)
            try:
                assert freshStart
                tDependency.initialize(netsim)
                #tDependency.recomputeClusterVolume(netsim)
            except AssertionError:
                pass
            tDependency.simulateOstRip(netsim, freshStart=freshStart)
            
            #print('::::::::::::::::::::::::::::')
            #from IPython import embed; embed()
            netsim.filling = True
            netsim.capPresMax = netsim.maxPc = netsim.aqAvgPres
            netsim.fillTillNWDisconnected = False
            netsim.minPc = netsim.Pc
            SecImbibition(netsim, writeData=writeData,writeTrappedData=writeTrappedData)
            SecImbibition.initialize(netsim)
            netsim._areaWP[:] = netsim.satList*netsim.areaSPhase
            netsim._areaNWP[:] = (1-netsim.satList)*netsim.areaSPhase
            arr = np.ones(netsim.totElements, dtype=bool)
            newPc = netsim.clusterNW.pc[netsim.clusterNW_ID]
            
            tPhaseImb.__CondTPImbibition__(netsim, arr, newPc, True, True)
            netsim.satW = do.Saturation(netsim, netsim.areaWPhase, netsim.areaSPhase)
            do.computePerm(netsim, netsim.capPresMin)
            tPhaseImb.imbibition(netsim)
            
        except AssertionError:
            pass
                   

                   
        
    except Exception as exc:
        print("\n\n Exception on processing: \n", exc, "Aborting!\n")
        return 1
    except:
        from IPython import embed; embed()
        print("\n\n Unknown exception! Aborting!\n")
        return 1

    return 0





if __name__ == "__main__":
    sys.exit(main())


