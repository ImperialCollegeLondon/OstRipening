from datetime import date
import sys
import os
import pandas as pd

sys.path.append("./pnflowPy")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pnflowPy.inputData import InputData
from pnflowPy.network import Network
from pnflowPy.sPhase import SinglePhase
from plot import makePlot


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
        netsim = SinglePhase(netsim)
        netsim.singlephase()
        writeData = False
        writeTrappedData = False
        fillTillNWDisconnected = True
        timeDependent = True
<<<<<<< Updated upstream
        #timeDependent = False
=======
        saveDrainage = True
        saveImbibition = True
        timeDependent = True
        skip_drainage_imbibition = True
        skip_drainage = False
        skip_imbibition = False
        start_from_scratch = True
>>>>>>> Stashed changes

        if timeDependent:
            from pnflowPy.tPhaseD import TwoPhaseDrainage as PDrainage
            from pnflowPy.tPhaseImb import TwoPhaseImbibition as PImbibition
            from pnflowPy.SecondaryProcesses import SecDrainage, SecImbibition 
            from timeDependency import TimeDependency
        else:
            from Percolation_without_Trapping import PDrainage, PImbibition, SecDrainage, SecImbibition



        # two Phase simulations
        if input_data.satControl():
            firstDrainCycle = True
            firstImbCycle = True
            for j in range(len(input_data.satControl())):
                netsim.finalSat, Pc, netsim.dSw, netsim.minDeltaPc,\
                 netsim.deltaPcFraction, netsim.calcKr, netsim.calcI,\
                 netsim.InjectFromLeft, netsim.InjectFromRight,\
                 netsim.EscapeFromLeft, netsim.EscapeFromRight =\
                 input_data.satControl()[j]
                netsim.filling = True

                try:
                    assert netsim.finalSat < netsim.satW
                    # Drainage process
                    netsim.is_oil_inj = True
                    netsim.maxPc = Pc
                    if firstDrainCycle:
                        (netsim.wettClass, netsim.minthetai, netsim.maxthetai, netsim.delta,
                            netsim.eta, netsim.distModel, netsim.sepAng) = input_data.initConAng(
                                'INIT_CONT_ANG')
                        netsim = PDrainage(netsim, writeData=writeData, 
                                           writeTrappedData=writeTrappedData)
                        netsim.prop_drainage = {}
                        netsim.prop_drainage['contactAng'] = netsim.contactAng.copy()
                        netsim.prop_drainage['thetaRecAng'] = netsim.thetaRecAng.copy()
                        netsim.prop_drainage['thetaAdvAng'] = netsim.thetaAdvAng.copy()
                        firstDrainCycle = False
                    else:
<<<<<<< Updated upstream
                        netsim = SecDrainage(netsim, writeData=writeData, 
                                             writeTrappedData=writeTrappedData)
                    netsim.drainage()
=======
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
                            
                        # import cProfile
                        # import pstats
                        # profiler = cProfile.Profile()
                        # profiler.enable()
                        # tPhaseD.drainage(netsim)
                        # profiler.disable()
                        # stats = pstats.Stats(profiler).sort_stats('cumtime')
                        # stats.print_stats()
                        #from IPython import embed; embed()
                        
                        tPhaseD.drainage(netsim)

                    netsim.maxCenterArea = netsim.areaNWPhase.copy()
                    firstDrainCycle = False
                        
                        
                else:
                    # Imbibition process
                    if skip_imbibition:
                        with open(os.path.join(f'./saved_simulation_{netsim.title}', 
                                               f"imbibition_1219.pkl"), "rb") as f:
                            loaded_obj = dill.load(f)
                        do.updateObj(netsim, loaded_obj)
                        write_imbibition_result(netsim)
                        
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
                
            
        else:
            with open(os.path.join(f'./saved_simulation_{netsim.title}', 
                                            f"imbibition_1219.pkl"), "rb") as f:
                loaded_obj = dill.load(f)
            do.updateObj(netsim, loaded_obj)
            write_imbibition_result(netsim)
>>>>>>> Stashed changes

                except AssertionError:
                    # Imbibition process
                    netsim.is_oil_inj = False
                    netsim.minPc = Pc
                    netsim.fillTillNWDisconnected = fillTillNWDisconnected
                    if firstImbCycle:
                        (netsim.wettClass, netsim.minthetai, netsim.maxthetai, netsim.delta,
                            netsim.eta, netsim.distModel, netsim.sepAng) = input_data.initConAng(
                                'EQUIL_CON_ANG')
                        netsim = PImbibition(netsim, writeData=writeData,
                                             writeTrappedData=writeTrappedData)
                        netsim.prop_imbibition = {}
                        netsim.prop_imbibition['contactAng'] = netsim.contactAng.copy()
                        netsim.prop_imbibition['thetaRecAng'] = netsim.thetaRecAng.copy()
                        netsim.prop_imbibition['thetaAdvAng'] = netsim.thetaAdvAng.copy()
                        firstImbCycle = False
                    else:
                        netsim = SecImbibition(netsim, writeData=writeData,
                                               writeTrappedData=writeTrappedData)
                        
                    netsim.imbibition()

            try:
                assert timeDependent
                tDependency = TimeDependency(
                    netsim, netsim.capPresMin, steps=40000, dt=0.0027, D=2.23e-9, H=3.4e-4)           
                tDependency.simulateOstRip(True)
            except AssertionError:
                pass
                   

                   
        else:
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


