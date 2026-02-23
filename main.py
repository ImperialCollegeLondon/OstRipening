<<<<<<< Updated upstream
from datetime import date
import sys
import os
import pandas as pd

sys.path.append("./pnflowPy")
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
        #timeDependent = False

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
                        netsim = SecDrainage(netsim, writeData=writeData, 
                                             writeTrappedData=writeTrappedData)
                    netsim.drainage()

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


=======
from datetime import date
import sys
import os
import numpy as np
import dill
import joblib

sys.path.append("./pnflowPy")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pnflowPy.inputData import InputData
from pnflowPy.network import Network
import pnflowPy.sPhase as sPhase
import pnflowPy.tPhaseD as tPhaseD
import pnflowPy.tPhaseImb as tPhaseImb
import pnflowPy.utilities as do
from pnflowPy.tPhaseD import TwoPhaseDrainage as PDrainage
from pnflowPy.tPhaseImb import TwoPhaseImbibition as PImbibition
from pnflowPy.SecondaryProcesses import SecDrainage, SecImbibition 
from OstRipening.mtimeDependency import TimeDependency
import OstRipening.mtimeDependency as tDependency


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
        #from IPython import embed; embed()
        
       
        writeData = True
        writeTrappedData = False
        fillTillNWDisconnected = True
        saveDrainage = True
        saveImbibition = True
        skip_drainage_imbibition = True
        skip_drainage = False
        skip_imbibition = False
        start_from_scratch = True
        equilibrium = False

        MEMORY_DIR = f"ostwald_ripening_results/"
        # two Phase simulations
        if not skip_drainage_imbibition and input_data.satControl():
            firstDrainCycle = True
            firstImbCycle = True
            netsim.cycle = 0
            netsim.saveDrainage = saveDrainage
            netsim.saveImbibition = saveImbibition

            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
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
                        file_path = os.path.join(MEMORY_DIR, f"drainage_{netsim.title}_69999.pkl")
                        loaded_obj = joblib.load(file_path)
                        do.updateObj(netsim, loaded_obj)
                        write_drainage_result(netsim)
                        
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
                            netsim.results_dir = MEMORY_DIR

                            tPhaseD.initialize(netsim)
                            netsim.prop_drainage = {}
                            netsim.prop_drainage['contactAng'] = netsim.contactAng.copy()
                            netsim.prop_drainage['thetaRecAng'] = netsim.thetaRecAng.copy()
                            netsim.prop_drainage['thetaAdvAng'] = netsim.thetaAdvAng.copy()
                        else:
                            SecDrainage(netsim, writeData=writeData, writeTrappedData=writeTrappedData)
                    
                        tPhaseD.drainage(netsim)
                    firstDrainCycle = False
                        
                else:
                    # Imbibition process
                    if skip_imbibition:
                        file_path = os.path.join(MEMORY_DIR, f"imbibition_{netsim.title}_1067.pkl")
                        loaded_obj = joblib.load(file_path)
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
                            netsim.results_dir = MEMORY_DIR

                            tPhaseImb.initialize(netsim)
                            netsim.prop_imbibition = {}
                            netsim.prop_imbibition['contactAng'] = netsim.contactAng.copy()
                            netsim.prop_imbibition['thetaRecAng'] = netsim.thetaRecAng.copy()
                            netsim.prop_imbibition['thetaAdvAng'] = netsim.thetaAdvAng.copy()
                            firstImbCycle = False
                        else:
                            SecImbibition(netsim, writeData=writeData,writeTrappedData=writeTrappedData)
                        
                        tPhaseImb.imbibition(netsim)
        else:
            
            if netsim.title=='Bentheimer':
                file_path = os.path.join(MEMORY_DIR, f"imbibition_{netsim.title}_1365.pkl")
            elif netsim.title=='BentSepi600':
                file_path = os.path.join(MEMORY_DIR, f"imbibition_{netsim.title}_4436.pkl")
                
            loaded_obj = joblib.load(file_path)
            do.updateObj(netsim, loaded_obj)
            write_imbibition_result(netsim)
        
        if equilibrium:
            #from IPython import embed; embed()
            import OstRipening.mequilibrium as equilibrium
            equilibrium.initialize(netsim)
            equilibrium.equilibrate(netsim)
        else:
            if netsim.title=='Bentheimer':
                D = 4.89e-9
                imposedP = 1e6
            elif netsim.title=='BentSepi600':
                D = 4.75e-9
                imposedP = 8e6
            if start_from_scratch:
                TimeDependency(
                    netsim, netsim.capPresMin, steps=40000, dt=0.005, 
                    D=D, imposedP=imposedP,                  
                    H = 7.8e-6)
        
                tDependency.initialize(netsim)
                
            tDependency.simulateOstRip(netsim, freshStart=start_from_scratch)
            
        print("\n\n Simulation finished successfully!\n")
        from IPython import embed; embed()
                   
        
    except Exception as exc:
        print("\n\n Exception on processing: \n", exc, "Aborting!\n")
        return 1
    except:
        from IPython import embed; embed()
        print("\n\n Unknown exception! Aborting!\n")
        return 1

    return 0


def write_drainage_result(self):
    print('----------------------------------------------------------------------------------')
    print('---------------------------------Two Phase Drainage Cycle {}---------------------'.format(self.cycle))
    print('Sw: %10.6g  \tqW: %8.6e  \tkrw: %12.6g  \tqNW: %8.6e  \tkrnw:\
    %12.6g  \tPc: %8.6g\t %8.0f invasions' % (
    self.satW, self.qW, self.krw, self.qNW, self.krnw, self.capPresMax, self.totNumFill, ))
    print('\n\n')


def write_imbibition_result(self):
    print('----------------------------------------------------------------------------------')
    print('---------------------------------Two Phase Imbibition Cycle {}---------------------'.format(self.cycle))
    print('Sw: %10.6g  \tqW: %8.6e  \tkrw: %12.6g  \tqNW: %8.6e  \tkrnw:\
    %12.6g  \tPc: %8.6g\t %8.0f invasions' % (
    self.satW, self.qW, self.krw, self.qNW, self.krnw, self.capPresMin, self.totNumFill, ))
    print('\n\n')


if __name__ == "__main__":
    sys.exit(main())


>>>>>>> Stashed changes
