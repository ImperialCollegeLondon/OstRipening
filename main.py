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
                    if timeDependent:
                        netsim.minCornerArea = netsim._cornArea.copy()
                        netsim.prevFilled = (netsim.fluid==1)

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
                    netsim, netsim.capPresMin, steps=40000, dt=0.0027, D=1.8e-9,
                    H=6.9e-6, imposedP=1e6)           
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


