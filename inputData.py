import numpy as np
from time import time
import os
from pnflowPy.inputData import InputData as inputData

class InputData(inputData):
    def __init__(self, inputFile): 
        super().__init__(inputFile)
        
    def initRipeningParams(self, obj):
        mode = False
        alpha = 0.0
        
        if self.data['RIPENING']:
            data = self.data['RIPENING']
            alpha = data[1]
            
            if data[0]=='T':
                mode = True
                # obj.start_from_scratch = True if data[1]=='T' else False
                obj.duration = data[2] 
                obj._dt = data[3]
                obj.D = data[4]
                obj.imposedP = data[5]
                obj.T = data[6]
                obj.H = data[7]
                obj.moles_tol = data[8]
                obj.pc_tol = data[9]
                obj.max_iter = data[10]
                
        obj.alpha = alpha
        obj.mode = mode
        
    def res_dir(self, obj):
        if self.data['RES_DIR']:
            dir = self.data['RES_DIR']
            if obj.mode:
                MEMORY_DIR = os.path.join("ostwald_ripening_results", dir)
            else:
                MEMORY_DIR = os.path.join("equilibrium_results", dir)
            
            os.makedirs(MEMORY_DIR, exist_ok=True)
            obj.MEMORY_DIR = MEMORY_DIR
                
            