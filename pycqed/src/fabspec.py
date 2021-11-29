#
#
#
#
from . import units as un
import numpy as np

## A class for defining fabrication properties
#
# For a given fabrication process, there should be a series of design constants
# With each design constant can be associated an error and distribution
#
#
class FabSpec:
    
    ## Class constants
    __process_constants = {
        "shadow_evap":[
            "Jc", # Critical current density
            "Ca", # Capacitance density
            "Lo"  # Shadow overlap
        ]
    }
    
    __distributions = ["Gaussian","Lorentzian","Poisson"]
    
    ## Constructor for the fabrication specifications class
    #
    #
    #
    def __init__(self,spec,process="shadow_evap",units=None):
        self.spec = dict([(k,[0,0,0,""]) for k in list(__process_constants[process].keys())])
        
    
    ## Add a design constant and associated error properties
    def addDesignConst(self,name,mean,error,distribution="Gaussian"):
        pass
    
    ## Add many design constants in one go
    def addDesignConsts(self,specdict):
        pass
    
    ## Get design constant values sampled from the associated distribution
    def sampleDesignConst(self,key,N):
        pass
    
    ## Define the relationship between inter-dependent parameters
    #
    # For example, in shadow evaporation process, the JJs and caps are dependent on each other
    # through the area of the junction.
    #
    def setDependentExpr(self,expr):
        pass


