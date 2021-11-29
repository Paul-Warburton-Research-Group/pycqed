import qutip as qt
import numpy as np
import sympy as sy
import scipy as sc
import networkx as nx
import time

from . import circuitspec as cs
from . import fabspec as fs
from . import parameters as pa
from . import units as un
from . import physical_constants as pc
from . import util
from . import units
from . import dataspec as ds

#
# DEV NOTES:
#
# To implement coordinate transformations, the following procedure could be done:
# - After class creation, identify a transformation and set it
# - Create new symbols for each vector, and associate them with the untransformed vectors
#
#####
#
# To implement generic and automatic dependency computing could do:
# - Make all independent sweepable getter functions (getHamiltonian for example) save results to attributes
# - For each dependency specified, check the attribute exists
#
# Attribute dependency list to determine which dependency needs to be evaluated.
# - paramSweep should be able to figure out when dependency needs to be evaluated in and nD sweep.
#
# Can have pre-dependencies: Only need to be evaluated once before sweeping.
# Can have post-dependencies: Evaluated every iteration.
#
# each dependency has a list of parameters, if these are in the sweep spec then they will need computing when one of the parameters changes.
#
# on top of this, the dependencies have positional and keyword arguments.
#
# pseudo code:
#   eval_res = {'eval1':[], 'eval2':[], ...}
#   eval_res_curr = {'eval1':[], 'eval2':[], ...}
#   for v in values:
#       for e in evaluables:
#           if e['depends'] is None:
#              res = e['func'](v,**e['kwargs'])
#              eval_res[e['name']].append(res)
#              eval_res_curr[e['name']] = res
#           else:
#              dep = e['depends']
#              res = e['func'](v,*eval_res_curr[dep['name']],**e['kwargs'])
#
# so eval_res_curr[dep['name']] needs to match the positional args of the dependent function
#
#####
#
# To prevent the regeneration of symbolic equations and resubstitutions during parameter sweeps:
#  - Need every evaluable entity to have a parameter dependency list.
#  - For a given evaluable, if none of its parameters appear in a sweep, generate the numeric expression right away.
#  - Parameterisations must have already been performed
#

## Base class for defining the Hamiltonian of a QED circuit
#
#
#
class HamilSpec(ds.TempData):
    
    ## Available basis representations
    __basis_repr = [
        "charge",
        "flux",
        "osc",
        "custom"
    ]
    
    ## Mode types
    __mode_types = [
        "osc", # DoFs have capacitive and inductive parts only
        "jos", # DoFs have capacitive, inductive and Josephson parts
        "isl", # DoFs have capacitive parts only
        "leg"  # DoFs have inductive parts only
    ]
    
    ## Consider listing dependencies here too
    __eval_avail = [
        "getHamiltonian",
        "getBranchCurrents",
        "getNodeVoltages",
        "getChargingEnergies",
        "getJosephsonEnergies",
        "getResonatorResponse"
    ]
    
    ## Original circuit parameter group name
    __orig_group_name = "orig"
    
    ## Initialise a Hamiltonian using a circuit specification
    def __init__(self, circuitspec, unit=units.Units("CQED1"), mode="node", include_subsystem=None):
        
        # Initialise the temporary data manager
        super().__init__()
        self.newSession(id(self))
        self.__use_temp = False
        
        # Assign the circuit
        self.cs = circuitspec
        
        # Nested dictionary for DoF operators, keyed by the relevant mode
        self.circ_operators = {}
        
        # Mode truncations and basis representation, also indicates whether
        # operators need to be considered for regeneration when parameters are changed.
        self.mode_truncations = {}
        
        # Included subsystems
        self.subsystem = include_subsystem
        
        # Parameter mappings to symbols
        self.orig_params_sym = self.cs.getParameterDict() # Current circuit parameters
        self.subsys_params_sym = {} # Subsystem circuit parameters
        self.all_params_sym = {} # All circuit parameters
        
        # Define the parameter collection
        self.params = pa.ParamCollection(list(self.orig_params_sym.keys()))
        
        # Parameter groups for keeping track of what needs to be recalculated
        self.params_groups = {
            self.__orig_group_name:list(self.orig_params_sym.keys())
        }
        
        # FIXME: Add something that tracks sweeps performed on this object and how they were dumped to temporary storage?
        
        # Check if a subsystem is required
        if self.subsystem is not None:
            
            # Eventually will allow more than one coupled subsystem
            #for k in self.subsystem:
            #    circ = self.cs.coupled_subsys[k]['circuitspec']
            
            # Update the parameters coming from the new circuit
            #self.params_sym = rcircuit.getParameterDict()
            self.all_params_sym = self.cs.rcircuit.getParameterDict()
            
            # Get the subsystem parameters
            self.subsys_params_sym = {k:v for k,v in self.all_params_sym.items() if k not in self.orig_params_sym.keys()}
            
            # Update the parameter collection
            self.params.addParameters(*list(self.all_params_sym.keys()))
            
            # Add new parameter groups
            self.params_groups[self.subsystem] = list(self.subsys_params_sym.keys())
            
            # Create an entry for the derived parameters
            self.subsys_dparams = {}
            self.subsys_dparams[self.subsystem] = {k:[] for k in self.cs.coupled_subsys[self.subsystem]['derived_parameters'].keys()}
            
        else:
            # All the parameters are the original system ones
            self.all_params_sym = self.orig_params_sym.copy()
        
        # For parameterisations
        self.params_subs = {}
        
        # Save the mode type
        self.Hmode = mode
        
        # For getting diagonal matrix elements
        self.getter_vecs = {}
        p = self.getPosList()
        for i,pos in enumerate(p):
            self.getter_vecs[pos] = sy.Matrix(qt.basis(len(p),i).data.todense())
        
        # Oscillator characteristics
        self.osc_params = {}
        
        # Set the unit system
        self.units = unit
        
        # Load default diagonaliser configuration
        self.setDiagConfig()
    
    # Called when deleting
    def __del__(self):
        self.clearSessionData()
    
    ###################################################################################################################
    #       Operator Generation and Functions
    ###################################################################################################################
    
    ##
    #
    #
    #
    def getPosList(self):
        if self.Hmode == "node":
            return list(self.cs.graph.nodes)[1:]
        elif self.Hmode == "branch":
            return list(self.cs.graph.edges)
    
    ##
    #
    #
    #
    def getOperatorList(self, trunc,basis="charge", osc_impedance=50.0):
        Q = None
        P = None
        D = None
        Ddag = None
        if basis == "charge":
            # For IJ modes the states are the eigenvectors of the following matrix
            q,s = (-qt.num(2*trunc + 1,trunc)).eigenstates()
            
            # Construct the flux states
            phik_list = []
            phik = None
            qm = [float(i)-trunc for i in range(2*trunc + 1)]
            for k in qm:
                phik = qt.basis(2*trunc + 1,0) - 1
                for j,qi in enumerate(qm):
                    phik += np.sqrt(1/(2*trunc + 1)) * np.exp(2j*np.pi*k*qi/(2*trunc + 1))*s[j]
                phik_list.append(phik)
            
            # From this build the flux operator
            phik_eigvals = [(float(i)-trunc)/(2*trunc + 1) for i in range(2*trunc + 1)]
            P = qt.qeye(2*trunc + 1) - qt.qeye(2*trunc + 1)
            for i,phik in enumerate(phik_list):
                P += phik_eigvals[i]*phik*phik.dag()
            
            # Get a simple charge number operator
            Q = -qt.num(2*trunc + 1)+float(trunc)
            
            # Generate displacement operators by first diagonalising the flux operator
            E,V = np.linalg.eigh(P.data.todense())
            
            # Create transformation matrices
            U = qt.Qobj(V)
            Uinv = qt.Qobj(np.linalg.inv(V))
            
            # Exponentiate the diagonal matrix
            D = U*qt.Qobj(np.diag(np.exp(-2j*np.pi*E)))*Uinv - qt.basis(2*trunc+1,2*trunc)*qt.basis(2*trunc+1,0).dag()#qt.basis(2*trunc+1,0)*qt.basis(2*trunc+1,2*trunc).dag()
            Ddag = D.dag()
            
        elif basis == "osc":
            
            # Using oscillator basis
            a = 1.0
            Q = 1j*np.sqrt(1/(2*osc_impedance))*(qt.create(trunc) - qt.destroy(trunc))*self.units.getPrefactor("ChgOsc")
            P = np.sqrt(osc_impedance/2)*(qt.create(trunc) + qt.destroy(trunc))*self.units.getPrefactor("FlxOsc")
            Pp = a*2*np.pi/pc.phi0*np.sqrt(pc.hbar)*P/self.units.getPrefactor("FlxOsc")
            
            # Generate displacement operators by first diagonalising the flux operator
            E,V = np.linalg.eigh(Pp.data.todense())
            
            # Create transformation matrices
            U = qt.Qobj(V)
            Uinv = qt.Qobj(np.linalg.inv(V))
            
            # Exponentiate the diagonal matrix
            D = U*qt.Qobj(np.diag(np.exp(1j*E)))*Uinv
            Ddag = D.dag()
        elif basis == "flux":
            raise Exception("flux basis not implemented yet.")
            pass # FIXME
        return Q,P,D,Ddag
    
    ##
    #
    #
    #
    def getSymbolicOscillatorParams(self):
        
        # Check truncations were defined
        if len(self.mode_truncations) != len(self.getPosList()):
            raise Exception("mode types and truncations not yet all set. Use setTruncation() for each degree of freedom first.")
        
        # Check oscillator basis is being used anywhere
        #if "osc" not in [self.mode_truncations[k][1] for k in self.getPosList()]:
        #    return
        
        # Check inv inductance matrix is diagonal
        #if not self.cs.getInverseInductanceMatrix(mode=self.Hmode).is_diagonal():
        #    raise Exception("inverse inductance matrix is not diagonal. Perform a transformation first.")
        Linv = self.cs.getInverseInductanceMatrix(mode=self.Hmode)
        Cinv = self.cs.getInverseCapacitanceMatrix(mode=self.Hmode)
        
        # Retrieve oscillator frequency and impedance symbolic expressions
        for pos in self.getPosList():
            Li = self.getter_vecs[pos].transpose()*Linv*self.getter_vecs[pos]
            Ci = self.getter_vecs[pos].transpose()*Cinv*self.getter_vecs[pos]
            if Li[0] == 0 or Ci[0] == 0:
                self.osc_params[pos] = None
            else:
                self.osc_params[pos] = {"frequency":sy.sqrt(Li[0]*Ci[0]),"impedance":sy.sqrt(Ci[0]/Li[0])}
    
    ##
    #
    #
    #
    def getCommutator(self, Q, P, basis="charge"):
        if basis == "charge": # Need to figure out the correction for this case
            dims = 2*pc.e*pc.phi0/pc.hbar
            return qt.commutator(P,Q)*dims
        elif basis == "osc":
            
            # Get highest number operator eigenvalue and associated eigenstate
            enum = Q.shape[0]
            vnum = qt.basis(enum,enum-1)
            
            # Create the correction matrix
            mat = 1 - (enum)*vnum*vnum.dag()
            
            # Invert it
            corrmat = qt.Qobj(np.linalg.inv(mat.data.todense()))
            
            # Get the correct commutator
            return qt.commutator(P,Q)*corrmat
    
    def setTruncation(self, pos, trunc, basis):
        """ Sets the truncation of the operators associated with the given branch or node.
        
        For charge and flux basis types, the truncation will be set to `2*trunc+1` to account for negative offsets of neutral charge or flux.
        
        :param pos: The node or branch.
        :type pos: int, tuple
        
        :param trunc: The truncation.
        :type trunc: int
        
        :param basis: The basis representation string, can be one defined in the `__basis_repr` private variable of this class.
        :type basis: str
        
        :return: None
        """
        if pos not in self.getPosList():
            raise Exception("pos '%s' does not exist." % repr(pos))
        self.mode_truncations[pos] = (trunc, basis, True if basis == "osc" else False)
    
    # FIXME: Implement me
    def setTruncations(self):
        pass
    
    ## Get Hilbert space size
    def getHilbertSpaceSize(self):
        """ Returns the Hilbert space size considering all currently defined operator truncations.
        
        :return: The Hilbert space size
        :rtype: int
        """
        ret = 1
        for m in list(self.mode_truncations.keys()):
            trunc, basis, update = self.mode_truncations[m]
            if basis == "charge":
                ret *= (2*trunc+1)
            elif basis == "osc":
                ret *= (trunc)
        return ret
    
    ## Expand operator Hilbert spaces and update mapping to associated symbols
    def getExpandedOperatorsMap(self):
        """ Creates all the operators associated with each node in the currently defined circuit. The operators are expanded into the total Hamiltonian Hilbert space.
        
        :return: None
        """
        #
        # FIXME: Need to pass in the oscillator basis parameters here
        #
        
        # Get the pos list for indexing the DoFs
        pos_list = self.getPosList()
        
        # Generate the Hilbert space expanders
        Ilist = np.empty([len(pos_list)],dtype=np.dtype(qt.Qobj))
        for i,pos in enumerate(pos_list):
            trunc, basis, update = self.mode_truncations[pos]
            if basis == "osc":
                Ilist[i] = qt.qeye(trunc)
            elif basis == "charge":
                Ilist[i] = qt.qeye(2*trunc+1)
        
        # Create mode operators
        Olist = np.empty([len(pos_list)],dtype=np.dtype(qt.Qobj))
        op_dict = {}
        for i,pos in enumerate(pos_list):
            self.circ_operators[pos] = []
            op_dict = {}
            trunc, basis, update = self.mode_truncations[pos]
            Q,P,D,Ddag = self.getOperatorList(trunc,basis=basis)
            
            # Indices minus the current index
            indices = list(range(len(pos_list)))
            indices.remove(i)
            
            # Current index is the operator
            Olist[i] = Q
            for j in indices:
                Olist[j] = Ilist[j]
            op_dict["charge"] = qt.tensor(Olist)
            
            Olist[i] = P
            for j in indices:
                Olist[j] = Ilist[j]
            op_dict["flux"] = qt.tensor(Olist)
            
            Olist[i] = D
            for j in indices:
                Olist[j] = Ilist[j]
            op_dict["disp"] = qt.tensor(Olist)
            
            Olist[i] = Ddag
            for j in indices:
                Olist[j] = Ilist[j]
            op_dict["disp_adj"] = qt.tensor(Olist)
            
            self.circ_operators[pos] = op_dict.copy()
    
    ##
    #
    #
    #
    def regenOps(self, pos, params={"basis":"osc","osc_impedance":50.0}):
        
        # Get the pos list for indexing the DoFs
        pos_list = self.getPosList()
        i = pos_list.index(pos)
        
        # Generate the Hilbert space expanders
        Ilist = np.empty([len(pos_list)],dtype=np.dtype(qt.Qobj))
        for i,pos in enumerate(pos_list):
            trunc, basis, update = self.mode_truncations[pos]
            if basis == "osc":
                Ilist[i] = qt.qeye(trunc)
            elif basis == "charge":
                Ilist[i] = qt.qeye(2*trunc+1)
        
        # Indices minus the current index
        indices = list(range(len(pos_list)))
        indices.remove(i)
        
        # Edit the circuit operators
        op_dict = {}
        Olist = np.empty([len(pos_list)],dtype=np.dtype(qt.Qobj))
        trunc, basis, update = self.mode_truncations[pos]
        Q,P,D,Ddag = self.getOperatorList(trunc,**params)
        
        Olist[i] = Q
        for j in indices:
            Olist[j] = Ilist[j]
        op_dict["charge"] = qt.tensor(Olist)
        
        Olist[i] = P
        for j in indices:
            Olist[j] = Ilist[j]
        op_dict["flux"] = qt.tensor(Olist)
        
        Olist[i] = D
        for j in indices:
            Olist[j] = Ilist[j]
        op_dict["disp"] = qt.tensor(Olist)
        
        Olist[i] = Ddag
        for j in indices:
            Olist[j] = Ilist[j]
        op_dict["disp_adj"] = qt.tensor(Olist)
        
        self.circ_operators[pos] = op_dict.copy()
        
    
    ##
    #
    #
    #
    def regenOp(self, pos, params={"basis":"osc","osc_impedance":50.0}):
        
        # Get the pos list for indexing the DoFs
        pos_list = self.getPosList()
        i = pos_list.index(pos)
        
        # Generate the Hilbert space expanders
        Ilist = np.empty([len(pos_list)],dtype=np.dtype(qt.Qobj))
        for i,pos in enumerate(pos_list):
            trunc, basis, update = self.mode_truncations[pos]
            if basis == "osc":
                Ilist[i] = qt.qeye(trunc)
            elif basis == "charge":
                Ilist[i] = qt.qeye(2*trunc+1)
        
        # Indices minus the current index
        indices = list(range(len(pos_list)))
        indices.remove(i)
        
        # Edit the circuit operators
        op_dict = {}
        Olist = np.empty([len(pos_list)],dtype=np.dtype(qt.Qobj))
        trunc, basis, update = self.mode_truncations[pos]
        Q,P,D,Ddag = self.getOperatorList(trunc,**params)
        
        Olist[i] = Q
        for j in indices:
            Olist[j] = Ilist[j]
        op_dict["charge"] = qt.tensor(Olist)
        
        Olist[i] = P
        for j in indices:
            Olist[j] = Ilist[j]
        op_dict["flux"] = qt.tensor(Olist)
        
        Olist[i] = D
        for j in indices:
            Olist[j] = Ilist[j]
        op_dict["disp"] = qt.tensor(Olist)
        
        Olist[i] = Ddag
        for j in indices:
            Olist[j] = Ilist[j]
        op_dict["disp_adj"] = qt.tensor(Olist)
        
        self.circ_operators[pos] = op_dict.copy()
    
    ###################################################################################################################
    #       Hamiltonian Building Functions
    ###################################################################################################################
    
    ## Apply a transformation to the mode coordinates
    #
    #
    #
    def applyTransformation(self, tmat):
        pass
    
    ##
    #
    #
    #
    def getChargeOpVector(self):
        pos = self.getPosList()
        pos_dofs = self.cs.getDoFSymbolList("charge")
        pos_ops = {k:v["charge"] for k,v in self.circ_operators.items()}

        # Reverse symbol map
        pos_dofs_rev = {v:k for k,v in pos_dofs.items()}
        arr = []
        for sym in self.cs.getChargeVector():
            pos = pos_dofs_rev[sym]
            arr.append(pos_ops[pos])
        self.Qnp = -np.asmatrix(arr,dtype=object).T
    
    ##
    #
    #
    #
    def getFluxOpVector(self):
        pos = self.getPosList()
        pos_dofs = self.cs.getDoFSymbolList("flux")
        pos_ops = {k:v["flux"] for k,v in self.circ_operators.items()}

        # Reverse symbol map
        pos_dofs_rev = {v:k for k,v in pos_dofs.items()}
        arr = []
        for sym in self.cs.getFluxVector():
            pos = pos_dofs_rev[sym]
            arr.append(pos_ops[pos])
        self.Pnp = np.asmatrix(arr,dtype=object).T
    
    ##
    #
    #
    #
    def getRightDispOpVector(self):
        pos = self.getPosList()
        pos_dofs1 = self.cs.getDoFSymbolList("disp")
        pos_dofs2 = self.cs.getDoFSymbolList("disp_adj")
        pos_ops1 = {k:v["disp"] for k,v in self.circ_operators.items()}
        pos_ops2 = {k:v["disp_adj"] for k,v in self.circ_operators.items()}

        # Reverse symbol map
        pos_dofs_rev1 = {v:k for k,v in pos_dofs1.items()}
        pos_dofs_rev2 = {v:k for k,v in pos_dofs2.items()}
        arr = []
        for sym in self.cs.getRightDisplacementOpVector(adjoint=False):
            if sym == 1:
                arr.append(1.0)
                continue
            try:
                pos = pos_dofs_rev1[sym]
                arr.append(pos_ops1[pos])
            except KeyError:
                pos = pos_dofs_rev2[sym]
                arr.append(pos_ops2[pos])
        self.Dr = np.asmatrix(arr,dtype=object).T
    
    ##
    #
    #
    #
    def getRightDispAdjOpVector(self):
        pos = self.getPosList()
        pos_dofs1 = self.cs.getDoFSymbolList("disp_adj")
        pos_dofs2 = self.cs.getDoFSymbolList("disp")
        pos_ops1 = {k:v["disp_adj"] for k,v in self.circ_operators.items()}
        pos_ops2 = {k:v["disp"] for k,v in self.circ_operators.items()}

        # Reverse symbol map
        pos_dofs_rev1 = {v:k for k,v in pos_dofs1.items()}
        pos_dofs_rev2 = {v:k for k,v in pos_dofs2.items()}
        arr = []
        for sym in self.cs.getRightDisplacementOpVector(adjoint=True):
            if sym == 1:
                arr.append(1.0)
                continue
            try:
                pos = pos_dofs_rev1[sym]
                arr.append(pos_ops1[pos])
            except KeyError:
                pos = pos_dofs_rev2[sym]
                arr.append(pos_ops2[pos])
        self.Dr_adj = np.asmatrix(arr,dtype=object).T
    
    ##
    #
    #
    #
    def getLeftDispOpMatrix(self):
        pos = self.getPosList()
        pos_dofs1 = self.cs.getDoFSymbolList("disp")
        pos_dofs2 = self.cs.getDoFSymbolList("disp_adj")
        pos_ops1 = {k:v["disp"] for k,v in self.circ_operators.items()}
        pos_ops2 = {k:v["disp_adj"] for k,v in self.circ_operators.items()}

        # Reverse symbol map
        pos_dofs_rev1 = {v:k for k,v in pos_dofs1.items()}
        pos_dofs_rev2 = {v:k for k,v in pos_dofs2.items()}
        arr = []
        for sym in self.cs.getLeftDisplacementOpMatrix(adjoint=False,as_vec=True):
            if sym == 1:
                arr.append(1.0)
                continue
            try:
                pos = pos_dofs_rev1[sym]
                arr.append(pos_ops1[pos])
            except KeyError:
                pos = pos_dofs_rev2[sym]
                arr.append(pos_ops2[pos])
        self.Dl = np.asmatrix(np.diag(np.array(arr,dtype=object)),dtype=object)
    
    ##
    #
    #
    #
    def getLeftDispAdjOpMatrix(self):
        pos = self.getPosList()
        pos_dofs1 = self.cs.getDoFSymbolList("disp_adj")
        pos_dofs2 = self.cs.getDoFSymbolList("disp")
        pos_ops1 = {k:v["disp_adj"] for k,v in self.circ_operators.items()}
        pos_ops2 = {k:v["disp"] for k,v in self.circ_operators.items()}

        # Reverse symbol map
        pos_dofs_rev1 = {v:k for k,v in pos_dofs1.items()}
        pos_dofs_rev2 = {v:k for k,v in pos_dofs2.items()}
        arr = []
        for sym in self.cs.getLeftDisplacementOpMatrix(adjoint=True,as_vec=True):
            if sym == 1:
                arr.append(1.0)
                continue
            try:
                pos = pos_dofs_rev1[sym]
                arr.append(pos_ops1[pos])
            except KeyError:
                pos = pos_dofs_rev2[sym]
                arr.append(pos_ops2[pos])
        self.Dl_adj = np.asmatrix(np.diag(np.array(arr,dtype=object)),dtype=object)
    
    ##
    #
    #
    #
    def getOpVectors(self):
        self.getChargeOpVector()
        self.getFluxOpVector()
        self.getRightDispOpVector()
        self.getRightDispAdjOpVector()
        self.getLeftDispOpMatrix()
        self.getLeftDispAdjOpMatrix()
    
    ##
    #
    #
    #
    def prepareOperators(self):
        
        # Generate symbolics
        self.getSymbolicOscillatorParams()
        self.getSymbolicExpressions()
        
        # Generate numericals
        self.getExpandedOperatorsMap()
        self.getOpVectors()
    
    ##
    #
    #
    #
    def prepareSubsystemOperators(self):
        pass
    
    ##
    #
    #
    #
    def getSymbolicExpressions(self):
        
        # Check for parameterisations
        subs = dict([(self.all_params_sym[k],v) for k,v in self.params_subs.items()])
        
        if self.subsystem is not None:
            
            # Generate final symbolic expressions
            self.Cinv = self.cs.rcircuit_sys_matrices['Cinv'].subs(subs)
            self.Linv = self.cs.rcircuit_sys_matrices['Linv'].subs(subs)
            
            # Get branch inverse inductance matrix for branch current calculations
            if self.Hmode == "node":
                self.Linv_b = None # FIXME: Need to save it in subsystem to get loaded version
                #self.Linv_b = self.cs.rcircuit_sys_matrices['Linv_b'].subs(subs)
            # Get node inverse capacitance matrix for node charge calculations
            elif self.Hmode == "branch":
                self.Cinv_n = None
                #self.Cinv_n = self.cs.rcircuit_sys_matrices['Cinv_n'].subs(subs)
            
            # Get derived parameters and associated units
            self.dp = {}
            self.dpu = {}
            for k,v in self.cs.coupled_subsys[self.subsystem]['derived_parameters'].items():
                self.dp[k] = v.subs(subs)
                self.dpu[k] = self.units.getPrefactor(self.cs.coupled_subsys[self.subsystem]['derived_parameter_units'][k])
            
        else:
            
            # Generate final symbolic expressions
            self.Cinv = self.cs.getInverseCapacitanceMatrix(mode=self.Hmode).subs(subs)
            self.Linv = self.cs.getInverseInductanceMatrix(mode=self.Hmode).subs(subs)
            
            # Get branch inverse inductance matrix for branch current calculations
            if self.Hmode == "node":
                self.Linv_b = self.cs.getInverseInductanceMatrix(mode='branch').subs(subs)
            # Get node inverse capacitance matrix for node charge calculations
            elif self.Hmode == "branch":
                self.Cinv_n = self.cs.getInverseCapacitanceMatrix(mode='node').subs(subs)
        
        # Symbolic expressions independent of a coupled subsystem
        self.Jvec = self.cs.getJosephsonVector().subs(subs)
        self.Qb = self.cs.getChargeBiasVector(mode=self.Hmode).subs(subs)
        self.Pb = self.cs.getFluxBiasVector(mode=self.Hmode).subs(subs)
        self.Pbm = self.cs.getFluxBiasMatrix(mode="branch").subs(subs)
        self.Z = dict(zip(self.getPosList(),[self.osc_params[k]["impedance"].subs(subs) if self.osc_params[k] is not None else None for k in self.getPosList()]))
        self.w = dict(zip(self.getPosList(),[self.osc_params[k]["frequency"].subs(subs) if self.osc_params[k] is not None else None for k in self.getPosList()]))
    
    ###################################################################################################################
    #       Parameterisation Functions
    ###################################################################################################################
    
    ##
    #
    #
    #
    def addParameterisation(self, old_param, expr, name_map):
        
        # Register the substitution to perform
        self.params_subs[old_param] = expr
        
        # Add the new symbols associated with parameters
        self.all_params_sym.update(name_map)
        
        # Add the new parameters
        for k,v in name_map.items():
            self.params.addParameter(k)
        
        # FIXME: Remove old_param if it does not appear in expr
    
    ##
    #
    #
    #
    def rmParameterisation(self, old_param):
        
        # Unregister the substitution
        del self.params_subs[old_param]
    
    ##
    #
    #
    #
    def applyParameterisations(self):
        pass
    
    ###################################################################################################################
    #       Numerical Hamiltonian Generation
    ###################################################################################################################
    
    ## 
    #
    #
    #
    def substitute(self, params_dict):
        
        # Set the parameter values
        for k,v in params_dict.items():
            self.params.setParameterValue(k,v)
        
        # If some parameters are omitted from params_dict, those saved in self.params will be used
        subs = {}
        for k,v in self.all_params_sym.items():
            subs[v] = self.params.getParameterValue(k)
        
        # Get the oscillator parameters
        self.Znp = dict(zip(self.getPosList(),[float(self.Z[k].subs(subs))*self.units.getPrefactor("Impe") if self.Z[k] is not None else None for k in self.getPosList()]))
        self.wnp = dict(zip(self.getPosList(),[float(self.w[k].subs(subs))*self.units.getPrefactor("Freq") if self.w[k] is not None else None for k in self.getPosList()]))
        
        # Regenerate the oscillator basis operators FIXME
        #for k,v in self.Znp.items():
        #    if v is None:
        #        continue
        #    self.regenOp(k,params={"basis":"osc","osc_impedance":self.Znp[k]})
        
        # Substitute circuit parameters
        self.Cinvnp = np.asmatrix(self.Cinv.subs(subs),dtype=np.float64)
        self.Linvnp = np.asmatrix(self.Linv.subs(subs),dtype=np.float64)
        self.Jvecnp = np.asmatrix(self.Jvec.subs(subs),dtype=np.float64).T
        
        # Substitute external biases
        self.Qbnp = np.asmatrix(self.Qb.subs(subs),dtype=np.float64) # x 2e
        self.Pbsm = np.asmatrix(self.Pbm.subs(subs),dtype=np.float64)
        self.Pbnp = np.asmatrix(self.Pb.subs(subs),dtype=np.float64) # x Phi0
        
        # Generate exponentiated biases
        # FIXME: This should be removed by expressing the exponentiated biases in 
        # terms of the flux bias symbol
        Pexp1 = []
        Pexp2 = []
        for i in range(self.Pbsm.shape[0]):
            Pexp1.append(np.exp(2j*np.pi*self.Pbsm[i,i]))
            Pexp2.append(np.exp(-2j*np.pi*self.Pbsm[i,i]))
        self.Pexpbnp = np.asmatrix(np.diag(Pexp1))
        self.Pexpbnpc = np.asmatrix(np.diag(Pexp2))
        
        # Get branch inverse inductance matrix for branch current calculations
        if self.Hmode == 'node':
            self.Linvnp_b = np.asmatrix(self.Linv_b.subs(subs),dtype=np.float64)
        
        # Calculate subsystem derived parameters
        if self.subsystem is not None:
            self.dpnp = {}
            for k,v in self.dp.items():
                self.dpnp[k] = np.float64(v.subs(subs))*self.dpu[k]
    
    ## Substitute parameters before a sweep to minimise total number of substitutions
    #
    # Called after an ndSweep has been created
    #
    def _presub(self):
        
        # FIXME: Do we want to use only prepareOperators() to register parameterisations, or do we leave this as it is?
        
        # Check for parameterisations
        p_subs = dict([(self.all_params_sym[k],v) for k,v in self.params_subs.items()])
        
        # Check which systems will need recalculations
        recalc_map = self._check_sys_sweep_params()
        
        # Set the parameters that are not being swept
        subs = self._set_non_swept_params()
        
        # Generate final symbolic expressions
        if self.subsystem is not None:
            
            # Evaluate subsystem parameters
            #for subsys in self.subsystem: # FIXME: Use this when possible to add more than one subsystem
            self.dp_pre = {}
            for k,v in self.cs.coupled_subsys[self.subsystem]['derived_parameters'].items():
                self.dp_pre[k] = v.subs(p_subs).subs(subs)
            
            # Use loaded component matrices
            self.Cinv_pre = self.cs.rcircuit_sys_matrices['Cinv'].subs(p_subs).subs(subs)
            self.Linv_pre = self.cs.rcircuit_sys_matrices['Linv'].subs(p_subs).subs(subs)
            
            # Get loaded (FIXME) branch inverse inductance matrix for branch current calculations
            if self.Hmode == 'node':
                self.Linv_b_pre = self.cs.getInverseInductanceMatrix(mode='branch').subs(p_subs).subs(subs)
            
            # Reset derived parameters arrays
            self.subsys_dparams[self.subsystem] = {k:[] for k in self.cs.coupled_subsys[self.subsystem]['derived_parameters'].keys()}
            
        else:
            self.Cinv_pre = self.cs.getInverseCapacitanceMatrix(mode=self.Hmode).subs(p_subs).subs(subs)
            self.Linv_pre = self.cs.getInverseInductanceMatrix(mode=self.Hmode).subs(p_subs).subs(subs)
            
            # Get branch inverse inductance matrix for branch current calculations
            if self.Hmode == 'node':
                self.Linv_b_pre = self.cs.getInverseInductanceMatrix(mode='branch').subs(p_subs).subs(subs)
        
        self.Jvec_pre = self.cs.getJosephsonVector().subs(p_subs).subs(subs)
        self.Qb_pre = self.cs.getChargeBiasVector(mode=self.Hmode).subs(p_subs).subs(subs)
        #self.Pb_pre = self.cs.getFluxBiasVector(mode=self.Hmode).subs(p_subs).subs(subs)
        self.Pbm_pre = self.cs.getFluxBiasMatrix(mode="branch").subs(p_subs).subs(subs)
        self.getSymbolicOscillatorParams()
        self.Z_pre = dict(zip(self.getPosList(),[self.osc_params[k]["impedance"].subs(p_subs).subs(subs) if self.osc_params[k] is not None else None for k in self.getPosList()]))
        self.w_pre = dict(zip(self.getPosList(),[self.osc_params[k]["frequency"].subs(p_subs).subs(subs) if self.osc_params[k] is not None else None for k in self.getPosList()]))
        
        # Find which operators will need to be regenerated for each sweep
        self._get_regen_ops_list()
    
    ## Substitute parameters during a sweep
    #
    #
    #
    def _postsub(self, params):
    
        # Set the parameter values
        subs = self._set_swept_params(params)
        
        # Get the oscillator parameters FIXME
        #self.Znp = dict(zip(self.getPosList(),[float(self.Z_pre[k].subs(subs))*self.units.getPrefactor("Impe") if self.Z[k] is not None else None for k in self.getPosList()]))
        #self.wnp = dict(zip(self.getPosList(),[float(self.w_pre[k].subs(subs))*self.units.getPrefactor("Freq") if self.w[k] is not None else None for k in self.getPosList()]))
        
        # Regenerate required operators FIXME
        #for k in self.Rlist:
        #    self.regenOp(k,params={"basis":"osc","osc_impedance":self.Znp[k]})
        
        # Substitute circuit parameters
        self.Cinvnp = np.asmatrix(self.Cinv_pre.subs(subs),dtype=np.float64)
        self.Linvnp = np.asmatrix(self.Linv_pre.subs(subs),dtype=np.float64)
        self.Jvecnp = np.asmatrix(self.Jvec_pre.subs(subs),dtype=np.float64).T
        if self.Hmode == 'node':
            self.Linvnp_b = np.asmatrix(self.Linv_b_pre.subs(subs),dtype=np.float64)
        
        # Substitute external biases
        self.Qbnp = np.asmatrix(self.Qb_pre.subs(subs),dtype=np.float64) # x 2e
        self.Pbsm = np.asmatrix(self.Pbm_pre.subs(subs),dtype=np.float64)
        #self.Pbnp = np.asmatrix(self.Pb_pre.subs(subs),dtype=np.float64) # x Phi0
        
        # Generate exponentiated biases
        # FIXME: This should be removed by expressing the exponentiated biases in 
        # terms of the flux bias symbol
        Pexp1 = []
        Pexp2 = []
        for i in range(self.Pbsm.shape[0]):
            Pexp1.append(np.exp(2j*np.pi*self.Pbsm[i,i]))
            Pexp2.append(np.exp(-2j*np.pi*self.Pbsm[i,i]))
        self.Pexpbnp = np.asmatrix(np.diag(Pexp1))
        self.Pexpbnpc = np.asmatrix(np.diag(Pexp2))
        
        # Calculate subsystem derived parameters
        if self.subsystem is not None:
            self.dpnp = {}
            for k,v in self.dp_pre.items():
                self.dpnp[k] = np.float64(v.subs(subs))*self.dpu[k]
    
    ###################################################################################################################
    #       Evaluables
    ###################################################################################################################
    
    ## Return the numerical Hamiltonian given a parameter collection
    #
    #
    #
    def getHamiltonian(self):
        
        # Get charging energy
        self.Hq = self.units.getPrefactor("Ec")*0.5*util.mdot((self.Qnp+self.Qbnp).T,self.Cinvnp,self.Qnp+self.Qbnp)
        
        # Get flux energy
        if len(self.cs.getEdgeList()) == 2 and len(self.cs.getNodeList()) == 1:
            self.Hp = self.units.getPrefactor("El")*0.5*util.mdot((self.Pnp).T,self.Linvnp,self.Pnp)
        else:
            # FIXME: Need to check that a suitable basis is used to include the bias term here
            #self.Hp = self.units.getPrefactor("El")*0.5*util.mdot((self.Pnp+self.Pbnp).T,self.Linvnp,self.Pnp+self.Pbnp)
            self.Hp = self.units.getPrefactor("El")*0.5*util.mdot((self.Pnp).T,self.Linvnp,self.Pnp)
        
        # Get Josephson energy
        self.Hj = -self.units.getPrefactor("Ej")*0.5*(util.mdot(self.Jvecnp,self.Pexpbnpc,self.Dl_adj,self.Dr_adj) + util.mdot(self.Jvecnp,self.Pexpbnp,self.Dl,self.Dr))
        
        # Total Hamiltonian
        self.Ht = (self.Hq+self.Hp+self.Hj)[0,0]
        return self.Ht
    
    ##
    #
    #
    #
    def getBranchCurrents(self):
        
        # Get edge components
        x = nx.get_edge_attributes(self.cs.graph,'edge_components')
        node_list = self.cs.getNodeList()
        edge_list = self.cs.getEdgeList()
        
        # Branch currents
        P = self.Pnp #+self.Pbnp
        J = 0.5j*(util.mdot(self.Pexpbnpc,self.Dl_adj,self.Dr_adj) - util.mdot(self.Pexpbnp,self.Dl,self.Dr))
        #R = np.asmatrix(self.cs.getNodeToBranchMatrix(),dtype=np.float64)
        self.Iops = {}
        for i,edge in enumerate(edge_list):
            # Check if edge has an inductor
            if len(x[edge]['L']) > 0:
                if self.Hmode == 'branch':
                    self.Iops[edge] = self.units.getPrefactor("IopL")*(P2-P1)*self.Linvnp[i,i]
                else:
                    n1,n2 = edge
                    if n1 > 0:
                        P1 = P[0,node_list.index(n1)]
                    else:
                        P1 = 0.0
                    if n2 > 0:
                        P2 = P[0,node_list.index(n2)]
                    else:
                        P2 = 0.0
                    
                    # Take difference of node fluxes of corresponding branch and use *branch* inverse inductance matrix
                    #self.Iops[edge] = self.units.getPrefactor("IopL")*(P2-P1)*util.mdot(R,self.Linvnp)[i,i]
                    self.Iops[edge] = self.units.getPrefactor("IopL")*(P2-P1)*self.Linvnp_b[i,i]
            else:
                self.Iops[edge] = self.units.getPrefactor("IopJ")*J.T[0,i]*self.Jvecnp[0,i]
        return self.Iops
    
    ##
    #
    #
    #
    def getNodeVoltages(self):
        
        # Get edge components
        x = nx.get_edge_attributes(self.cs.graph,'edge_components')
        node_list = self.cs.getNodeList()
        
        # Node voltages
        Q = self.Qnp + self.Qbnp
        self.Vops = {}
        for i,node in enumerate(node_list):
            self.Vops[node] = self.units.getPrefactor("Vop")*Q[i,0]*self.Cinvnp[i,i]
        return self.Vops
    
    ## Get the expectation value of an operator
    #
    #
    #
    def getOpExpectationValue(self, op, lket, rket):
        if type(op) == qt.qobj.Qobj and type(lket) == qt.qobj.Qobj and type(rket) == qt.qobj.Qobj:
            return (lket.dag()*op*rket)[0][0][0]
        elif type(op) in [list, np.ndarray] and type(lket) == qt.qobj.Qobj and type(rket) == qt.qobj.Qobj:
            return np.array([(lket.dag()*opi*rket)[0][0][0] for opi in op])
        elif type(op) in [list, np.ndarray] and type(lket) in [list, np.ndarray] and type(rket) in [list, np.ndarray]:
            return np.array([(lket[i].dag()*op[i]*rket[i])[0][0][0] for i in range(len(op))])
        elif type(op) == qt.qobj.Qobj and type(lket) in [list, np.ndarray] and type(rket) in [list, np.ndarray]:
            return np.array([(lket[i].dag()*op*rket[i])[0][0][0] for i in range(len(lket))])
        else:
            raise Exception("incompatible input types for op (%s), lket (%s) and rket (%s)." % (repr(type(op)), repr(type(lket)), repr(type(rket))))
    
    ##
    #
    #
    #
    def getBranchCurrentOperator(self, edge):
        # FIXME: Check edge exists?
        return self.Iops[edge]
    
    ##
    #
    #
    #
    def getNodeVoltageOperator(self, node):
        # FIXME: Check node exists?
        return self.Vops[node]
    
    ##
    #
    #
    #
    def getSubsystemDerivedParameters(self, subsystem, which=None):
        if which is None:
            return self.subsys_dparams[subsystem]
        else:
            return {k:self.subsys_dparams[subsystem][k] for k in which}
    
    """
    ## Get charging energies associated with each mode
    #
    #
    #
    def getChargingEnergies(self,params_dict,symbolic=False):
        
        # Check for parameterisations
        p_subs = dict([(self.params_sym[k],v) for k,v in self.params_subs.items()])
        
        # FIXME: Make separate function for symbolic expressions
        if symbolic:
            ret = {}
            for i,pos in enumerate(self.getPosList()):
                ret[pos] = 0.5 * self.Cinv.subs(p_subs)[i,i] * self.cs.qcp**2
            return ret
        
        # Set the parameter values
        for k,v in params_dict.items():
            self.params.setParameterValue(k,v)
        
        # If some parameters are omitted from params_dict, those saved in self.params will be used
        subs = {}
        for k,v in self.params_sym.items():
            subs[v] = self.params.getParameterValue(k)
        
        # Use diagonal elements of the inverse capacitance matrix
        ret = {}
        self.Cinvnp = np.asmatrix(self.Cinv.subs(p_subs).subs(subs),dtype=np.float64)
        for i,pos in enumerate(self.getPosList()):
            ret[pos] = 0.5 * self.Cinvnp[i,i] * self.Ec_unit
        return ret
    """
    
    def getSymbolicChargingEnergies(self, node=None):
        # Check for parameterisations
        p_subs = dict([(self.all_params_sym[k],v) for k,v in self.params_subs.items()])
        
        if node is None:
            ret = {}
            for i,pos in enumerate(self.getPosList()):
                ret[pos] = 0.5 * self.Cinv.subs(p_subs)[i,i] * self.cs.qcp**2
            return ret
        else:
            i = self.getPosList().index(node)
            return 0.5 * self.Cinv.subs(p_subs)[i,i] * self.cs.qcp**2
    
    def getChargingEnergies(self, node=None):
        if node is None:
            ret = {}
            for i,pos in enumerate(self.getPosList()):
                ret[pos] = 0.5 * self.Cinvnp[i,i] * self.units.getPrefactor("Ec")
            return ret
        else:
            i = self.getPosList().index(node)
            return 0.5 * self.Cinvnp[i,i] * self.units.getPrefactor("Ec")
    
    ## Get flux energies associated with each mode
    #
    #
    #
    def getFluxEnergies(self, edge=None):
        
        # Use diagonal elements of the inverse inductance matrix
        
        pass
    
    """
    ## Get Josephson energies associated with each JJ
    #
    #
    #
    def getJosephsonEnergies(self,params_dict,symbolic=False):
        
        # Check for parameterisations
        p_subs = dict([(self.params_sym[k],v) for k,v in self.params_subs.items()])
        
        # FIXME: Make separate function for symbolic expressions
        if symbolic:
            ret = {}
            for i,edge in enumerate(self.cs.getEdgeList()):
                ret[edge] = self.Jvec.subs(p_subs)[i] * self.cs.phi0/(2*self.cs.pi)
            return ret
        
        # Set the parameter values
        for k,v in params_dict.items():
            self.params.setParameterValue(k,v)
        
        # If some parameters are omitted from params_dict, those saved in self.params will be used
        subs = {}
        for k,v in self.params_sym.items():
            subs[v] = self.params.getParameterValue(k)
        
        # Use the Josephson vector
        ret = {}
        self.Jvecnp = np.asmatrix(self.Jvec.subs(p_subs).subs(subs),dtype=np.float64).T
        for i,edge in enumerate(self.cs.getEdgeList()):
            ret[edge] = self.Jvecnp[0,i] * self.Ej_unit
        return ret
    """
    
    def getSymbolicJosephsonEnergies(self, edge=None):
        # Check for parameterisations
        p_subs = dict([(self.all_params_sym[k],v) for k,v in self.params_subs.items()])
        
        if edge is None:
            ret = {}
            for i,edge in enumerate(self.cs.getEdgeList()):
                ret[edge] = self.Jvec.subs(p_subs)[i] * self.cs.phi0/(2*self.cs.pi)
            return ret
        else:
            i = self.cs.getEdgeList().index(edge)
            return self.Jvec.subs(p_subs)[i] * self.cs.phi0/(2*self.cs.pi)
    
    def getJosephsonEnergies(self, edge=None):
        if edge is None:
            ret = {}
            for i,edge in enumerate(self.cs.getEdgeList()):
                ret[edge] = self.Jvecnp[0,i] * self.units.getPrefactor("Ej")
            return ret
        else:
            i = self.cs.getEdgeList().index(edge)
            return self.Jvecnp[0,i] * self.units.getPrefactor("Ej")
    
    ## Get linear resonator response
    #
    #  FIXME: When multiple subsystems available, have subsystem keyword and rename to getSubsystemResponse
    #
    #  The current implementation is the case subsys_type = 'linear_resonator'
    #
    def getResonatorResponse(self, E, V, nmax=100, eval_dp=False):
        
        # Save the derived parameter values for each sweep value
        if eval_dp:
            for k in self.cs.coupled_subsys[self.subsystem]['derived_parameters'].keys():
                self.subsys_dparams[self.subsystem][k].append(self.dpnp[k])
        
        # Get the operator associated with selected node
        cpl_node = self.cs.coupled_subsys[self.subsystem]['cpl_node']
        index = self.getPosList().index(cpl_node)
        Op = self.Qnp[index,0] + self.Qbnp[index,0]
        
        # Get coupling terms
        E = E - E[0]
        tmax = len(E)
        nmax = nmax + 1 + tmax
        norm = (V[0].dag()*Op*V[1])[0][0][0]
        g_list = []
        for i in range(tmax-1):
            g_list.append((V[i].dag()*Op*V[i+1])[0][0][0])
        
        # Apply the circuit derived prefactor
        g_list = self.dpnp['gC'] * np.abs(np.array(g_list)/norm)
        
        # Diagonalise RWA strips
        order = np.zeros(tmax,dtype=np.int)
        diag_bare = np.array([-i*self.dpnp['w'] + E[i]
                              for i in range(tmax)])
        eigensolver_order = np.linalg.eigvalsh(np.diag(diag_bare))
        for i in range(tmax):
            index, = np.where(eigensolver_order==diag_bare[i])
            order[i] = index[0]
        Erwa = [0.0]*nmax
        diagonal_elements = None
        offdiagonal_elements = None
        strip_H = None
        e = None
        for n in range(nmax):
            diagonal_elements = np.array([(n-i)*self.dpnp['w'] + E[i] for i in range(tmax)])
            offdiagonal_elements = np.array([g_list[i]*np.sqrt((n-i)*(n-i>0)) for i in range(tmax-1)])
            strip_H = (np.diag(diagonal_elements) + np.diag(offdiagonal_elements, 1) +
                       np.diag(offdiagonal_elements, -1))
            e = sc.linalg.eigvalsh(strip_H)
            Erwa[n] = np.array([e[i].real for i in order])
        
        return np.array(Erwa)
    
    ## Calculate sparsity of operators
    #
    #
    #
    def sparsity(self, op):
        # FIXME: There must be a better way to do this
        op_data = op.data.todense()
        s = 1 - np.count_nonzero(op_data)/self.getHilbertSpaceSize()**2
        del op_data
        return s
    
    ###################################################################################################################
    #       Diagonaliser Configuration
    ###################################################################################################################
    
    ## Configure diagonalizer
    #
    #
    #
    def setDiagConfig(self, eigvalues=5, get_vectors=False, sparse=False, sparsesolveropts={"sigma":0,"mode":"normal","maxiter":None,"tol":0}):
        self.diagonalizer_config = {
            'kwargs':{
                'eigvalues':eigvalues,
                'get_vectors':get_vectors,
                'sparsesolveropts':sparsesolveropts
            },
            'sparse':sparse
        }
        
        # Choose the diagonalizer function and matrix conversion operation
        if sparse:
            self.diagonalizer_config['func'] = util.diagSparseH
        else:
            self.diagonalizer_config['func'] = util.diagDenseH
    
    ## Get diagonalizer configuration
    #
    #
    #
    def getDiagConfig(self):
        return self.diagonalizer_config
    
    ## Assumes always Hermitian
    #
    # M should be a qutip Hermitian operator
    #
    def diagonalize(self, M):
        return self.diagonalizer_config['func'](M, **self.diagonalizer_config['kwargs'])
    
    ###################################################################################################################
    #       Parameter Collection Wrapper Functions and Extended Functions
    ###################################################################################################################
    
    ## Create a sweep specification for a single parameter.
    def sweepSpec(self, *args, **kwargs):
        return self.params.paramSweepSpec(*args,**kwargs)
    
    ## Create an evaluation specification for a single function.
    def evalSpec(self, func, diag, depends, **kwargs):
        return {'eval':func, 'diag':diag, 'depends':depends, 'kwargs':kwargs}
    
    ## Set the value of a parameter.
    def setParameterValue(self, name, value):
        return self.params.setParameterValue(name,value)
    
    ## Get the value of a parameter.
    def getParameterValue(self, name):
        return self.params.getParameterValue(name)
    
    ## Set many parameter values.
    def setParameterValues(self, *name_value_pairs):
        return self.params.setParameterValues(*name_value_pairs)
    
    ## Get many parameter values.
    def getParameterValues(self, *names):
        return self.params.getParameterValues(*names, dictionary=True)
    
    ## Gets all parameters
    def getParameterValuesDict(self):
        return self.params.getParameterValuesDict()
    
    def getParameterSweep(self, name):
        return self.params.getParameterSweep(name)
    
    def getParameterNames(self):
        return self.params.getParameterNamesList()
    
    def getPrefactor(self, name):
        return self.units.getPrefactor(name)
    
    def getSubsystemList(self):
        return list(self.cs.coupled_subsys.keys())
    
    def getSubsystemDerivedParametersList(self, subsystem):
        return list(self.cs.coupled_subsys[subsystem]['derived_parameters'].keys())
    
    def getSubsystemDerivedParameter(self, param):
        return self.subsys_dparams[self.subsystem][param]
    
    ###################################################################################################################
    #       Parameter Sweep Functions
    ###################################################################################################################
    
    ## Compute parameter sweeps
    #
    #
    #
    def paramSweep(self, sweep_spec, eval_spec=[{'eval':'getHamiltonian','diag':True, 'depends':None, 'kwargs':{}}], timesweep=False):
        
        # Time initialisation
        if timesweep:
            init_time = time.time()
        
        # Generate sweep grid
        self.params.ndSweep(sweep_spec)
        
        # FIXME: Determine if we should be saving the data to temp files rather than in RAM:
        # Use the diagonaliser configuration, the requested evaluation functions, and the total number of sweep setpoints that will be used.
        self.__use_temp = True
        if self.__use_temp:
            tmp_results = []
        
        # FIXME: Remove this
        #self.eval_dp = True
        
        # Do pre-substitutions to avoid repeating un-necessary substitutions in loops
        self._presub()
        
        # FIXME: Check that all symbolic variables have an associated value at this point
        
        # Do the requested evaluations
        if len(eval_spec) > 1:
            
            # Prepare the results structure
            results = {}
            for entry in eval_spec:
                results[entry['eval']] = []
            
            # Time loop
            if timesweep:
                loop_time = time.time()
            for i in range(self.params.sweep_grid_npts):
                # Do the post-substitutions
                self._postsub(dict([(k,v[i]) for k,v in self.params.sweep_grid_c.items()]))
                
                # Get requested evaluables
                E = None
                V = None
                for entry in eval_spec:
                    
                    # Check if this evaluable depends on another
                    if entry['depends'] is not None:
                        try:
                            if self.__use_temp:
                                dep = results[entry['depends']]
                            else:
                                dep = results[entry['depends']][i]
                        except:
                            raise Exception("eval spec with 'depends':'%s' entry should be specified after the one it depends on ('%s'), or Possibly invalid 'depends' value." % (entry['depends'],entry['eval'])) # FIXME
                        
                        # In almost every case the depends will be on the eigenvalues and eigenvectors of the independent eval spec
                        try:
                            E, V = dep
                        except:
                            raise Exception("need eigenvectors for 'depends'")
                    
                    # Check if evaluation depends on eigenvalues and eigenvectors
                    if V is not None:
                        M = getattr(self,entry['eval'])(E,V,**entry['kwargs'])
                    else:
                        M = getattr(self,entry['eval'])(**entry['kwargs'])
                    
                    # Check if diagonalisation is required
                    if self.__use_temp:
                        if entry['diag']:
                            results[entry['eval']] = self.diagonalize(M)
                        else:
                            results[entry['eval']] = M
                    else:
                        if entry['diag']:
                            results[entry['eval']].append(self.diagonalize(M))
                        else:
                            results[entry['eval']].append(M)
                    E = None
                    V = None
                
                if self.__use_temp:
                    # Write to temp file
                    f = self.writePart(results)
                    tmp_results.append(f)
            
            # Convert the result entries to ndarray
            if not self.__use_temp:
                for entry in eval_spec:
                    results[entry['eval']] = np.array(results[entry['eval']])
        
        # Do the single requested evaluation
        else:
            results = []
            entry = eval_spec[0]
            
            # Time loop
            if timesweep:
                loop_time = time.time()
            for i in range(self.params.sweep_grid_npts):
                # Do the post-substitutions
                self._postsub(dict([(k,v[i]) for k,v in self.params.sweep_grid_c.items()]))
                
                # Get requested evaluable
                M = getattr(self,entry['eval'])(**entry['kwargs'])
                if self.__use_temp:
                    if entry['diag']:
                        results = self.diagonalize(M)
                    else:
                        results = M
                else:
                    if entry['diag']:
                        results.append(self.diagonalize(M))
                    else:
                        results.append(M)
                
                if self.__use_temp:
                    # Write to temp file
                    f = self.writePart(results)
                    tmp_results.append(f)
            
            # Convert results to ndarray
            if not self.__use_temp:
                results = np.array(results)
        
        # Report timings
        if timesweep:
            end_time = time.time()
            print ("Parameter Sweep Duration:")
            print ("  Initialization:\t%.3f s" % (loop_time-init_time))
            print ("  Loop duration:\t%.3f s" % (end_time-loop_time))
            print ("  Avg iteration:\t%.3f s" % ((end_time-loop_time)/self.params.sweep_grid_npts))
        if self.__use_temp:
            return tmp_results
        else:
            return results
        
    
    ## Compute parameter sweeps of a set of functions that depend on the Hamiltonian
    #
    # These can be user defined functions, or predefined ones found in the global scope of this module
    #
    def paramSweepFunc(self, sweep_spec, ufcn, ufcn_args={}, get_vectors=False, sparse=False, sparselevels=6, sparsesolveropts={"sigma":0,"mode":"normal","maxiter":None,"tol":0}):
        self.params.ndSweep(sweep_spec)
        #self.eval_dp = True # Signal we want to calculate derived parameters
        
        # FIXME: Need a way to check if the Hamiltonian will need to be updated, this complicates things.
        
        self._presub()
        result = []
        func_result = []
        params = None
        E = None
        V = None
        if not get_vectors:
            if not sparse:
                for i in range(self.params.sweep_grid_npts):
                    params = dict([(k,v[i]) for k,v in self.params.sweep_grid_c.items()])
                    self._postsub(params)
                    E = self.getHamiltonian().eigenenergies()
                    result.append(E)
                    func_result.append(ufcn(self,params,E,**ufcn_args))
            else:
                H = None
                for i in range(self.params.sweep_grid_npts):
                    params = dict([(k,v[i]) for k,v in self.params.sweep_grid_c.items()])
                    self._postsub(params)
                    
                    # Solve
                    E = sc.sparse.linalg.eigsh(self.getHamiltonian().data,k=sparselevels,return_eigenvectors=False,**sparsesolveropts)
                    
                    # Sort the results
                    E.sort()
                    result.append(E)
                    func_result.append(ufcn(self,params,E,**ufcn_args))
        else:
            if not sparse:
                for i in range(self.params.sweep_grid_npts):
                    params = dict([(k,v[i]) for k,v in self.params.sweep_grid_c.items()])
                    self._postsub(params)
                    E,V = self.getHamiltonian().eigenstates()
                    result.append([E,V])
                    func_result.append(ufcn(self,params,E,V,**ufcn_args))
            else:
                H = None
                for i in range(self.params.sweep_grid_npts):
                    params = dict([(k,v[i]) for k,v in self.params.sweep_grid_c.items()])
                    self._postsub(params)
                    
                    # Solve
                    E,V = sc.sparse.linalg.eigsh(self.getHamiltonian().data,k=sparselevels,return_eigenvectors=True,**sparsesolveropts)
                    
                    # FIXME: Sort the results and convert eigenvectors to qutip 
                    result.append((E,V.T))
                    func_result.append(ufcn(self,params,E,V.T,**ufcn_args))
                    
        return np.array(result),np.array(func_result)
    
    ##
    #
    #
    #
    def getSweep(self, data, ind_var, static_vars, evaluable="getHamiltonian"):
            return self.params.getSweepResult(ind_var, static_vars, data=data, key=evaluable)
    
    ###################################################################################################################
    #       Internal Functions
    ###################################################################################################################
    
    ## Get list of swept and non-swept parameters
    #
    #
    #
    def _separate_sweep_params(self):
        swept = []
        non_swept = []
        
        # Check which parameters are not being swept
        for k,v in self.all_params_sym.items():
            if k not in self.params.sweep_grid_params:
                non_swept.append(k)
            else:
                swept.append(k)
        return swept, non_swept
    
    ## Set non-swept parameters and return substitutions dict
    #
    #
    #
    def _set_non_swept_params(self):
        subs = {}
        self.subsystem_params_nonswept = []
        for k,v in self.all_params_sym.items():
            if k not in self.params.sweep_grid_params:
                subs[v] = self.params.getParameterValue(k)
        return subs
    
    ## Set swept parameters and return substitutions dict
    #
    #
    #
    def _set_swept_params(self, params):
        subs = {}
        for k,v in params.items():
            subs[self.all_params_sym[k]] = v
            self.params.setParameterValue(k,v)
        return subs
    
    ## Check which subsystems will require recalculations during a sweep
    #
    #
    #
    def _check_sys_sweep_params(self):
        # FIXME: This won't work, need to check if symbols related to a subsystem are used in another, for each substitutable equation
        ret = {}
        for subsys in self.params_groups.keys():
            ret[subsys] = False
            for p in self.params_groups[subsys]:
                if p in self.params.sweep_grid_params:
                    ret[subsys] = True # System requires recalculation
        return ret
    
    ## Find which operators will need to be regenerated for each sweep
    #
    #
    #
    def _get_regen_ops_list(self):
        self.Rlist = []
        for k,v in self.Z_pre.items():
            if v is None:
                continue
            try:
                r = float(v)
            except TypeError:
                self.Rlist.append(k)
    
    ## Error checking for keywords
    #
    #
    #
    def _check_args(self):
        pass



class HierarchicalHamilSpec(HamilSpec):
    """
    """
    
    ## Available basis representations
    #
    #
    #
    __basis_repr = [
        "charge",
        "flux",
        "osc",
        "custom"
    ]
    
    ## Mode types
    #
    #
    #
    __mode_types = [
        "osc", # DoFs have capacitive and inductive parts only
        "jos", # DoFs have capacitive, inductive and Josephson parts
        "isl", # DoFs have capacitive parts only
        "leg"  # DoFs have inductive parts only
    ]
    
    ## Consider listing dependencies here too
    #
    #
    #
    __eval_avail = [
        "getHamiltonian",
        "getBranchCurrents",
        "getNodeVoltages",
        "getChargingEnergies",
        "getJosephsonEnergies",
        "getResonatorResponse"
    ]
    
    ## Original circuit parameter group name
    __orig_group_name = "orig"
    
    ## Initialise a Hamiltonian using a circuit specification
    #
    #
    #
    def __init__(self, circuitspec, unit=units.Units("CQED1"), mode="node", include_subsystem=None):
        
        # Initialise the temporary data manager
        super(HamilSpec, HamilSpec).__init__(self)
        self.newSession(id(self))
        self.__use_temp = False
        
        # Assign the circuit
        self.cs = circuitspec
        
        # Set the unit system
        self.units = unit
        
        # Check there are subcircuits defined
        if self.cs.subcircuits == {}:
            raise Exception("There are no subcircuits associated with this CircuitSpec instance.")
        
        # Parameter mappings to symbols
        self.orig_params_sym = self.cs.getParameterDict() # Current circuit parameters
        self.subsys_params_sym = {} # Subsystem circuit parameters
        self.all_params_sym = {} # All circuit parameters
        
        # Initialise individual HamilSpec objects for subcircuits
        self.subcircuit_hamil = {k: HamilSpec(v, unit=unit, mode=mode, include_subsystem=include_subsystem) for k, v in self.cs.subcircuits.items()}
        
        # Map which nodes belong to which subcircuit
        self.subcircuit_nodes = {}
        for k, v in self.subcircuit_hamil.items():
            nodes = v.getPosList()
            self.subcircuit_nodes.update({n: k for n in nodes})
        
        # Sub circuit truncations
        self.subcircuit_trunc = {}
        
        # Sub circuit Hilbert space expanders
        self.subcircuit_hse = {}
        
        # Load default diagonaliser configuration
        self.setDiagConfig()
        
        ###############
        
        # Nested dictionary for DoF operators, keyed by the relevant mode
        self.circ_operators = {}
        
        # Mode truncations and basis representation
        self.mode_truncations = {}
        
        # Included subsystems
        self.subsystem = include_subsystem
        
        # Define the parameter collection
        self.params = pa.ParamCollection(list(self.orig_params_sym.keys()))
        
        # Parameter groups for keeping track of what needs to be recalculated
        self.params_groups = {
            self.__orig_group_name:list(self.orig_params_sym.keys())
        }
        
        # FIXME: Add something that tracks sweeps performed on this object and how they were dumped to temporary storage?
        
        # Check if a subsystem is required
        if self.subsystem is not None:
            
            # Eventually will allow more than one coupled subsystem
            #for k in self.subsystem:
            #    circ = self.cs.coupled_subsys[k]['circuitspec']
            
            # Update the parameters coming from the new circuit
            #self.params_sym = rcircuit.getParameterDict()
            self.all_params_sym = self.cs.rcircuit.getParameterDict()
            
            # Get the subsystem parameters
            self.subsys_params_sym = {k:v for k,v in self.all_params_sym.items() if k not in self.orig_params_sym.keys()}
            
            # Update the parameter collection
            self.params.addParameters(*list(self.all_params_sym.keys()))
            
            # Add new parameter groups
            self.params_groups[self.subsystem] = list(self.subsys_params_sym.keys())
            
            # Create an entry for the derived parameters
            self.subsys_dparams = {}
            self.subsys_dparams[self.subsystem] = {k:[] for k in self.cs.coupled_subsys[self.subsystem]['derived_parameters'].keys()}
            
        else:
            # All the parameters are the original system ones
            self.all_params_sym = self.orig_params_sym.copy()
        
        # For parameterisations
        self.params_subs = {}
        
        # Save the mode type
        self.Hmode = mode
        
        # For getting diagonal matrix elements
        self.getter_vecs = {}
        p = self.getPosList()
        for i,pos in enumerate(p):
            self.getter_vecs[pos] = sy.Matrix(qt.basis(len(p),i).data.todense())
        
        # Oscillator characteristics
        self.osc_params = {}
        
    def getPosList(self):
        poslist = []
        for name, hamil in self.subcircuit_hamil.items():
            poslist.extend(hamil.getPosList())
        return sorted(poslist)
    
    def getSubCircuit(self, name):
        """ Returns the :class:`HamilSpec` object associated with `name`.
        
        :param name: The name of the subcircuit sought.
        :type name: str
        
        :return: The instance of :class:`HamilSpec` associated with `name`
        :rtype: :class:`HamilSpec`
        """
        return self.subcircuit_hamil[name]
    
    def getSubCircuitNames(self):
        """ Returns the list of subcircuit names associated with the underlying :class:`CircuitSpec` instance.
        
        :return: A list of subcircuit names
        :rtype: list
        """
        return list(self.subcircuit_hamil.keys())
    
    def setSubCircuitTruncation(self, name, trunc):
        if name not in self.getSubCircuitNames():
            raise Exception("name '%s' does not exist" % name)
        self.subcircuit_trunc[name] = trunc
        self.subcircuit_hse[name] = qt.qeye(trunc)
        # FIXME: Check that trunc does not exceed the Hilbert space size of the circuit.
    
    ###################################################################################################################
    #       Hamiltonian Building Functions
    ###################################################################################################################
    
    def getSymbolicOscillatorParams(self):
    
        # Check truncations were defined
        if len(self.mode_truncations) != len(self.getPosList()):
            raise Exception("mode types and truncations not yet all set. Use setTruncation() for each degree of freedom first.")
        
        # Check oscillator basis is being used anywhere
        #if "osc" not in [self.mode_truncations[k][1] for k in self.getPosList()]:
        #    return
        
        # Check inv inductance matrix is diagonal
        #if not self.cs.getInverseInductanceMatrix(mode=self.Hmode).is_diagonal():
        #    raise Exception("inverse inductance matrix is not diagonal. Perform a transformation first.")
        Linv = self.cs.getInverseInductanceMatrix(mode=self.Hmode)
        Cinv = self.cs.getInverseCapacitanceMatrix(mode=self.Hmode)
        
        # Retrieve oscillator frequency and impedance symbolic expressions
        for pos in self.getPosList():
            Li = self.getter_vecs[pos].transpose()*Linv*self.getter_vecs[pos]
            Ci = self.getter_vecs[pos].transpose()*Cinv*self.getter_vecs[pos]
            if Li[0] == 0 or Ci[0] == 0:
                self.osc_params[pos] = None
            else:
                self.osc_params[pos] = {"frequency":sy.sqrt(Li[0]*Ci[0]),"impedance":sy.sqrt(Ci[0]/Li[0])}
    
    def prepareOperators(self):
        
        # Check for parameterisations
        subs = dict([(self.all_params_sym[k],v) for k,v in self.params_subs.items()])
        for k,v in self.all_params_sym.items():
            subs[v] = self.params.getParameterValue(k)
        
        # Prepare the operators of each subcircuit
        for k in self.subcircuit_hamil.keys():
            self.subcircuit_hamil[k].prepareOperators()
        
        # Update only the inverse Capacitance and Inductance matrices here
        # for building interaction term.
        self.Cinvnp = np.asmatrix(self.cs.getInverseCapacitanceMatrix(mode=self.Hmode).subs(subs), dtype=np.float64)
        self.Linvnp = np.asmatrix(self.cs.getInverseInductanceMatrix(mode=self.Hmode).subs(subs), dtype=np.float64)
    
    ## Substitute parameters before a sweep to minimise total number of substitutions
    #
    # Called after an ndSweep has been created
    #
    def _presub(self):
        
        print("_presub():")
        print(self.subcircuit_hamil['qb1'].circ_operators[1]['charge'])
        
        # Check for parameterisations
        p_subs = dict([(self.all_params_sym[k],v) for k,v in self.params_subs.items()])
        
        # Set the parameters that are not being swept
        subs = self._set_non_swept_params()
        
        self.Cinv_pre = self.cs.getInverseCapacitanceMatrix(mode=self.Hmode).subs(p_subs).subs(subs)
        self.Linv_pre = self.cs.getInverseInductanceMatrix(mode=self.Hmode).subs(p_subs).subs(subs)
        
        # Get branch inverse inductance matrix for branch current calculations
        if self.Hmode == 'node':
            self.Linv_b_pre = self.cs.getInverseInductanceMatrix(mode='branch').subs(p_subs).subs(subs)
        
        # Update all subsystem substitutions
        for name in self.subcircuit_hamil.keys():
            self.subcircuit_hamil[name].params = self.params
            self.subcircuit_hamil[name]._presub()
    
    ## Substitute parameters during a sweep
    #
    #
    #
    def _postsub(self, params):
        
        # Set the parameter values
        subs = self._set_swept_params(params)
        
        # Substitute circuit parameters
        self.Cinvnp = np.asmatrix(self.Cinv_pre.subs(subs),dtype=np.float64)
        self.Linvnp = np.asmatrix(self.Linv_pre.subs(subs),dtype=np.float64)
        
        # Get branch inverse inductance matrix for branch current calculations
        if self.Hmode == 'node':
            self.Linvnp_b = np.asmatrix(self.Linv_b_pre.subs(subs),dtype=np.float64)
        
        # Update all subsystem substitutions
        for name in self.subcircuit_hamil.keys():
            self.subcircuit_hamil[name]._postsub(params)
        
    ###################################################################################################################
    #       Evaluables
    ###################################################################################################################
    
    ## Return the numerical Hamiltonian given a parameter collection
    def getHamiltonian(self):
        
        # Unperturbed Hamiltonian
        H = None
        E = None
        V = None
        P = []
        keys = list(self.subcircuit_hamil.keys())
        sc_pos = {keys[i]:i for i in range(len(keys))}
        H0_list = []
        exp_list = []
        for i, k in enumerate(keys):
            
            Hobj = self.subcircuit_hamil[k]
            trunc = self.subcircuit_trunc[k]
            
            # Solve sub circuit Hamiltonian
            Hobj.substitute({})
            H = Hobj.getHamiltonian()
            old_config = Hobj.getDiagConfig()
            Hobj.setDiagConfig(get_vectors=True, eigvalues=trunc)
            E,V = Hobj.diagonalize(H)
            Hobj.diagonalizer_config = old_config # Restore old configuration
            
            # Create truncated sub circuit Hamiltonian
            exp_list = [self.subcircuit_hse[k] for k in keys]
            exp_list[i] = qt.Qobj(np.diag(E))
            H0_list.append(qt.tensor(*exp_list))
            
            # Get the modified subspace projector for this sub circuit
            P.append(sum([qt.basis(trunc,j)*V[j].dag() for j in range(trunc)]))
        
        Hi_list = []
        exp_list = []
        exp_list2 = []
        # Interaction term
        for e in self.cs.subcircuit_coupling_edges:
            
            # Check the value of the inverse capacitance matrix
            Cij = self.Cinvnp[e[0]-1, e[1]-1]
            if Cij != 0.0: # FIXME: Use util.san here?
                
                # Charge interaction terms, the projection onto LE subspace is done here
                exp_list = [self.subcircuit_hse[k] for k in keys]
                exp_list[sc_pos[self.subcircuit_nodes[e[0]]]] = \
                P[sc_pos[self.subcircuit_nodes[e[0]]]] * \
                self.subcircuit_hamil[self.subcircuit_nodes[e[0]]].circ_operators[e[0]]['charge'] * \
                P[sc_pos[self.subcircuit_nodes[e[0]]]].dag()
                
                exp_list2 = [self.subcircuit_hse[k] for k in keys]
                exp_list2[sc_pos[self.subcircuit_nodes[e[1]]]] = \
                P[sc_pos[self.subcircuit_nodes[e[1]]]] * \
                self.subcircuit_hamil[self.subcircuit_nodes[e[1]]].circ_operators[e[1]]['charge'] * \
                P[sc_pos[self.subcircuit_nodes[e[1]]]].dag()
                
                
                Hi_list.append(qt.tensor(*exp_list) * qt.tensor(*exp_list2) * Cij * self.units.getPrefactor("Ec"))
            
            # Check the value of the inverse inductance matrix
            Lij = self.Linvnp[e[0]-1, e[1]-1]
            if Lij != 0.0: # FIXME: Use util.san here?
            
                # Flux interaction terms
                exp_list = [self.subcircuit_hse[k] for k in keys]
                exp_list[sc_pos[self.subcircuit_nodes[e[0]]]] = \
                P[sc_pos[self.subcircuit_nodes[e[0]]]] * \
                self.subcircuit_hamil[self.subcircuit_nodes[e[0]]].circ_operators[e[0]]['flux'] * \
                P[sc_pos[self.subcircuit_nodes[e[0]]]].dag()
                
                exp_list2 = [self.subcircuit_hse[k] for k in keys]
                exp_list2[sc_pos[self.subcircuit_nodes[e[1]]]] = \
                P[sc_pos[self.subcircuit_nodes[e[1]]]] * \
                self.subcircuit_hamil[self.subcircuit_nodes[e[1]]].circ_operators[e[1]]['flux'] * \
                P[sc_pos[self.subcircuit_nodes[e[1]]]].dag()
                
                Hi_list.append(qt.tensor(*exp_list) * qt.tensor(*exp_list2) * Lij * self.units.getPrefactor("El"))
        
        # Total Hamiltonian
        return sum(H0_list) + sum(Hi_list)
    
    ###################################################################################################################
    #       Parameter Collection Wrapper Functions and Extended Functions
    ###################################################################################################################
    
    ## Create a sweep specification for a single parameter.
    def sweepSpec(self, *args, **kwargs):
        return self.params.paramSweepSpec(*args,**kwargs)
    
    ## Create an evaluation specification for a single function.
    def evalSpec(self, func, diag, depends, **kwargs):
        return {'eval':func, 'diag':diag, 'depends':depends, 'kwargs':kwargs}
    
    ## Set the value of a parameter.
    def setParameterValue(self, name, value):
        ret = self.params.setParameterValue(name,value)
        self._update_subcircuit_params()
        return ret
    
    ## Get the value of a parameter.
    def getParameterValue(self, name):
        return self.params.getParameterValue(name)
    
    ## Set many parameter values.
    def setParameterValues(self, *name_value_pairs):
        ret = self.params.setParameterValues(*name_value_pairs)
        self._update_subcircuit_params()
        return ret
    
    ## Get many parameter values.
    def getParameterValues(self, *names):
        return self.params.getParameterValues(*names, dictionary=True)
    
    ## Gets all parameters
    def getParameterValuesDict(self):
        return self.params.getParameterValuesDict()
    
    def getParameterSweep(self, name):
        return self.params.getParameterSweep(name)
    
    def getParameterNames(self):
        return self.params.getParameterNamesList()
    
    def getPrefactor(self, name):
        return self.units.getPrefactor(name)
    
    def getSubsystemList(self):
        return list(self.cs.coupled_subsys.keys())
    
    def getSubsystemDerivedParametersList(self, subsystem):
        return list(self.cs.coupled_subsys[subsystem]['derived_parameters'].keys())
    
    def getSubsystemDerivedParameter(self, param):
        return self.subsys_dparams[self.subsystem][param]
    
    ###################################################################################################################
    #       Internal Functions
    ###################################################################################################################
    
    def _update_subcircuit_params(self):
        params_dict = self.getParameterValuesDict()
        for k in self.subcircuit_hamil.keys():
            self.subcircuit_hamil[k].setParameterValues(params_dict)
            
    
