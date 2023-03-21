""" The :py:mod:`pycqed.src.parameters` module defines two classes :class:`Param` and :class:`ParamCollection` that are used to manipulate scalar parameters used in simulations and experiments.
"""
import os
import numpy as np
import sympy as sy
import networkx as nx
import graphviz as gv
import pydot as pd
import platform

# FIXME: Make this more intelligent
if platform.system() == 'Windows':
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

from . import text2latex as t2l
from . import util

class Param:
    """This class defines the properties of a parameter used in simulations and experiments. It supports upper and lower bounds and the generation of sweeps for scalars.
    
    :param name: The utf name of the parameter.
    :type name: str
    
    :param value: The initial value to set the parameter to, defaults to `None`.
    :type value: int float np.float64, optional
    
    :param bounds: The lower and upper bounds of the parameter, defaults to `[-np.inf, np.inf]`.
    :type bounds: list of floats, optional
    
    :param unit_pref: A prefactor that encode the unit of the parameter, defaults to `1.0`.
    :type unit_pref: int float np.float64, optional
    
    :raises Exception: If the parameters are not accepted types or the initial value is out of bounds.
    
    :return: A new instance of :class:`Param`.
    :rtype: :class:`Param`
    
    This constructor attempts to generate a latex version of the parameter name for use with plots. It uses the :py:mod:`pycqed.src.text2latex` module for this functionality. The constructor also creates a `sympy` symbol associated with the parameter. The first character is the symbol and the following characters are subscripted.
    
    The following class attributes are accessible by the user:
    
    :ivar name: The utf string of the parameter.
    :ivar name_latex: A string to be used with latex.
    :ivar symbol: A `sympy` symbol for the parameter.
    :ivar sweep: The sweep array.
    """
    __types = ["type1", "type2"]
    __valid_scalar_types = [int, float, np.float64]
    
    def __init__(self, name, value=None, bounds=[-np.inf, np.inf], unit_pref=1.0):
        """Constructor method."""
        # Ensure name is a string and it has the correct format
        if type(name) is not str:
            raise Exception("'name' is not a string.")
        if name.find(' ') >= 0:
            raise Exception("'name' should not have any whitespace characters.")
        # Ensure value is a float if it is not None
        if value is not None:
            if type(value) not in self.__valid_scalar_types:
                raise Exception("'value' is not a float.")
        # Ensure bounds is a list of two floats
        if type(bounds) is not list:
            raise Exception("'bounds' is not a list.")
        else:
            if len(bounds) != 2:
                raise Exception("'bounds' should only have two values, %i found." % len(bounds))
            if (type(bounds[0]) is not float or type(bounds[1]) is not float) and (type(bounds[0]) is not int or type(bounds[1]) is not int):
                raise Exception("'bounds' should contain floats.")
        # Ensure lower bound is smaller than upper bound
        if bounds[0] > bounds[1]:
            raise Exception("Lower bound is greater than upper bound in 'bounds'.")
        # Ensure unit pref is a float and is positive
        if type(unit_pref) not in self.__valid_scalar_types:
            raise Exception("'unit_pref' is not a float.")
        if float(unit_pref) < 0.0:
            raise Exception("'unit_pref' is negative.")
        
        
        self.name = name
        self.symbol = sy.symbols("%s_{%s}" % (name[0], name[1:]))
        if value is not None:
            self.__value = float(value)
        else:
            self.__value = None
        self.__lower_bound = float(bounds[0])
        self.__upper_bound = float(bounds[1])
        self.__upref = float(unit_pref)
        self.name_latex = t2l.latexify_param_name(self.name)
        self.sweep = np.array([])
    
    def getValue(self):
        """ Get the current value of the parameter.
        
        :return: The current value of the parameter.
        :rtype: int float np.float64
        """
        return self.__value
    
    def setValue(self, value):
        """ Set the value of the parameter.
        
        :param value: The value to set the parameter to.
        :type value: int float np.float64
        
        :raises Exception: If the value is not an accepted type, or it is out of bounds.
        
        :return: None
        """
        # Ensure value is a float
        if type(value) not in self.__valid_scalar_types:
            raise Exception("'value' is not a float.")
        
        # Check bounds
        if float(value) > self.__upper_bound:
            raise Exception("Param %s 'value' exceeds specified upper bound." % (self.name))
        if float(value) < self.__lower_bound:
            raise Exception("Param %s 'value' exceeds specified lower bound." % (self.name))
        self.__value = float(value)
    
    def getBounds(self):
        """ Get the bounds of the parameter.
        
        :return: The lower and upper bounds of the parameter.
        :rtype: list of two floats
        """
        return [self._lower_bound, self.__upper_bound]
    
    def setBounds(self, bounds):
        """ Set the bounds of the parameter.
        
        :param bounds: The lower and upper bounds of the parameter.
        :type bounds: list of two floats, or np.inf.
        
        :raises Exception: If the bounds are not in the correct format, or the lower bound is greater than the upper bound.
        
        :return: None.
        """
        # Ensure bounds is a list of two floats
        if type(bounds) is not list:
            raise Exception("'bounds' is not a list.")
        else:
            if len(bounds) != 2:
                raise Exception("'bounds' should only have two values, %i found." % len(bounds))
            if (type(bounds[0]) is not float or type(bounds[1]) is not float) and (type(bounds[0]) is not int or type(bounds[1]) is not int):
                raise Exception("'bounds' should contain floats.")
        # Ensure lower bound is smaller than upper bound
        if bounds[0] > bounds[1]:
            raise Exception("Lower bound is greater than upper bound in 'bounds'.")
        
        self.__lower_bound = float(bounds[0])
        self.__upper_bound = float(bounds[1])
    
    def linearSweep(self, start, end, N):
        """ Generates a linear sweep using `numpy.linspace` with added bounds checking. The sweep is saved internally, and is overwritten by subsequent calls to this function.
        
        :param start: The initial value of the sweep.
        :type start: float
        
        :param end: The last value of the sweep.
        :type end: float
        
        :param N: The number of points from start to end.
        :type N: int
        
        :raises Exception: If the argument types are incorrect, or if start and end are out of bounds.
        
        :return: The parameter sweep array.
        :rtype: numpy.ndarray
        """
        # Ensure start is a float
        if type(start) is not float and type(start) is not int:
            raise Exception("'start' is not a float.")
        # Ensure end is a float
        if type(end) is not float and type(end) is not int:
            raise Exception("'end' is not a float.")
        # Ensure N is an int
        if type(N) is not int:
            raise Exception("'N' is not an int.")
        
        # Check bounds
        if float(start) > self.__upper_bound:
            raise Exception("Param %s 'start' exceeds specified upper bound." % (self.name))
        if float(start) < self.__lower_bound:
            raise Exception("Param %s 'start' exceeds specified lower bound." % (self.name))
        if float(end) > self.__upper_bound:
            raise Exception("Param %s 'end' exceeds specified upper bound." % (self.name))
        if float(end) < self.__lower_bound:
            raise Exception("Param %s 'end' exceeds specified lower bound." % (self.name))
        
        self.sweep = np.linspace(float(start), float(end), N)
        self.N = N
        return self.sweep

# FIXME: This class should technically inherit the unit system
class ParamCollection:
    """ This class uses an array of :class:`Param` instances and provides methods to manipulate them in useful ways, for example to create multidimensional sweeps and return substitution dictionaries. It also provides an equation system, to allow parameters to be created in terms of others, or to specify inter-dependencies.
    
    :param names: A list of parameter names to create.
    :type names: list of str
    
    :raises Exception: If the names are not strings (raised by the underlying :class:`Param` constructor).
    
    :return: A new instance of :class:`ParamCollection`
    :rtype: :class:`ParamCollection`
    
    The following class attributes are accessible by the user. These attributes are created or overwritten by calls to :func:`ndSweep`, except the last, which is created or overwritten by calls to :func:`computeFuncSweep` and :func:`computeExprSweep`.
    
    :ivar sweep_spec: An array of sweep specifications.
    :ivar sweep_grid_npts: The number of points in the current parameters sweep.
    :ivar sweep_grid_ndims: The number of dimensions or parameters being swept.
    :ivar sweep_grid_params: The list of parameters being swept.
    :ivar sweep_grid_c: Dictionary of collapsed sweeps keyed by parameter name.
    :ivar sweep_grid_nc: Dictionary of non-collapsed sweeps keyed by parameter name.
    :ivar sweep_grid_result: One-dimensional array of values (or objects) that result from computing a sweep with the collapsed grid.
    
    """
    
    def __init__(self, names):
        self.__collection = {}
        self.__symbol_map = {}
        self.__parameterisation = {}
        self.__parameterisation_graph = nx.DiGraph()
        for name in names:
            self.__collection[name] = Param(name)
            self.__symbol_map[name] = self.__collection[name].symbol
    
    ###################################################################################################################
    #       Basic Parameter Manipulation Functions
    ###################################################################################################################
    
    # FIXME: Should be renamed to getParameterDict
    def getParameterList(self):
        """ Gets the parameter dictionary, mapping the utf name to the :class:`Param` instance.
        
        :return: A dictionary of param names to :class:`Param` instances.
        :rtype: dict
        """
        return self.__collection
    
    # FIXME: Should be renamed to getSymbolDict
    def getSymbolList(self):
        """ Gets the symbol dictionary, mapping the utf name to the Param `sympy` symbol.
        
        :return: A dictionary of param names mapping to `sympy` symbols.
        :rtype: dict
        """
        return self.__symbol_map
    
    def getSymbol(self, name):
        """ Gets the symbol associated with the parameter `name`.
        
        :param name: The name of the parameter.
        :type name: str
        
        :raises Exception: If the parameter is not in the collection.
        
        :return: The symbol of the specified parameter.
        :rtype: sympy.Symbol
        """
        if type(name) != str:
            raise Exception("Parameter name %s is not a string." % repr(name))
        return self.__symbol_map[name]
    
    def getParameterValuesDict(self):
        """ Returns a dictionary of all the current set values of the parameters in a dictionary format.
        
        :return: A dictionary of all param names to values.
        :rtype: dict
        """
        return {k: v.getValue() for k, v in self.__collection.items()}
    
    def getSymbolValuesDict(self):
        """ Returns a dictionary of all the current set values of the parameter symbols in a dictionary format.
        
        :return: A dictionary of all param symbols to values.
        :rtype: dict
        """
        return {self.__symbol_map[k]: v.getValue() for k, v in self.__collection.items()}
    
    def addParameter(self, name, symbol_override=None):
        """ Adds a new parameter to the collection if it does not already exist. If it does exist, nothing is reported.
        
        :param name: The name of the parameter to add.
        :type name: str
        
        :param symbol_override: An different symbol to use than the internally generated one.
        :type symbol_override: sympy.Symbol
        
        :raises Exception: If `name` is not a str or `symbol_override` is not a sympy.Symbol.
        
        :return: None
        """
        if type(name) != str:
            raise Exception("Parameter name %s is not a string." % repr(name))
        
        if symbol_override is not None:
            if type(symbol_override) != sy.Symbol:
                raise Exception("Symbol %s is not a sympy.Symbol instance." % repr(symbol_override))
        
        if name not in list(self.__collection.keys()):
            self.__collection[name] = Param(name)
            if symbol_override is not None:
                self.__collection[name].symbol = symbol_override
            self.__symbol_map[name] = self.__collection[name].symbol
    
    def addParameters(self, *names):
        """ Adds multiple new parameters to the collection if they do not already exist. If some or all exist, nothing is reported.
        
        :param \*name: Arguments list of parameter names to add.
        :type \*name: str, str ...
        
        :return: None
        """
        for name in list(names):
            self.addParameter(name)
    
    def rmParameter(self, name):
        """ Removes a parameter from the collection if it exists.
        
        :param name: The parameter to remove.
        :type name: str
        
        :raises Exception: If the parameter does not exist in the collection.
        
        :return: None
        """
        if name not in list(self.__collection.keys()):
            raise Exception("'%s' parameter was not found." % name)
        else:
            del self.__collection[name]
            del self.__symbol_map[name]
    
    def getParameterNamesList(self):
        """ Gets the list of available parameter names in the collection.
        
        :return: The list of parameter names.
        :rtype: list
        """
        return list(self.__collection.keys())
    
    def getParameterSymbolsList(self):
        """ Gets the list of available parameter symbols in the collection.
        
        :return: The list of `sympy` symbols.
        :rtype: list
        """
        return list(self.__symbol_map.values())
    
    def getParameterFromSymbol(self, symbol):
        """ Gets parameter string name associated with the provided `sympy.Symbol`.
        
        :raises Exception: If the symbol does not exist in the collection.
        
        :return: The parameter name.
        :rtype: str
        """
        if symbol not in self.__symbol_map.values():
            raise Exception("Symbol '%s' was not found." % repr(symbol))
        for k, v in self.__symbol_map.items():
            if symbol == v:
                return k
    
    def setParameterValue(self, name, value):
        """ Set the value of a given parameter.
        
        :param name: The name of the parameter to set.
        :type name: str
        
        :param value: The value to set the parameter to.
        :type value: float
        
        :raises Exception: If the parameter is not in the collection.
        
        :return: None
        """
        # Check name is defined
        if name not in self.__collection.keys():
            raise Exception("'%s' parameter was not found." % name)
        
        # Don't update the value if this parameter is parameterised by others
        if name in self.__parameterisation.keys():
            print("Warning: Parameter %s is parameterised so it will not be set to the requested value." % name)
            return
        
        # Update the value
        self.__collection[name].setValue(value)
        self._update_parameterisations()
    
    def getParameterValue(self, name):
        """ Get the value of a given parameter.
        
        :param name: The name of the parameter.
        :type name: str
        
        :raises Exception: If the parameter is not in the collection.
        
        :return: The current value of the specified parameter.
        :rtype: float
        """
        # Check name is defined
        if name not in list(self.__collection.keys()):
            raise Exception("'%s' parameter was not found." % name)
        
        return self.__collection[name].getValue()
    
    def getParameterSweep(self, name):
        """ Get the parameter sweep associated with a parameter.
        
        :param name: The name of the parameter.
        :type name: str
        
        :raises Exception: If the parameter is not in the collection.
        
        :return: The sweep array.
        :rtype: numpy.ndarray
        """
        # Check name is defined
        if name not in list(self.__collection.keys()):
            raise Exception("'%s' parameter was not found." % name)
        return self.__collection[name].sweep
    
    def getParameterLatexName(self, name):
        """ Get the latex name of the parameter.
        
        :param name: The name of the parameter.
        :type name: str
        
        :raises Exception: If the parameter is not in the collection.
        
        :return: The parameter latex name.
        :rtype: str
        """
        # Check name is defined
        if name not in list(self.__collection.keys()):
            raise Exception("'%s' parameter was not found." % name)
        return self.__collection[name].name_latex
    
    def setParameterValues(self, *name_value_pairs):
        """ Set many parameter values.
        
        :param \*name_value_pairs: Arguments list, formatted as the parameter name followed by its value, or optionally passed as a dictionary.
        :type \*name_value_pairs: str, float, str, float ..., or a dict.
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds.
        
        :return: None
        """
        if len(list(name_value_pairs)) == 1:
            if type(name_value_pairs[0]) is not dict:
                raise Exception("Argument should be a dictionary, not '%s'." % repr(type(name_value_pairs[0])))
            #for k,v in name_value_pairs[0].items():
            #    self.__collection[k].setValue(v)
            #return
        
        # Separate names from values
        keys = None
        values = None
        if type(name_value_pairs[0]) == dict:
            keys = list(name_value_pairs[0].keys())
            values = list(name_value_pairs[0].values())
        else:
            keys = list(name_value_pairs)[::2]
            values = list(name_value_pairs)[1::2]
        
        # Check there are as many parameters as values
        if len(keys) != len(values):
            raise Exception("'name_value_pairs' definition invalid.")
        
        for i, name in enumerate(keys):
            # Check name is defined
            if name not in list(self.__collection.keys()):
                raise Exception("'%s' parameter was not found." % name)
            if name in self.__parameterisation.keys():
                print("Warning: Parameter %s is parameterised so it will not be set to the requested value." % name)
                continue
            self.__collection[name].setValue(values[i])
        self._update_parameterisations()
    
    def getParameterValues(self, *names):
        """ Get the values of many parameters as a dictionary.
        
        :param \*names: Arguments list, formatted as the parameter names.
        :type \*names: str, str ...
        
        :raises Exception: If the argument types are incorrect, ill-formatted or not found.
        
        :return: A dictionary with the parameter names as keys
        :rtype: dict
        """
        values = {}
        for name in list(names):
            # Check name is defined
            if name not in list(self.__collection.keys()):
                raise Exception("'%s' parameter was not found." % name)
            values[name] = self.__collection[name].getValue()
        return values
    
    def getSymbolValues(self, *names):
        """ Get the values of many parameters keyed by symbol. Useful for getting a substitution dict of a selection of parameters.
        
        :param \*names: Arguments list, formatted as the parameter names.
        :type \*names: str, str ...
        
        :raises Exception: If the argument types are incorrect, ill-formatted or not found.
        
        :return: A dictionary with the parameter symbols as keys
        :rtype: dict
        """
        values = {}
        for name in list(names):
            # Check name is defined
            if name not in list(self.__collection.keys()):
                raise Exception("'%s' parameter was not found." % name)
            values[self.__symbol_map[name]] = self.__collection[name].getValue()
        return values
    
    def allParametersSet(self):
        """ Checks if all the parameters in the collection have been initialised.
        
        :return: True if all parameters have been initialised else False
        :rtype: bool
        """
        if None in self.getParameterValuesDict().values():
            return False
        return True
    
    ###################################################################################################################
    #       Parameterisations
    ###################################################################################################################
    
    def getSymbols(self, *names, symbol_overrides=None):
        """ Generates a set of `sympy` symbols for use in parameterisation. They are added as independent Param instances.
        The symbols are returned in a dictionary so that they can be used to create expressions.
        
        :param \*names: Arguments list, formatted as the parameter names.
        :type \*names: str, str ...
        
        :raises Exception: If the argument types are incorrect, ill-formatted or not found.
        
        :return: A dictionary of `sympy` symbols with the parameter names as keys
        :rtype: dict
        """
        if symbol_overrides is None:
            symbol_overrides = [None]*len(list(names))
        else:
            symbol_overrides = list(symbol_overrides)
        values = {}
        for i, name in enumerate(list(names)):
            # Check name is defined
            if type(name) != str:
                raise Exception("Parameter name %s is not a string." % repr(name))
            self.addParameter(name, symbol_override=symbol_overrides[i])
            values[name] = self.__symbol_map[name]
        return values
    
    def getParametricParametersList(self):
        """ Get the list of parameters that have a parametric expression.
        
        :return: A list of parameter names.
        :rtype: list
        """
        return list(self.__parameterisation.keys())
    
    def addParameterisation(self, name, expression):
        """ Registers a parameterisation of the parameter `name` in terms of symbols returned by :func:`getSymbols`. It is allowed to use `sympy` functions such as `sympy.cos` in expressions, and also any previously defined parameters. If the parameterisation already exists, it is overwritten.
        
        :param name: The name of the parameter that is being parameterised.
        :type name: str
        
        :param expression: A `sympy` expression in terms of other parameters.
        :type expression: float, int, variable
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds. Also raises an exception if this parameterisation would cause a cycle in the dependency tree of nested parameterisations.
        
        :return: None
        :rtype: None
        """
        if name not in list(self.__collection.keys()):
            raise Exception("'%s' parameter was not found." % name)
        
        # Check the symbols in expression are actually all registered and get their names
        names = []
        rev_map = {v: k for k, v in self.__symbol_map.items()}
        for sym in expression.free_symbols:
            if sym not in self.getSymbolList().values():
                raise Exception("Symbol '%s' not registered." % repr(sym))
            names.append(rev_map[sym])
        
        for pname in names:
            # Add an edge to the graph if this parameterisation depends on others
            if pname in self.__parameterisation.keys():
                self.__parameterisation_graph.add_edge(pname, name)
        
        # Check that there are no cycles induced by this parameterisation
        cycles = None
        try:
            cycles = nx.find_cycle(self.__parameterisation_graph, orientation="original")
        except nx.NetworkXNoCycle:
            pass
        if cycles is not None:
            # Remove the bad edges
            for pname in names:
                if pname in self.__parameterisation.keys():
                    self.__parameterisation_graph.remove_edge(pname, name)
            raise Exception("Cycle(s) found in parameterisation graph upon addition of name '%s': %s" % (name, repr(cycles)))
        
        # Register the parameterisation
        self.__parameterisation_graph.add_node(name)
        self.__parameterisation[name] = {
            "expression": expression,
            "parameters": names
        }
    
    def addParameterisationPrefactor(self, name, prefactor):
        """ Add a prefactor to a parameterisation expression. This is mechanism for implementing unit conversion between parameters if required.
        
        :param name: The name of the parametric parameter.
        :type name: str
        
        :param prefactor: A `sympy` expression in terms of other parameters.
        :type prefactor: float, int, variable
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds.
        
        :return: None
        :rtype: None
        """
        if name not in list(self.__parameterisation.keys()):
            raise Exception("'%s' parameter is not parameterised." % name)
        self.__parameterisation[name]["expression"] *= prefactor
    
    def getParametricExpression(self, name, expand=False):
        """ Gets the `sympy` expression of parameter `name`.
        
        :param name: The name of the parameter.
        :type name: str
        
        :param expand: Indicates whether to expand into nested parameterisations
        :type expand: bool, optional
        
        :raises Exception: If the parameter was not parameterised.
        
        :return: A `sympy` expression.
        :rtype: sympy type
        """
        if name not in list(self.__parameterisation.keys()):
            raise Exception("'%s' parameter is not parameterised." % name)
        if not expand:
            return self.__parameterisation[name]["expression"]
        
        def exprParametric(expr):
            pp = self.getParametricParametersList()
            for sym in list(expr.free_symbols):
                p = self.getParameterFromSymbol(sym)
                if p in pp:
                    return True
            return False
        
        # Recursively substitute expressions
        base = self.__parameterisation[name]["expression"]
        expr = None
        while exprParametric(base):
            expr = None
            for sym in base.free_symbols:
                try:
                    subs = {sym: self.getParametricExpression(self.getParameterFromSymbol(sym))}
                    expr = base.subs(subs)
                except: # Fail should only be caused when parameter is not parameterised
                    continue
            base = expr
        return base
    
    def getParameterisationParameters(self, name):
        """ Gets the parameters that form the parametric expression of parameter `name`.
        
        :param name: The name of the parameter.
        :type name: str
        
        :raises Exception: If the parameter was not parameterised.
        
        :return: A list of parameter names.
        :rtype: list
        """
        if name not in list(self.__parameterisation.keys()):
            raise Exception("'%s' parameter is not parameterised." % name)
        
        return self.__parameterisation[name]["parameters"]
    
    def rmParameterisation(self, name):
        """ Unregisters the parameterisation of parameter `name`.
        
        :param name: The name of the parameter that is being parameterised.
        :type name: str
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds.
        
        :return: None
        :rtype: None
        """
        if name not in list(self.__collection.keys()):
            raise Exception("'%s' parameter was not found." % name)
        
        if name not in list(self.__parameterisation.keys()):
            raise Exception("'%s' parameter is not parameterised." % name)
        
        
        # If we want to remove the associated parameters, we'll also need to check they can actually be removed without breaking everything
        #for sname in self.__parameterisation[name]["parameters"]:
        #    del self.__collection[sname]
        self.__parameterisation_graph.remove_node(name)
        del self.__parameterisation[name]
    
    def parameterisationParametersSet(self, name):
        """ Checks if the parameters that parameterise `name` have been initialised.
        
        :param name: The name of the parameterised parameter.
        :type name: str
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds.
        
        :return: True if the parameters are initialised else False
        :rtype: bool
        """
        if name not in list(self.__parameterisation.keys()):
            raise Exception("'%s' parameter was not found." % name)
        
        names = self.__parameterisation[name]['parameters']
        if None in self.getParameterValues(*names).values():
            return False
        return True
    
    def getParameterisationsInvolving(self, *names):
        """ Gets the list of parametric parameters that depend on the supplied parameter names. Returning an empty list if `name` is parametric parameter or doesn't exist in any parametric expressions.
        
        :param \*names: The names of the parameters.
        :type \*names: str
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds.
        
        :return: A list of parametric parameters. The list is empty if a parametric parameter is supplied as `name`.
        :rtype: list
        """
        pnames = []
        for name in list(names):
            if name not in list(self.__collection.keys()):
                raise Exception("'%s' parameter was not found." % name)
            
            # FIXME: Parameterisations can depend on others
            if name in list(self.__parameterisation.keys()):
                return []
            
            for pname in self.getParametricParametersList():
                if name in self.getParameterisationParameters(pname):
                    if pname not in pnames:
                        pnames.append(pname)
        return pnames
    
    def drawParameterisationGraph(self, filename=None):
        # Get the pydot graph
        pd_graph = nx.nx_pydot.to_pydot(self.__parameterisation_graph)
        
        # Compile the graphviz source
        gv_graph = gv.Source(pd_graph.create(format='dot').decode('utf8'))
        
        # Return the object, which should be rendered in a jupyter notebook
        if filename is None:
            return gv_graph
    
    ###################################################################################################################
    #       Parameter Sweeping Functions
    ###################################################################################################################
    
    def paramSweepSpec(self, name, *sweep_params):
        """ Convenience method to generate a sweep specification for use with :func:`ndSweep`.
        
        :param name: The name of the parameter to sweep.
        :type name: str
        
        :param \*sweep_params: The arguments of the sweep generating function.
        :type \*sweep_params: float, int, variable
        
        :raises Exception: If the argument types are incorrect, ill-formatted, not found, or out of bounds.
        
        :return: A sweep specification dictionary for use with :func:`ndSweep`.
        :rtype: dict
        """
        # Ensure name is a string
        if type(name) is not str:
            raise Exception("'name' is not a string.")
        
        # Check name is defined
        if name not in list(self.__collection.keys()):
            raise Exception("'%s' parameter was not found." % name)
        
        swp = list(sweep_params)
        return {
            "name": name,
            "start": swp[0],
            "end": swp[1],
            "N": swp[2]
        }
    
    def ndSweep(self, spec):
        """ Generates a single or multidimensional sweep of parameters in such a way that only a single for loop is required to apply the parameters.
        
        :param spec: An array of parameter sweep specifications, optionally created by :func:`paramSweepSpec`.
        :type spec: list of dict
        
        :raises Exception: If the requested sweeps exceed the bounds of a parameter.
        
        :return: None
        
        **Sweep Grid Explanation**
        
        Two version of the sweep grid are created, a *collapsed* and *non-collapsed* version. The former is convenient for use with a single for-loop, whereas the latter is more convenient for plotting data. The grids are saved as class attributes `sweep_grid_c` and `sweep_grid_nc` respectively, and their format is as follows
        
        .. code-block:: python
        
           grid = {
               param_name1: ndarray1,
               param_name2: ndarray2,
               ...
               param_nameN: ndarrayN
           }
        
        where `ndarrayN` is a one-dimensional array in the collapsed case, and a k dimensional array for k parameters in the non-collapsed case.
        
        **Order of the Sweeps**
        
        The last entry of the `spec` list corresponds to the inner-most nested loop, thus the results of that sweep are contiguous in the result of sweeping the collapsed list.
        
        **Retrieving Sweeps**
        
        If the collapsed sweep arrays are used in single loop, the results can be appended to a single one dimensional array. To retrieve the result of single sweep given constant values of the other parameters, use the :func:`getSweepResult` function.
        
        **Example**
        
        Here we create a three dimensional sweep and show the collapsed and non-collapsed sweep grids.
        
        .. code-block:: python
           
           # Create new instance
           p = ParamCollection(["p1","p2","p3"])
           
           # Generate a sweep specification
           spec = [
               p.paramSweepSpec("p1",-1.0,1.0,11),
               p.paramSweepSpec("p2",-10.0,10.0,3),
               p.paramSweepSpec("p3",-2.0,2.0,101)
           ]
           
           # Generate the sweep grid
           p.ndSweep(spec)
           
           # Show the collapsed grid
           print (p.sweep_grid_c)
           
           # Show the non collapsed grid
           print (p.sweep_grid_nc)
        
        """
        # Save the specification
        self.sweep_spec = spec
        
        # Parse specifications for sweeps
        sweeps = []
        keys = []
        self.sweep_grid_npts = 1
        for param_spec in spec:
            k = param_spec["name"]
            keys.append(k)
            sweeps.append(self.__collection[k].linearSweep(param_spec["start"], param_spec["end"], param_spec["N"]))
            self.sweep_grid_npts *= param_spec["N"]
        self.sweep_grid_params = keys
        self.sweep_grid_ndims = len(sweeps)
        
        # Generate mesh grid
        grid = np.meshgrid(*sweeps, indexing="ij")
        
        # Non-collapsed grid
        self.sweep_grid_nc = dict(zip(keys,grid))
        
        # Collapsed grid
        self.sweep_grid_c = {}
        for i, k in enumerate(keys):
            self.sweep_grid_c[k] = self.sweep_grid_nc[k].flatten()
            if i == 0:
                self.sweep_grid_c_len = len(self.sweep_grid_c[k])
    
    def getSweepParametersDict(self):
        """ Gets the list of parameters that will be swept, and substituted into the circuit equations during the evaluation loop.
        
        :return: A list of parameter names.
        :rtype: list
        """
        
        # Need to ensure that we substitute the parameterised parameter
        not_for_presub = set()
        for name in self.sweep_grid_params:
            not_for_presub.add(name)
            params = self.getParameterisationsInvolving(name)
            for param in params:
                successors = nx.dfs_successors(self.__parameterisation_graph, param)
                not_for_presub.add(param)
                for k, v in successors.items():
                    not_for_presub |= set(v)
        actual_sub_names = list(not_for_presub)
        return self.getSymbolValues(*actual_sub_names)
    
    def getNonSweepParametersDict(self):
        """ Gets the symbol-value dictionary of parameters that will NOT be swept, and substituted into the circuit equations before the evaluation loop.
        
        :return: A dictionary of symbol value pairs.
        :rtype: dict
        """
        
        # For each parameter we want to find it's parametric dependencies, and then exclude those.
        not_for_presub = set()
        for name in self.sweep_grid_params:
            not_for_presub.add(name)
            params = self.getParameterisationsInvolving(name)
            for param in params:
                successors = nx.dfs_successors(self.__parameterisation_graph, param)
                not_for_presub.add(param)
                for k, v in successors.items():
                    not_for_presub |= set(v)
        
        non_sweep = list(set(self.__collection.keys()) - not_for_presub)
        return self.getSymbolValues(*non_sweep)
    
    def collapsedIndices(self, *indices):
        """ Computes the indices of the collapsed array for corresponding indices of the non-collapsed array. Should not be used by the user. Will be hidden in the future.
        
        :param \*indices: Indices of the single parameter sweeps.
        :type \*indices: int, int ...
        
        :return: A single index to retrieve an entry from a results array.
        :rtype: int
        """
        # Get list of number of values in each dimension
        Narr = np.zeros(self.sweep_grid_ndims)
        for i,k in enumerate(self.sweep_grid_params):
            Narr[i] = self.__collection[k].N
        
        # Convert indices to collapsed format
        return sum([int(np.prod(Narr[i+1:]))*index for i, index in enumerate(list(indices))])
    
    def computeFuncSweep(self, func, spec, *fcn_args, **fcn_kwargs):
        """ Compute a function over a sweep. Uses the collapsed grid created by :func:`ndSweep` internally.
        
        :param func: Function reference to compute over the sweep
        :type func: function
        
        :param spec: An array of parameter sweep specifications, optionally created by :func:`paramSweepSpec`.
        :type spec: list of dict
        
        :param \*fcn_args: Positional arguments to pass to `func`.
        :type \*fcn_args: variable
        
        :param \*fcn_kwargs: Keyword arguments to pass to `func`.
        :type \*fcn_kwargs: variable
        
        :raises Exception: If the requested sweeps exceed the bounds of a parameter.
        
        :return: None
        """
        # Generate sweep matrix
        self.ndSweep(spec)
        
        # Iterate over collapsed grid
        self.sweep_grid_result = []
        for i in range(self.sweep_grid_c_len):
            params = dict([(p,self.sweep_grid_c[p][i]) for p in self.sweep_grid_params])
            self.sweep_grid_result.append(func(params,*fcn_args,**fcn_kwargs))
            
    ## Compute an expression over sweep. Use this to avoid regenerating sympy expressions.
    #
    # @param cobj A class object that contains parameters and functions that return a sympy expression.
    # @param paramcb A string that describes a class attribute containing a dictionary of sympy symbols.
    # @param exprcb A string that describes a class function that returns the sympy expression.
    # @param spec The specification generated by \ref paramSweepSpec.
    # @return None.
    #
    def computeExprSweep(self, cobj, paramcb, exprcb, spec, *expr_args, **expr_kwargs):
        """ Compute an expression over sweep. Use this to avoid regenerating sympy expressions.
        """
        # Generate sweep matrix
        self.ndSweep(spec)
        
        # Get the parameters and member function
        syparams = getattr(cobj,paramcb)
        syfunc = getattr(cobj,exprcb)
        
        # Generate the expression
        expr = syfunc(*expr_args,**expr_kwargs)
        
        # Iterate over collapsed grid
        self.sweep_grid_result = []
        for i in range(self.sweep_grid_c_len):
            params = dict([(p,self.sweep_grid_c[p][i]) for p in self.sweep_grid_params])
            subs = dict([(syparams[p],params[p]) for p in self.sweep_grid_params])
            self.sweep_grid_result.append(expr.subs(subs))
        
    def getSweepResult(self, ind_var, static_vars, data=None, key=None):
        """ Get the result of a sweep as a function of one or more independent variables.
        
        :param ind_var: The parameter name that is swept. This is the independent variable. If more than one independent variable are specified, the order in which they appear does not affect the returned result.
        :type ind_var: str, list
        
        :param static_vars: A dictionary of parameters and associated values for which to get the sweep result. The values need not correspond to the actual sweep values used. The closest value found will be used and returned.
        :type static_vars: dict
        
        :param data: A data set obtained from sweeping parameters.
        :type data: list, np.ndarray, dict, optional
        
        :param key: A string corresponding to a key if the optional `data` parameter is a `dict`.
        :type key: str, optional
        
        :raises Exception: If specified parameters are not found, or some are missing.
        
        :return: list of swept parameter arrays, the sweep result and a dictionary corresponding to the static values requested.
        
        If `ind_var` is specified as a list, the corresponding higher dimensional sweep result is returned in a mesh format.
        """
        if data is None:
            data = self.sweep_grid_result
        
        # Determine if the input data is actually a list of files
        using_tmp_files = False
        if type(data) in [list, np.ndarray]:
        
            if type(data[0]) in [str, bytes, os.PathLike]:
                using_tmp_files = True
                
                # Determine if data is keyed
                if type(util.pickleRead(data[0])) == dict:
                    if key is None:
                        raise Exception("'key' optional parameter should be specified for 'data' of type dict (in temporary files).")
                else:
                    key = None
                
        elif type(data) == dict:
            
            # Extract the keyed data
            if key is None:
                raise Exception("'key' optional parameter should be specified for 'data' of type dict.")
            data = data[key]
            
            if type(data[0]) in [str, bytes, os.PathLike]:
                using_tmp_files = True
        
        if type(ind_var) is str:
            # Check the independent variable exists
            if ind_var not in self.getParameterNamesList():
                raise Exception("Independent variable '%s' does not exist." % ind_var)
            
            # Check the static variables exist
            for k in static_vars.keys():
                if k not in self.getParameterNamesList():
                    raise Exception("Static variable '%s' does not exist." % k)
            
            if len(self.sweep_spec) == 1:
                if not using_tmp_files:
                    # FIXME: getParameterSweep will return what is asked even if data is not actually associated with that parameter sweep.
                    return self.getParameterSweep(ind_var), np.array(data).T, {}
                else:
                    if key is None:
                        return self.getParameterSweep(ind_var), np.array([util.pickleRead(f) for f in data]).T, {}
                    else:
                        return self.getParameterSweep(ind_var), np.array([util.pickleRead(f)[key] for f in data]).T, {}
            
            # Get the indices of the parameters in the sweep specification
            # and the length of the data arrays
            indices = {}
            Ns = {}
            for i,s in enumerate(self.sweep_spec):
                indices[s["name"]] = len(self.sweep_spec)-i-1
                Ns[s["name"]] = s["N"]
            
            # Find the indices of the requested points of the static parameters
            static_indices = {}
            static_vals = {}
            for k,v in static_vars.items():
                p = self.getParameterSweep(k)
                i = np.argmin(np.abs(p-v))
                static_indices[k] = i
                static_vals[k] = p[i]
            
            # Arrange the indices to extract desired values
            final = [0]*(1+len(static_vars))
            final[indices[ind_var]] = [i for i in range(Ns[ind_var])]
            for k,v in static_vars.items():
                final[indices[k]] = [static_indices[k] for i in range(Ns[ind_var])]
            final = np.array(final).T
            
            # Get the desired values
            res = []
            for l in final:
                res.append(data[self.collapsedIndices(*l[::-1])])
                
            if not using_tmp_files:
                return self.getParameterSweep(ind_var), np.array(res).T, static_vals
            else:
                if key is None:
                    return self.getParameterSweep(ind_var), np.array([util.pickleRead(f) for f in res]).T, static_vals
                else:
                    return self.getParameterSweep(ind_var), np.array([util.pickleRead(f)[key] for f in res]).T, static_vals
            
        elif type(ind_var) in [list, np.ndarray]:
            # Check the independent variables exist
            for k in ind_var:
                if k not in self.getParameterNamesList():
                    raise Exception("Independent variable '%s' does not exist." % k)
            
            # Check the static variables exist
            for k in static_vars.keys():
                if k not in self.getParameterNamesList():
                    raise Exception("Static variable '%s' does not exist." % k)
            
            # Get the indices of the parameters in the sweep specification
            # and the length of the data arrays
            # Also sort ind_var
            indices = {}
            Ns = {}
            new_ind_var = []
            for i,s in enumerate(self.sweep_spec):
                indices[s["name"]] = len(self.sweep_spec)-i-1
                Ns[s["name"]] = s["N"]
                if s["name"] in ind_var:
                    new_ind_var.append(s["name"])
            ind_var = new_ind_var
            
            # Find the indices of the requested points of the static parameters
            static_indices = {}
            static_vals = {}
            for k,v in static_vars.items():
                p = self.getParameterSweep(k)
                i = np.argmin(np.abs(p-v))
                static_indices[k] = i
                static_vals[k] = p[i]
            
            # Arrange the indices to extract desired values
            final = [0]*(len(ind_var)+len(static_vars))
            for iv in ind_var:
                final[indices[iv]] = [i for i in range(Ns[iv])]
            for k,v in static_vars.items():
                final[indices[k]] = static_indices[k]#[static_indices[k] for i in range(Ns[k])]
            mesh_nc = np.meshgrid(*final[::-1],indexing="ij")
            final = np.array([x.flatten() for x in mesh_nc]).T
            
            # Get the desired values
            res = []
            for l in final:
                res.append(data[self.collapsedIndices(*l)])
            
            # Get the shape of an entry in the data
            if not using_tmp_files:
                shape = np.array(data[self.collapsedIndices(*l)]).shape
                return [self.getParameterSweep(iv) for iv in ind_var], np.array(res).reshape(*[Ns[iv] for iv in ind_var],*shape).T, static_vals
            else:
                if key is None:
                    shape = np.array(util.pickleRead(data[self.collapsedIndices(*l)])).shape
                    return [self.getParameterSweep(iv) for iv in ind_var], np.array([util.pickleRead(f) for f in res]).reshape(*[Ns[iv] for iv in ind_var],*shape).T, static_vals
                else:
                    shape = np.array(util.pickleRead(data[self.collapsedIndices(*l)])[key]).shape
                    return [self.getParameterSweep(iv) for iv in ind_var], np.array([util.pickleRead(f)[key] for f in res]).reshape(*[Ns[iv] for iv in ind_var],*shape).T, static_vals
        else:
            raise Exception("Invalid independent variable specification. Found type '%s'" % repr(type(ind_var)))
    
    ###################################################################################################################
    #       Internal
    ###################################################################################################################
    
    def _get_pc_internal_data(self):
        return (
            self.__collection,
            self.__symbol_map,
            self.__parameterisation,
            self.__parameterisation_graph
        )
    
    def _set_pc_internal_data(self, data):
        self.__collection = data[0]
        self.__symbol_map = data[1]
        self.__parameterisation = data[2]
        self.__parameterisation_graph = data[3]
    
    # Use this with care, probably many scenarios where it would break things
    def _update_pc_internal_data(self, data):
        # Update the collection
        for param in data[0].keys():
            if param not in self.__collection.keys():
                self.__collection[param] = data[0][param]
        
        # Update the symbol map
        for param in data[1].keys():
            if param not in self.__symbol_map.keys():
                self.__symbol_map[param] = data[1][param]
        
        # Update the parameterisations
        #for param in data[2].keys():
        #    if param not in self.__parameterisation.keys():
        #        self.__parameterisation[param] = data[2][param]
        
        # Ok to just copy these for current use case
        self.__parameterisation = data[2]
        self.__parameterisation_graph = data[3]
    
    def _update_parameterisations(self):
        
        checked_nodes = set()
        checked_edges = set()
        G = self.__parameterisation_graph
        all_nodes = set(G.nodes)
        all_edges = set(G.edges)
        
        # Start with the nodes that have no in_degree (independent parameterisations)
        names = [node for node in all_nodes if G.in_degree[node] == 0]
        for name in names:
            params = self.__parameterisation[name]
            subs = self.getSymbolValues(*params['parameters'])
            #print(params['expression'])
            #print(subs)
            #print(params['expression'].subs(subs))
            synum = params['expression'].subs(subs)
            num = float(synum)
            self.__collection[name].setValue(num)
            checked_nodes.add(name)
        if checked_nodes == all_nodes:
            #print("All nodes checked")
            return
        
        # Now we need to iterate until all nodes are accounted for
        while checked_edges != all_edges:# and not self.allParametersSet():
            # Get the next set of names to check
            new_names = []
            for name in names:
                edges = G.out_edges(name)
                new_names.extend([edge[1] for edge in edges])
                checked_edges |= set(edges)
            
            # Perform the update
            for name in new_names:
                params = self.__parameterisation[name]
                subs = self.getSymbolValues(*params['parameters'])
                #print(params['expression'])
                #print(subs)
                #print(params['expression'].subs(subs))
                synum = params['expression'].subs(subs)
                num = float(synum)
                self.__collection[name].setValue(num)
            
            names = new_names.copy()
        
        #try:
        #    for k, v in self.__parameterisation.items():
        #        subs = self.getSymbolValues(*v['parameters'])
        #        self.__collection[k].setValue(float(v['expression'].subs(subs)))
        #except:
        #    pass












