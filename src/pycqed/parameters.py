""" The :py:mod:`pycqed.src.parameters` module defines two classes :class:`Param` and :class:`ParamCollection` that are used to manipulate scalar parameters used in simulations and experiments.
"""
import os
import numpy as np
from . import text2latex as t2l
from . import util

class Param:
    """This class defines the properties of a parameter used in simulations and experiments. It supports upper and lower bounds and the generation of sweeps for scalars.
    
    :param name: The utf name of the parameter.
    :type name: str
    
    :param value: The initial value to set the parameter to, defaults to `0.0`.
    :type value: int float np.float64, optional
    
    :param bounds: The lower and upper bounds of the parameter, defaults to `[-np.inf, np.inf]`.
    :type bounds: list of floats, optional
    
    :raises Exception: If the parameters are not accepted types or the initial value is out of bounds.
    
    :return: A new instance of :class:`Param`.
    :rtype: :class:`Param`
    
    This constructor attempts to generate a latex version of the parameter name for use with plots. It uses the :py:mod:`pycqed.src.text2latex` module for this functionality.
    
    The following class attributes are accessible by the user:
    
    :ivar name: The utf string of the parameter.
    :ivar name_latex: A string to be used with latex.
    :ivar sweep: The sweep array.
    """
    __types = ["type1", "type2"]
    __valid_scalar_types = [int, float, np.float64]
    
    def __init__(self, name, value=0.0, bounds=[-np.inf, np.inf]):
        """Constructor method."""
        # Ensure name is a string
        if type(name) is not str:
            raise Exception("'name' is not a string.")
        # Ensure value is a float
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
        
        self.name = name
        self.__value = float(value)
        self.__lower_bound = float(bounds[0])
        self.__upper_bound = float(bounds[1])
        self.name_latex = t2l.latexify_param_name(self.name)
        self.sweep = np.array([])
    
    def getValue(self):
        """ Get the current value of the parameter.
        
        :raises Exception: If the value is out of bounds.
        
        :return: The current value of the parameter.
        :rtype: int float np.float64
        """
        # Check bounds
        if self.__value > self.__upper_bound:
            raise Exception("Param %s 'value' exceeds specified upper bound." % (self.name))
        if self.__value < self.__lower_bound:
            raise Exception("Param %s 'value' exceeds specified lower bound." % (self.name))
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
        return [self._lower_bound,self.__upper_bound]
    
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
        
        self.sweep = np.linspace(float(start),float(end),N)
        self.N = N
        return self.sweep

class ParamCollection:
    """ This class uses an array of :class:`Param` instances and provides methods to manipulate them in useful ways, for example to create multidimensional sweeps.
    
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
        for name in names:
            self.__collection[name] = Param(name)
    
    def getParameterList(self):
        """ Gets the parameter dictionary, mapping the utf name to the :class:`Param` instance.
        
        :return: A dictionary of param names to :class:`Param` instances.
        :rtype: dict
        """
        return self.__collection
    
    def getParameterValuesDict(self):
        """ Returns a dictionary of the current set values of the parameters in a dictionary format.
        
        :return: A dictionary of all param names to values.
        :rtype: dict
        """
        return {k: v.getValue() for k, v in self.__collection.items()}
    
    def addParameter(self, name):
        """ Adds a new parameter to the collection if it does not already exist. If it does exist, nothing is reported.
        
        :param name: The name of the parameter to add.
        :type name: str
        
        :return: None
        """
        if name not in list(self.__collection.keys()):
            self.__collection[name] = Param(name)
        else:
            return
    
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
    
    def getParameterNamesList(self):
        """ Gets the list of available parameter names in the collection.
        
        :return: The list of parameter names.
        :rtype: list
        """
        return list(self.__collection.keys())
    
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
        if name not in list(self.__collection.keys()):
            raise Exception("'%s' parameter was not found." % name)
        self.__collection[name].setValue(value)
    
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
            for k,v in name_value_pairs[0].items():
                self.__collection[k].setValue(v)
            return
        
        # Seperate names from values
        keys = list(name_value_pairs)[::2]
        values = list(name_value_pairs)[1::2]
        
        # Check there are as many parameters as values
        if len(keys) != len(values):
            raise Exception("'name_value_pairs' definition invalid.")
        
        for i,name in enumerate(keys):
            # Check name is defined
            if name not in list(self.__collection.keys()):
                raise Exception("'%s' parameter was not found." % name)
            self.__collection[name].setValue(values[i])
    
    def getParameterValues(self, *names, dictionary=False):
        """ Get the values of many parameters.
        
        :param \*names: Arguments list, formatted as the parameter names.
        :type \*names: str, str ...
        
        :param dictionary: Specifies whether to return the result as a dictionary, defaults to `False`
        :type dictionary: bool, optional
        
        :raises Exception: If the argument types are incorrect, ill-formatted or not found.
        
        :return: A list if `dictionary=False`, a dictionary with the parameter names as keys if `dictionary=True`
        :rtype: list or dict
        """
        values = []
        for name in list(names):
            # Check name is defined
            if name not in list(self.__collection.keys()):
                raise Exception("'%s' parameter was not found." % name)
            values.append(self.__collection[name].getValue())
        
        # Return as dictionary if required
        if dictionary:
            return dict(zip(list(names),values))
        return values
    
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
        return {"name":name,"start":swp[0],"end":swp[1],"N":swp[2]}
    
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
            sweeps.append(self.__collection[k].linearSweep(param_spec["start"],param_spec["end"],param_spec["N"]))
            self.sweep_grid_npts *= param_spec["N"]
        self.sweep_grid_params = keys
        self.sweep_grid_ndims = len(sweeps)
        
        # Generate mesh grid
        grid = np.meshgrid(*sweeps,indexing="ij")
        
        # Non-collapsed grid
        self.sweep_grid_nc = dict(zip(keys,grid))
        
        # Collapsed grid
        self.sweep_grid_c = {}
        for i,k in enumerate(keys):
            self.sweep_grid_c[k] = self.sweep_grid_nc[k].flatten()
            if i == 0:
                self.sweep_grid_c_len = len(self.sweep_grid_c[k])
    
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
        return sum([int(np.prod(Narr[i+1:]))*index for i,index in enumerate(list(indices))])
    
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
    
    


