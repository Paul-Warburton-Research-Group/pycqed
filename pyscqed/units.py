""" The :py:mod:`pycqed.src.util` module defines the class :class:`Units` which is used to specify a unit system for the parameters of the system and to convert quantities between different units.
"""
import numpy as np
from . import physical_constants as pc

## Unit system class
#
# A class used for easily switching between units.
#
# Instances of this class should be built into a library
# so that they can be selected from a list in a user program
#
# Ideas:
#
#  - Use 'sympy' unit system? More natural choice considering this is used for symbolic computation
#  - Use 'pint' library? https://pint.readthedocs.io/en/0.9/tutorial.html
#
#
# How to implement
#
# - Specify units for all parameters types:
#
# capacitors, inductors, currents, voltages, fluxes, charges
# energy, frequency, time
#
# - Encode prefixes, m -> 1e-3, k -> 1e3, G -> 1e9 for example
#
# For example frequency:
#
# setUnit('GHz'):
#   extract [0] for prefix
#   extract rest for quantity
# 
# - Encode prefined unit setups
#

#: Unit system presets
units_presets = {
    "CQED1":{
        "Energy":{'unit':'GHz', 'factor':1/(pc.h*1e9)},
        "Current":{'unit':'uA', 'factor':1/1e-6},
        "Voltage":{'unit':'uV', 'factor':1/1e-6},
        "F":{'unit':'fF', 'factor':1e-15},
        "H":{'unit':'pH', 'factor':1e-12},
        "Ohm":{'unit':'Ohm', 'factor':1.0},
        "A":{'unit':'uA', 'factor':1e-6},
        "V":{'unit':'uV', 'factor':1e-6},
        "Hz":{'unit':'GHz', 'factor':1e9},
        "Wb":{'unit':'phi0', 'factor':pc.phi0},
        "C":{'unit':'2e', 'factor':2*pc.e},
        "J":{'unit':'J', 'factor':1.0},
        "K":{'unit':'K', 'factor':1.0},
        "eV":{'unit':'eV', 'factor':1.0}
    },
    "CQED2":{
        "Energy":{'unit':'GHz', 'factor':1/(pc.h*1e9)},
        "Current":{'unit':'nA', 'factor':1/1e-9},
        "Voltage":{'unit':'nV', 'factor':1/1e-9},
        "F":{'unit':'fF', 'factor':1e-15},
        "H":{'unit':'pH', 'factor':1e-12},
        "Ohm":{'unit':'Ohm', 'factor':1.0},
        "A":{'unit':'nA', 'factor':1e-9},
        "V":{'unit':'nV', 'factor':1e-9},
        "Hz":{'unit':'GHz', 'factor':1e9},
        "Wb":{'unit':'phi0', 'factor':pc.phi0},
        "C":{'unit':'2e', 'factor':2*pc.e},
        "J":{'unit':'J', 'factor':1.0},
        "K":{'unit':'K', 'factor':1.0},
        "eV":{'unit':'eV', 'factor':1.0}
    }
}


class Units:
    """ This class initialises the prefactors used in :class:`HamilSpec` and can be used to convert between different unit systems. A series of presets are available.
    
    :param name: The name of the unit system. Can be the name of a preset.
    :type name: str
    
    :param system: The unit system to use. Can be "SI" only at the moment.
    :type system: str, optional
    
    :return: A new instance of :class:`Units`.
    :rtype: :class:`Units`
    """
    
    __unit_systems = ["SI","cgs"]
    __SI_units = ["F","H","Ohm","A","V","Hz","Wb","C","J","K","eV"]
    __SI_units_latex = {
        "F":"\\mathrm{F}",            # Farad
        "H":"\\mathrm{H}",            # Henry
        "Ohm":"\\mathrm{\\Omega}",    # Ohm
        "A":"\\mathrm{A}",            # Ampere
        "V":"\\mathrm{V}",            # Volt
        "Hz":"\\mathrm{Hz}",          # Hertz
        "Wb":"\\mathrm{Wb}",          # Weber
        "C":"\\mathrm{C}",            # Coulomb
        "J":"\\mathrm{J}",            # Joule
        "K":"\\mathrm{K}",            # Kelvin
        "eV":"\\mathrm{eV}"           # Electron-Volt
    }
    __output_units = ["Energy","Current","Voltage"]
    __prefixes = {
        "E":1e18,
        "P":1e15,
        "T":1e12,
        "G":1e9,
        "M":1e6,
        "k":1e3,
        "h":1e2,
        "D":1e1,
        "0":1e0,
        "d":1e-1,
        "c":1e-2,
        "m":1e-3,
        "u":1e-6,
        "n":1e-9,
        "p":1e-12,
        "f":1e-15,
        "a":1e-18
    }
    __prefixes_latex = {
        "E":"\\mathrm{E}",
        "P":"\\mathrm{P}",
        "T":"\\mathrm{T}",
        "G":"\\mathrm{G}",
        "M":"\\mathrm{M}",
        "k":"\\mathrm{k}",
        "h":"\\mathrm{h}",
        "D":"\\mathrm{D}",
        "0":"",
        "d":"\\mathrm{d}",
        "c":"\\mathrm{c}",
        "m":"\\mathrm{m}",
        "u":"\\mathrm{\\mu}",
        "n":"\\mathrm{n}",
        "p":"\\mathrm{p}",
        "f":"\\mathrm{f}",
        "a":"\\mathrm{a}"
    }
    
    def __init__(self, name, system="SI"):
        
        # Initialise the base unit conversion values
        self.c = {}
        for u in self.__SI_units:
            self.c[u] = {'unit':u, 'factor':1.0}
        
        # Initialise the output unit conversion values
        self.cd = {}
        self.cd['Energy'] = {'unit':'J','factor':1.0}
        self.cd['Current'] = {'unit':'A', 'factor':1.0}
        self.cd['Voltage'] = {'unit':'V', 'factor':1.0}
        
        # Initialise the prefactors
        self._updatePrefactors()
        
        # Check if name in preset
        if name in units_presets.keys():
            self.loadPreset(name)
    
    def loadPreset(self, preset):
        """ Load a preset.
        
        :param preset: The name of the preset.
        :type preset: str
        
        :return: None
        
        All prefactors are updated following this call.
        """
        for key in self.__output_units:
            self.cd[key] = units_presets[preset][key]
        for key in self.__SI_units:
            self.c[key] = units_presets[preset][key]
        self._updatePrefactors()
    
    def setUnit(self, new_unit):
        """ Set an input unit. The prefix and the unit are detected to determine the correct factor.
        
        :param new_unit: The new unit to use, for example 'mA' for milli-Ampere.
        :type new_unit: str
        
        :raises Exception: If the base unit or the prefix are not found.
        
        :return: None
        
        All prefactors are updated following this call.
        """
        # Check if just the unit without prefix
        if new_unit in self.__SI_units:
            self.c[new_unit] = {'unit':new_unit, 'factor':1.0}
            self.updatePrefactors()
            return
        
        # Get the prefix and the unit name
        p = new_unit[0]
        u = new_unit[1:]
        if u not in self.__SI_units:
            raise Exception("SI unit '%s' not found" % u)
        self.c[u] = {'unit':new_unit, 'factor':self.__prefixes[p]}
        self._updatePrefactors()
    
    def setUnitPrefactor(self, unit, new_unit, value):
        """ Set an input unit prefactor directly. This allows specifying units in terms of physical constants.
        
        :param unit: The base unit of the selected system to use, form example 'A' for Ampere in the SI system.
        :type new_unit: str
        
        :param new_unit: A string describing the new unit.
        :type new_unit: str
        
        :param value: The value to set the conversion factor to.
        :type value: float
        
        :raises Exception: If the base unit is not found.
        
        :return: None
        
        All prefactors are updated following this call.
        """
        if unit not in self.__SI_units:
            raise Exception("SI unit '%s' not found" % unit)
        self.c[unit] = {'unit':new_unit, 'factor':value}
        self._updatePrefactors()
    
    def getUnitPrefactor(self, unit):
        """ Get an unit prefactor.
        
        :param unit: The base unit of the selected system to get, form example 'A' for Ampere in the SI system.
        :type new_unit: str
        
        :raises Exception: If the base unit is not found.
        
        :return: The unit prefactor
        
        All prefactors are updated following this call.
        """
        if unit not in self.__SI_units:
            raise Exception("SI unit '%s' not found" % unit)
        return self.c[unit]['factor']
    
    def setEnergyUnit(self, new_unit):
        """ Set the output energy unit. The prefix and the unit are detected to determine the correct factor. A number of commonly used units are available, such as Kelvin and electron-Volts.
        
        :param new_unit: The new unit to use, for example 'keV' for kilo-electron-Volts.
        :type new_unit: str
        
        :raises Exception: If the base unit or the prefix are not found.
        
        :return: None
        
        All prefactors are updated following this call.
        """
        if new_unit in self.__SI_units:
            unit = new_unit
            p = "0"
        else:
            unit = new_unit[1:]
            if unit not in self.__SI_units:
                raise Exception("SI unit '%s' not found" % unit)
            p = new_unit[0]
        
        if unit == "Hz":
            self.cd['Energy'] = {'unit':new_unit, 'factor':1/(pc.h*self.__prefixes[p])}
        elif unit == "J":
            self.cd['Energy'] = {'unit':new_unit, 'factor':1/self.__prefixes[p]}
        elif unit == "K":
            self.cd['Energy'] = {'unit':new_unit, 'factor':1/(pc.kB*self.__prefixes[p])}
        elif unit == "eV":
            self.cd['Energy'] = {'unit':new_unit, 'factor':1/(pc.e*self.__prefixes[p])}
        else:
            raise Exception("incompatible unit for Energy '%s'" % unit)
        self._updatePrefactors()
    
    def setCurrentUnit(self, new_unit):
        """ Set the output current unit. The prefix and the unit are detected to determine the correct factor.
        
        :param new_unit: The new unit to use, for example 'mA' for milli-Ampere.
        :type new_unit: str
        
        :raises Exception: If the base unit or the prefix are not found.
        
        :return: None
        
        All prefactors are updated following this call.
        """
        if new_unit in self.__SI_units:
            unit = new_unit
            p = "0"
        else:
            unit = new_unit[1:]
            if unit not in self.__SI_units:
                raise Exception("SI unit '%s' not found" % unit)
            p = new_unit[0]
        if unit == "A":
            self.cd['Current'] = {'unit':new_unit, 'factor':1/self.__prefixes[p]}
        else:
            raise Exception("incompatible unit for Current '%s'" % unit)
        self._updatePrefactors()
    
    def setVoltageUnit(self, new_unit):
        """ Set the output voltage unit. The prefix and the unit are detected to determine the correct factor.
        
        :param new_unit: The new unit to use, for example 'mV' for milli-Volt.
        :type new_unit: str
        
        :raises Exception: If the base unit or the prefix are not found.
        
        :return: None
        
        All prefactors are updated following this call.
        """
        if new_unit in self.__SI_units:
            unit = new_unit
            p = "0"
        else:
            unit = new_unit[1:]
            if unit not in self.__SI_units:
                raise Exception("SI unit '%s' not found" % unit)
            p = new_unit[0]
        if unit == "V":
            self.cd['Voltage'] = {'unit':new_unit, 'factor':1/self.__prefixes[p]}
        else:
            raise Exception("incompatible unit for Voltage '%s'" % unit)
        self._updatePrefactors()
    
    def convertEnergy(self, data, old_unit, new_unit):
        """ Convert given data between energy units.
        """
        pass
    
    def convertCurrent(self, data, old_unit, new_unit):
        """ Convert given data between current units.
        """
        pass
    
    def convertVoltage(self, data, old_unit, new_unit):
        """ Convert given data between voltage units.
        """
        pass
    
    def getPrefactor(self, name):
        """ Get the unit conversion value for a specific term.
        
        :param name: The name of the conversion value to get.
        :type name: str
        
        :return: None
        """
        return self.prefactors[name]
    
    def _getEcUnit(self):
        """ Get the charging energy unit.
        """
        return self.cd['Energy']['factor']*self.c['C']['factor']**2/self.c['F']['factor']
    
    def _getElUnit(self):
        return self.cd['Energy']['factor']*self.c['Wb']['factor']**2/self.c['H']['factor']
    
    def _getEjUnit(self):
        return self.cd['Energy']['factor']*self.c['Wb']['factor']*self.c['A']['factor']/(2*np.pi)
    
    def _getEpUnit(self):
        return self.cd['Energy']['factor']*self.c['C']['factor']*self.c['V']['factor']/(2*np.pi)
    
    def _getFreqUnit(self):
        return 1.0/(self.c['Hz']['factor']*np.sqrt(self.c['F']['factor']*self.c['H']['factor'])*2*np.pi)
    
    def _getImpeUnit(self):
        return np.sqrt(self.c['H']['factor']/self.c['F']['factor'])
    
    def _getFlxOscUnit(self):
        return np.sqrt(pc.hbar)/self.c['Wb']['factor']
    
    def _getChgOscUnit(self):
        return np.sqrt(pc.hbar)/self.c['C']['factor']
    
    def _getVopUnit(self):
        return self.cd['Voltage']['factor']*self.c['C']['factor']/self.c['F']['factor']
    
    def _getIopLUnit(self):
        return self.cd['Current']['factor']*self.c['Wb']['factor']/self.c['H']['factor']
    
    def _getIopJUnit(self):
        return self.cd['Current']['factor']*self.c['A']['factor']
    
    def _getFlxUnit(self):
        return self.c['Wb']['factor']
    
    def _getChgUnit(self):
        return self.c['C']['factor']
    
    def _getPhaUnit(self):
        return self.c['Wb']['factor']/pc.phi0
    
    def _getChgOscCplUnit(self):
        return self.cd['Energy']['factor']*np.sqrt(pc.hbar)*self.c['C']['factor']/(self.c['F']['factor']*(self.c['H']['factor']/self.c['F']['factor'])**(0.25))
    
    def _getFlxOscCplUnit(self):
        return 0.0
    
    def _updatePrefactors(self):
        self.prefactors = {
            "Ec":self._getEcUnit(),
            "El":self._getElUnit(),
            "Ej":self._getEjUnit(),
            "Ep":self._getEpUnit(),
            "Freq":self._getFreqUnit(),
            "Impe":self._getImpeUnit(),
            "FlxOsc":self._getFlxOscUnit(),
            "ChgOsc":self._getChgOscUnit(),
            "Vop":self._getVopUnit(),
            "IopL":self._getIopLUnit(),
            "IopJ":self._getIopJUnit(),
            "Flx":self._getFlxUnit(),
            "Chg":self._getChgUnit(),
            "Pha":self._getPhaUnit(),
            "ChgOscCpl":self._getChgOscCplUnit(),
            "FlxOscCpl":self._getFlxOscCplUnit()
        }













