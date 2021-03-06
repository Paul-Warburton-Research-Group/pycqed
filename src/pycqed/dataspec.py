""" The :py:mod:`pycqed.src.dataspec` module defines the classes :class:`TempData` and :class:`ProjectData`.

The class :class:`TempData` is used as a base class to provide functionality on how to save and handle data generated by the classes :class:`CircuitSpec`, :class:`HamilSpec` and :class:`ParamCollection` for memory management purposes.

The class :class:`ProjectData` is used to create structured output for later use. This is a mechanism for saving work in a consistent way.

This module also defines a series of plotting utilities for data representation, based on `matplotlib`.
"""

#
# FIXME: Would be useful to have a logging capability for time-consuming codes that run outside notebooks.
# Should consider an asynchronous logger such as https://github.com/b2wdigital/aiologger
#

import datetime as dt
import pickle
import platform
import os
import sys
import glob
import numpy as np
import shutil
import uuid
import tempfile
from . import util
#import util

class TempData:
    """ This class is used to manage data generated by the various classes used in PyCQED. It provides a mechanism for saving data to disk temporarily rather than using RAM.
    
    On instantiation the root temporary directory is created in a suitable location determined by the operating system.
    """
    
    __temp_out_root_suffix = ".pycqed" + os.sep # if platform.system() in ["Linux","Java"] else "PyCQED" + os.sep # FIXME: Put temp data somewhere other than home directory in case of Windows?
    __session_default_prefix = "session-"
    __part_default_prefix = "part-"
    
    def __init__(self):
        
        # Set the temp data directory
        self.temp_out_root = tempfile.gettempdir() + os.sep + self.__temp_out_root_suffix
        if not os.path.isdir(self.temp_out_root):
            print ("Creating PyCQED tmp output root directory '%s'." % self.temp_out_root)
            os.makedirs(self.temp_out_root)
        else:
            print ("Using existing PyCQED tmp output root directory '%s'." % self.temp_out_root)
        
        # Set attributes to defaults
        self.session_path = self.temp_out_root
        self.part_prefix = self.__part_default_prefix
        self.__session_exists = False
        self.part_count = 0
    
    def newSession(self, obj_id):
        """ A new session should be created in association with the lifetime of a PyCQED class. During the lifetime of that class, any temporary data will be written in that session so that it can be later retrieved. This prevents the temporary data from being overwritten if a new instance of the same PyCQED class is created. Using the objects memory address to uniquely identify it means the same directory will be used for that object.
        
        A new tmp directory will be created in the root tmp directory to identify this session, named after the hex representation of the memory address.
        
        :param obj_id: The memory address of the object obtained using `id(my_obj)`.
        :type obj_id: int
        
        :return: The path to the tmp directory associated with this session.
        :rtype: str
        """
        
        # Generate a unique ID from the data
        identifier = self.__session_default_prefix+hex(obj_id)[2:]
        
        # Create the session tmp directory
        self.session_path = self.temp_out_root + identifier + os.sep
        if not os.path.isdir(self.session_path):
            os.makedirs(self.session_path)
        self.__session_exists = True
        return self.session_path
    
    def newPrefix(self, pref_id):
        """ A new prefix should be created in association with a particular dataset generated by a PyCQED class. During the lifetime of a class, if it has an associated session, multiple datasets may be generated. A new prefix allows different datasets to be uniquely identified in the temporary storage space, otherwise they will be overwritten using the default prefix. A unique prefix can be generated from the parameters names and values that were used to generate the dataset.
        
        Currently this isn't necessary as during the lifetime of this class, the part number is incremented continuously, and thus data will not be overwritten.
        """
        pass
    
    def writePart(self, data):
        """ Write a temporary data file and return its name so that it can be saved for later use. The data is pickled and written to a binary file.
        
        :param data: An arbitrary python object that must be serialisable.
        :type data: object
        
        :return: The name of the data file including its path.
        :rtype: str
        """
        
        # FIXME: Dispatch threads to write these files to not impact performance?
        # Would only be needed in cases where each data point is evaluated very fast, in which case
        # typically temp files would not be needed, unless there really are so many points.
        
        # Generate filename
        fname = self.session_path + self.part_prefix + str(self.part_count) + ".bin"
        
        # Write it
        util.pickleWrite(data, fname)
        self.part_count += 1
        
        return fname
        
    def readPart(self, filename):
        """ Read a temporary data file with pickled data contained within.
        
        :param filename: The path to the binary file to read.
        :type filename: str
        
        :return: The object retrieved from the file.
        :rtype: object
        """
        
        return util.pickleRead(filename)
    
    def sessionExists(self):
        """ Checks if a session exists.
        
        :return: True if the current object has an associated session, else False.
        :rtype: bool
        """
        
        return self.__session_exists
    
    def clearSessionData(self):
        """ Deletes temporary data associated with the current session. Should be called when an object is deleted ideally, to ensure the HDD is not filled with excessive amounts of data. In any case the OS should automatically delete temporary files at least following a power cycle.
        """
        
        # Should check this is safe really
        if self.__session_exists:
            shutil.rmtree(self.session_path)
            self.__session_exists = False
        self.part_count = 0
        self.session_path = ""

class ProjectData:
    pass
