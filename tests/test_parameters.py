"""Test the parameter API."""
import unittest

import numpy as np

from pyscqed.parameters import Param, ParamCollection


class ParamTest(unittest.TestCase):
    """Test the parameters.Param class."""

    def setUp(self):
        """Run before tests"""

    def tearDown(self):
        """Run after tests"""

    def test_parameter_creation_rules(self):
        # Naming rules
        self.assertRaises(TypeError, Param, 1)
        self.assertRaises(ValueError, Param, "bad name")
        p = Param("good_name")
        self.assertEqual(p.getValue(), None)
        # Value rules
        self.assertRaises(TypeError, Param, "good_name", value="s")
        for val in [0.5, 1, np.float64(1.2)]:
            p = Param("good_name", value=val)
            self.assertEqual(p.getValue(), val)
        # Bounds rules
        self.assertRaises(TypeError, Param, "good_name", bounds="s")
        self.assertRaises(TypeError, Param, "good_name", bounds=[1.0, "s"])
        self.assertRaises(TypeError, Param, "good_name", bounds=["s", 1.0])
        self.assertRaises(ValueError, Param, "good_name", bounds=[1.0, -1.0])
        p = Param("good_name", bounds=[-0.5, 0.5])
        self.assertRaises(ValueError, p.setValue, -0.50001)
        p.setValue(0.2)
        self.assertEqual(p.getValue(), 0.2)
        # Unit prefactor rules
        self.assertRaises(TypeError, Param, "good_name", unit_pref="s")
        self.assertRaises(ValueError, Param, "good_name", unit_pref=-1.0)
        # Latex names
        self.assertEqual(p.name_latex, "g_\\mathrm{ood_name}")

    def test_parameter_functions(self):
        p = Param("good_name", bounds=[-0.5, 0.5])
        self.assertEqual(p.getBounds(), [-0.5, 0.5])
        self.assertRaises(TypeError, p.setBounds, "s")
        self.assertRaises(TypeError, p.setBounds, [1.0, "s"])
        self.assertRaises(TypeError, p.setBounds, ["s", 1.0])
        self.assertRaises(ValueError, p.setBounds, [1.0, -1.0])
        p.setBounds([-0.5, 0.5])
        sweep = p.linearSweep(-0.5, 0.5, 2)
        self.assertTrue(all(x == y for x, y in zip(sweep, p.sweep)))
        self.assertRaises(TypeError, p.linearSweep, "s", 1, 2)
        self.assertRaises(TypeError, p.linearSweep, 1, "s", 2)
        self.assertRaises(TypeError, p.linearSweep, 0., 1, "s")
        self.assertRaises(TypeError, p.linearSweep, 0, 1., 2.0)
        self.assertRaises(ValueError, p.linearSweep, 0, 1.0, 2)
        self.assertRaises(ValueError, p.linearSweep, -0.501, 1.0, 2)


class ParamCollectionTest(unittest.TestCase):
    """Test the parameters.ParamCollection class."""

    def setUp(self):
        """Run before tests"""
        self.names = ["Jc", "L1", "L2", "Long_one"]
        self.values = [1.0, -0.5, -0.25, 1e-9]

    def tearDown(self):
        """Run after tests"""

    def test_paramcollection_creation_rules(self):
        ParamCollection(self.names)
        self.assertRaises(TypeError, ParamCollection, [1])
        ParamCollection([])

    def test_getters(self):
        pc = ParamCollection(self.names)
        self.assertTrue(all(x == y for x, y in zip(pc.getParameterList(), self.names)))

    def test_setters(self):
        pc = ParamCollection(self.names)
        self.assertFalse(pc.allParametersSet())
        set1 = {self.names[i]: self.values[i] for i in range(len(self.names))}
        pc.setParameterValues(set1)
        set2 = []
        for i in range(len(self.names)):
            set2.append(self.names[i])
            set2.append(self.values[i])
        pc.setParameterValues(*set2)
        self.assertTrue(pc.allParametersSet())

    def test_can_use_param_with_none_value(self):
        pass
