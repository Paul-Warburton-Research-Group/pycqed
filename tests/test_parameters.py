"""Test the parameter API."""
import unittest
import os

import numpy as np
import sympy as sy
import graphviz as gv

from pyscqed.parameters import Param, ParamCollection


def check_numerically_equal(Expr1, Expr2, n=100):
    """ Adapted from https://stackoverflow.com/questions/37112738/sympy-comparing-expressions
    """
    # Determine over what range to generate random numbers
    sample_min = -1
    sample_max = 1

    # Regroup all free symbols from both expressions
    free_symbols = set(Expr1.free_symbols) | set(Expr2.free_symbols)

    # Numeric (brute force) equality testing n-times
    for i in range(n):
        your_values = np.random.uniform(sample_min, sample_max, len(free_symbols))
        Expr1_num=Expr1
        Expr2_num=Expr2
        for symbol, number in zip(free_symbols, your_values):
            Expr1_num=Expr1_num.subs(symbol, sy.Float(number))
            Expr2_num=Expr2_num.subs(symbol, sy.Float(number))
        Expr1_num=complex(Expr2_num)
        Expr2_num=complex(Expr2_num)
        if not np.allclose(Expr1_num, Expr2_num, rtol=0, atol=1e-15):
            return False
    return True


def check_symbolically_equal(Expr1, Expr2):
    if (Expr1.equals(Expr2)):
        return True
    return False


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

    def test_parametric_expressions_symbolic(self):
        # Setup the param collection
        pc = ParamCollection(self.names)
        pc.getSymbols("Ltot", "wweird")
        self.assertTrue({"Ltot", "wweird"} < set(pc.getParameterNamesList()))

        # Can create a new parametric expression that depends on existing parameters
        symbols = pc.getSymbolList()
        pc.addParameterisation("Ltot", (symbols["L1"] + symbols["L2"]) * 0.5)
        self.assertTrue({"Ltot"} == set(pc.getParametricParametersList()))
        expected = symbols["L1"]/2 + symbols["L2"]/2
        self.assertTrue(sy.simplify(pc.getParametricExpression("Ltot") - expected) == 0)

        # Can create a new parametric expression that makes existing parameters dependent on others
        pc.addParameterisation("Long_one", (10*sy.cos(2*sy.pi*symbols["wweird"]*symbols["Jc"])))
        self.assertTrue({"Long_one", "Ltot"} == set(pc.getParametricParametersList()))

        # Cannot create a circular dependency
        self.assertRaises(AssertionError, pc.addParameterisation, "wweird", 2 * symbols["Long_one"])
        self.assertTrue({"Long_one", "Ltot"} == set(pc.getParametricParametersList()))

        # Can create a nested parameterisation
        pc.addParameterisation("wweird", sy.sqrt(symbols["Ltot"]))
        self.assertTrue({"wweird", "Long_one", "Ltot"} == set(pc.getParametricParametersList()))
        self.assertTrue(check_numerically_equal(pc.getParametricExpression("Ltot"), expected))

        # By default the nested parameterisations are not expanded
        expected = sy.sqrt(symbols["L1"]/2 + symbols["L2"]/2)
        self.assertFalse(check_symbolically_equal(pc.getParametricExpression("wweird"), expected))
        self.assertTrue(check_numerically_equal(pc.getParametricExpression("wweird", True), expected))

        # Can get the parameters involved in the parameterisation to depth 1.
        self.assertTrue(set(pc.getParameterisationParameters("Long_one")) ==  set(['Jc', 'wweird']))

        # Can remove parameterisations
        pc.rmParameterisation("Ltot")
        self.assertTrue({"wweird", "Long_one"} == set(pc.getParametricParametersList()))
        pc.rmParameterisation("Long_one")
        self.assertTrue({"wweird"} == set(pc.getParametricParametersList()))
        expected = symbols["L1"]/2 + symbols["L2"]/2
        self.assertRaises(ValueError, pc.getParametricExpression, "Ltot")
        self.assertRaises(ValueError, pc.getParametricExpression, "Long_one")

        # Can write get the parameterisation graph
        obj = pc.drawParameterisationGraph("test1.dot")
        self.assertTrue(type(obj) == gv.Source)
        self.assertTrue(os.path.exists("./test1.dot"))
        os.remove("./test1.dot")  # Comment to visualize with `dot -Tsvg .\test.dot -o test.svg`

        # Cannot redefine a parameter using the same name
        pc.getSymbols("a")
        symbols = pc.getSymbolList()
        self.assertRaises(AssertionError, pc.addParameterisation, "L1", symbols["L1"] * symbols["a"])

    def test_parametric_expressions_numeric(self):
        pc = ParamCollection(self.names)
        pc.getSymbols("Ltot", "wweird")
        symbols = pc.getSymbolList()
        pc.addParameterisation("Ltot", (symbols["L1"] + symbols["L2"]) * 0.5)
        pc.addParameterisation("Long_one", (10*sy.cos(2*sy.pi*symbols["wweird"]*symbols["Jc"])))
        pc.addParameterisation("wweird", sy.sqrt(symbols["Ltot"]))
        # Partial setting of parameters is allowed, parameters that cannot be updated remain None
        pc.setParameterValues({
            "L1": 0.5,
            "L2": 2.0
        })
        values_dict = pc.getParameterValuesDict()
        # Only "Long_one" parameter is missing a value, as it depends on "Jc"
        self.assertTrue(values_dict['Long_one'] is None)
        pc.setParameterValues({
            "Jc": 0.1,
            "L1": 1.0,
            "L2": 2.0
        })
        values_dict = pc.getParameterValuesDict()
        L1 = pc.getParameterValue("L1")
        L2 = pc.getParameterValue("L2")
        Jc = pc.getParameterValue("Jc")
        # Does substitution honour the parametric equations?
        self.assertTrue(np.isclose(values_dict['Ltot'], 0.5 * (L1 + L2), rtol=0, atol=1e-15))
        self.assertTrue(np.isclose(values_dict['wweird'], np.sqrt(0.5 * (L1 + L2)), rtol=0, atol=1e-15))
        self.assertTrue(np.isclose(
            values_dict['Long_one'], 10*np.cos(2*np.pi*np.sqrt(0.5 * (L1 + L2))*Jc), rtol=0, atol=1e-14)
        )

    def test_parameter_sweeping(self):
        pass

    def test_can_use_param_with_none_value(self):
        pass
