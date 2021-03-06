"""Unit tests for evaluating functions on non-matching meshes"""

# Copyright (C) 2013 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2013-12-14
# Last changed:

import unittest
import numpy
from dolfin import *

class Quadratic2D(Expression):
    def eval(self, values, x):
        values[0] = x[0]*x[0] + x[1]*x[1] + 1.0

class Quadratic3D(Expression):
    def eval(self, values, x):
        values[0] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + 1.0

class NonmatchingFunctionInterpolationTest(unittest.TestCase):

    def test_functional2D(self):
        """Test integration of function interpolated in non-matching meshes"""

        f = Quadratic2D()

        # Interpolate quadratic function on course mesh
        mesh0 = UnitSquareMesh(8, 8)
        V0 = FunctionSpace(mesh0, "Lagrange", 2)
        u0 = Function(V0)
        u0.interpolate(f)

        if MPI.size(mesh0.mpi_comm()) == 1:
            # Interpolate FE function on finer mesh
            mesh1 = UnitSquareMesh(31, 31)
            V1 = FunctionSpace(mesh1, "Lagrange", 2)
            u1 = Function(V1)
            u1.interpolate(u0)
            self.assertAlmostEqual(assemble(u0*dx), assemble(u1*dx), 10)

            mesh1 = UnitSquareMesh(30, 30)
            V1 = FunctionSpace(mesh1, "Lagrange", 2)
            u1 = Function(V1)
            u1.interpolate(u0)
            self.assertAlmostEqual(assemble(u0*dx), assemble(u1*dx), 10)

    def test_functional3D(self):
        """Test integration of function interpolated in non-matching meshes"""

        f = Quadratic3D()

        # Interpolate quadratic function on course mesh
        mesh0 = UnitCubeMesh(4, 4, 4)
        V0 = FunctionSpace(mesh0, "Lagrange", 2)
        u0 = Function(V0)
        u0.interpolate(f)

        if MPI.size(mesh0.mpi_comm()) == 1:
            # Interpolate FE function on finer mesh
            mesh1 = UnitCubeMesh(11, 11, 11)
            V1 = FunctionSpace(mesh1, "Lagrange", 2)
            u1 = Function(V1)
            u1.interpolate(u0)
            self.assertAlmostEqual(assemble(u0*dx), assemble(u1*dx), 10)

            mesh1 = UnitCubeMesh(10, 11, 10)
            V1 = FunctionSpace(mesh1, "Lagrange", 2)
            u1 = Function(V1)
            u1.interpolate(u0)
            self.assertAlmostEqual(assemble(u0*dx), assemble(u1*dx), 10)

if __name__ == "__main__":
    unittest.main()
