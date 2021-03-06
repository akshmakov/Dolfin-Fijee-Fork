"""Unit tests for error control"""

# Copyright (C) 2011 Marie E. Rognes
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
# First added:  2011-04-05
# Last changed: 2011-06-17

import unittest
#from unittest import skipIf # Awaiting Python 2.7
from ufl.algorithms import replace

from dolfin import *
from dolfin.fem.adaptivesolving import *

# FIXME: Move this to dolfin for user access?
def reconstruct_refined_form(form, functions, mesh):
    function_mapping = {}
    for u in functions:
        w = Function(u.leaf_node().function_space())
        w.assign(u.leaf_node())
        function_mapping[u] = w
    domain = mesh.leaf_node().ufl_domain()
    newform = replace_integral_domains(replace(form, function_mapping), domain)
    return newform, function_mapping


#@skipIf("Skipping error control test in parallel", MPI.size() > 1)
class ErrorControlTest(unittest.TestCase):

    def setUp(self):

        # Define variational problem
        mesh = UnitSquareMesh(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        bc = [DirichletBC(V, 0.0, "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS")]

        u = TrialFunction(V)
        v = TestFunction(V)
        f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=1)
        g = Expression("sin(5*x[0])", degree=1)
        a = inner(grad(u), grad(v))*dx()
        L = f*v*dx() + g*v*ds()

        # Define solution function
        u = Function(V)

        # Define LinearVariationalProblem
        problem = LinearVariationalProblem(a, L, u, bc)

        # Define goal
        M = u*dx()
        self.goal = M

        # Asserting that domains are ok before trying error control generation
        assert len(M.domains()) == 1, "Expecting only the domain from the mesh to get here through u."
        assert M.domains()[0] == mesh.ufl_domain(), "Expecting only the domain from the mesh to get here through u."
        assert len(a.domains()) == 1, "Expecting only the domain from the mesh to get here through u."
        assert a.domains()[0] == mesh.ufl_domain(), "Expecting only the domain from the mesh to get here through u."
        assert len(L.domains()) == 1, "Expecting only the domain from the mesh to get here through u."
        assert L.domains()[0] == mesh.ufl_domain(), "Expecting only the domain from the mesh to get here through u."

        # Generate ErrorControl object
        ec = generate_error_control(problem, M)

        # Store created stuff
        self.mesh = mesh
        self.problem = problem
        self.u = u
        self.ec = ec

    def test_error_estimation(self):

        if MPI.size(self.mesh.mpi_comm()) > 1:
            return

        # Solve variational problem once
        solver = LinearVariationalSolver(self.problem)
        solver.solve()

        # Compute error estimate
        error_estimate = self.ec.estimate_error(self.u, self.problem.bcs())

        # Compare estimate with defined reference
        reference = 0.0011789985750808342
        self.assertAlmostEqual(error_estimate, reference)

    def test_error_indicators(self):

        if MPI.size(self.mesh.mpi_comm()) > 1:
            return

        # Solve variational problem once
        solver = LinearVariationalSolver(self.problem)
        solver.solve()

        # Compute error indicators
        indicators = Vector(self.mesh.mpi_comm(), self.u.function_space().mesh().num_cells())
        indicators[0] = 1.0
        #self.ec.compute_indicators(indicators, self.u) #

        reference = 1.0 # FIXME
        self.assertAlmostEqual(indicators.sum(), reference)

    def test_adaptive_solve(self):

        if MPI.size(self.mesh.mpi_comm()) > 1:
            return

        # Solve problem adaptively
        solver = AdaptiveLinearVariationalSolver(self.problem, self.goal)
        tol = 0.00087
        solver.solve(tol)

        # Note: This old approach is now broken, as it doesn't change the integration domain:
        #M = replace(self.goal, {self.u: w})
        # This new approach handles the integration domain properly:
        M, fm = reconstruct_refined_form(self.goal, [self.u], self.mesh)

        # Compare computed goal with reference
        reference = 0.12583303389560166
        self.assertAlmostEqual(assemble(M), reference)

if __name__ == "__main__":
    print ""
    print "Testing automated adaptivity operations"
    print "------------------------------------------------"
    unittest.main()
