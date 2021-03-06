// Copyright (C) 2005-2008 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2006-2011
// Modified by Anders E. Johansen 2011
//
// First added:  2005-10-23
// Last changed: 2013-11-20

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <utility>
#include <memory>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  // Forward declarations
  class GenericLinearSolver;
  class GenericLinearAlgebraFactory;
  class GenericMatrix;
  class GenericVector;
  class NonlinearProblem;

  /// This class defines a Newton solver for nonlinear systems of
  /// equations of the form :math:`F(x) = 0`.

  class NewtonSolver : public Variable
  {
  public:

    /// Create nonlinear solver
    NewtonSolver();

    /// Create nonlinear solver using provided linear solver
    ///
    /// *Arguments*
    ///     solver (_GenericLinearSolver_)
    ///         The linear solver.
    ///     factory (_GenericLinearAlgebraFactory_)
    ///         The factory.
    NewtonSolver(std::shared_ptr<GenericLinearSolver> solver,
                 GenericLinearAlgebraFactory& factory);

    /// Destructor
    virtual ~NewtonSolver();

    /// Solve abstract nonlinear problem :math:`F(x) = 0` for given
    /// :math:`F` and Jacobian :math:`\dfrac{\partial F}{\partial x}`.
    ///
    /// *Arguments*
    ///     nonlinear_function (_NonlinearProblem_)
    ///         The nonlinear problem.
    ///     x (_GenericVector_)
    ///         The vector.
    ///
    /// *Returns*
    ///     std::pair<std::size_t, bool>
    ///         Pair of number of Newton iterations, and whether
    ///         iteration converged)
    std::pair<std::size_t, bool> solve(NonlinearProblem& nonlinear_function,
                                       GenericVector& x);

    /// Return Newton iteration number
    ///
    /// *Returns*
    ///     std::size_t
    ///         The iteration number.
    std::size_t iteration() const;

    /// Return current residual
    ///
    /// *Returns*
    ///     double
    ///         Current residual.
    double residual() const;

    /// Return current relative residual
    ///
    /// *Returns*
    ///     double
    ///       Current relative residual.
    double relative_residual() const;

    /// Return the linear solver
    ///
    /// *Returns*
    ///     _GenericLinearSolver_
    ///         The linear solver.
    GenericLinearSolver& linear_solver() const;

    /// Default parameter values
    ///
    /// *Returns*
    ///     _Parameters_
    ///         Parameter values.
    static Parameters default_parameters();

  private:

    // Convergence test
    virtual bool converged(const GenericVector& r,
                           const NonlinearProblem& nonlinear_problem,
                           std::size_t iteration);

    // Current number of Newton iterations
    std::size_t _newton_iteration;

    // Most recent residual and intitial residual
    double _residual, _residual0;

    // Solver
    std::shared_ptr<GenericLinearSolver> _solver;

    // Jacobian matrix
    std::shared_ptr<GenericMatrix> _matA;

    // Solution vector
    std::shared_ptr<GenericVector> _dx;

    // Resdiual vector
    std::shared_ptr<GenericVector> _b;

    // MPI communicator
    MPI_Comm _mpi_comm;

  };

}

#endif
