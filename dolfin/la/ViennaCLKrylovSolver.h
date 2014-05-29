// Copyright (C) 2006-2009 Garth N. Wells
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
// Modified by Cobigo Logg 2013-
//
// First added:  2013-09-09
// Last changed: 

#ifndef __VIENNACL_KRYLOV_SOLVER_H
#define __VIENNACL_KRYLOV_SOLVER_H

#include <set>
#include <string>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include "ublas.h"
#include "GenericLinearSolver.h"
#include "uBLASLinearOperator.h"
#include "uBLASMatrix.h"
#include "uBLASVector.h"
#include "ViennaCLSolver.h"
#include "ViennaCLPreconditioner.h"

namespace dolfin
{
  // forward declaration
  class GenericLinearOperator;
  class GenericVector;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b using ViennaCL.

  class ViennaCLKrylovSolver : public GenericLinearSolver
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    ViennaCLKrylovSolver(std::string method="default",
			 std::string preconditioner="default");

//    /// Create Krylov solver for a particular method and ViennaCLPreconditioner
//    ViennaCLKrylovSolver(std::string method,
//			 ViennaCLPreconditioner& pc);

    /// Destructor
    ~ViennaCLKrylovSolver();

    /// Solve the operator (matrix)
    virtual void set_operator( std::shared_ptr<const GenericLinearOperator> A)
    { set_operators(A, A); }

    /// Set operator (matrix) and preconditioner matrix
    virtual void set_operators( std::shared_ptr<const GenericLinearOperator> A,
			        std::shared_ptr<const GenericLinearOperator> P)
    { _A = A; }

    /// Set null space of the operator (matrix). This is used to solve
    /// singular systems
    virtual void set_nullspace(const VectorSpaceBasis& nullspace)
    {
      dolfin_error("ViennaCLKrylovSolver.h",
                   "set nullspace for operator",
                   "Not supported by current linear algebra solver backend");
    }

    /// Return the operator (matrix)
    const GenericLinearOperator& get_operator() const
    {
      if (!_A)
	{
	  dolfin_error("ViennaCLKrylovSolver.cpp",
		       "access operator for ViennaCL Krylov solver",
		       "Operator has not been set");
	}
      return *_A;
    }

    /// Solve linear system Ax = b and return number of iterations
    virtual std::size_t solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    virtual std::size_t solve(const GenericLinearOperator& A, GenericVector& x,
			      const GenericVector& b);


    /// Solve linear system A^Tx = b
    virtual std::size_t solve_transpose(const GenericLinearOperator& A,
                                        GenericVector& x,
                                        const GenericVector& b)
    {
      dolfin_error("ViennaCLKrylovSolver.h",
                   "solve linear system transpose",
                   "Not supported by current linear algebra backend. Consider using solve_transpose(x, b)");
      return 0;
    }

    /// Solve linear system A^Tx = b
    virtual std::size_t solve_transpose(GenericVector& x,
                                        const GenericVector& b)
    {
      dolfin_error("ViennaCLKrylovSolver.h",
                   "solve linear system transpose",
                   "Not supported by current linear algebra backend. Consider using solve_transpose(x, b)");
      return 0;
    }

    // FIXME: This should not be needed. Need to cleanup linear solver
    // name jungle: default, lu, iterative, direct, krylov, etc
    /// Return parameter type: "krylov_solver" or "lu_solver"
    virtual std::string parameter_type() const
    {
      return "default";
    }

    /// Return a list of available solver methods
    static std::vector<std::pair<std::string, std::string> > methods();

    /// Return a list of available preconditioners
    static std::vector<std::pair<std::string, std::string> > preconditioners();

    /// Default parameter values
    static Parameters default_parameters();

  private:

    /// Select and create named preconditioner
    void select_preconditioner(std::string preconditioner);

    /// Read solver parameters
    void read_parameters();

    /// Solver type
    std::string _method;

    // Available solvers
    static const std::map<std::string, int> _methods;

    // Available solvers descriptions
    static const std::vector<std::pair<std::string, std::string> >
      _methods_descr;

    /// Solver parameters
    double rtol, atol, div_tol;
    std::size_t max_it, restart;
    bool report;

    /// Operator (the matrix)
    std::shared_ptr<const GenericLinearOperator> _A;

   // Underlying solver strategy
    std::shared_ptr< ViennaCLSolver > _solver;
  };
}

#endif
