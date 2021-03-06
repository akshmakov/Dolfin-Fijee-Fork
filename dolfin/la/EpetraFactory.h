// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring
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
// Modified by Anders Logg 2011-2012
//
// First added:  2008-04-21
// Last changed: 2012-08-20

#ifdef HAS_TRILINOS

#ifndef __EPETRA_FACTORY_H
#define __EPETRA_FACTORY_H

#include <string>
#include <memory>
#include "EpetraKrylovSolver.h"
#include "EpetraLUSolver.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "TensorLayout.h"
#include "GenericLinearAlgebraFactory.h"

namespace dolfin
{

  class GenericLinearSolver;

  class EpetraFactory : public GenericLinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~EpetraFactory();

    /// --- GenericLinearAlgebraFactory interface

    /// Create empty matrix
    std::shared_ptr<GenericMatrix> create_matrix() const;

    /// Create empty vector
    std::shared_ptr<GenericVector> create_vector() const;

    /// Create empty tensor layout
    std::shared_ptr<TensorLayout> create_layout(std::size_t rank) const;

    /// Create empty linear operator
    std::shared_ptr<GenericLinearOperator> create_linear_operator() const;

    /// Create LU solver
    std::shared_ptr<GenericLUSolver>
      create_lu_solver(std::string method) const;

    /// Create Krylov solver
    std::shared_ptr<GenericLinearSolver>
      create_krylov_solver(std::string method, std::string preconditioner) const;

    /// Return a list of available LU solver methods
    std::vector<std::pair<std::string, std::string> >
    lu_solver_methods() const;

    /// Return a list of available Krylov solver methods
    std::vector<std::pair<std::string, std::string> >
      krylov_solver_methods() const;

    /// Return a list of available preconditioners
    std::vector<std::pair<std::string, std::string> >
      krylov_solver_preconditioners() const;

    /// --- EpetraFactory interface

    // Return singleton instance
    static EpetraFactory& instance()
    { return factory; }

  private:

    // Private constructor
    EpetraFactory();

    // Singleton instance
    static EpetraFactory factory;

  };

}

#endif

#endif
