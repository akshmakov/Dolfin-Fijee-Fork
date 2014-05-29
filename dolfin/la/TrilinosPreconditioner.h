// Copyright (C) 2010 Garth N. Wells
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
#ifndef __DOLFIN_TRILINOS_PRECONDITIONER_H
#define __DOLFIN_TRILINOS_PRECONDITIONER_H

#ifdef HAS_TRILINOS

#include <string>
#include <vector>
#include <memory>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/parameter/Parameters.h>
#include "GenericPreconditioner.h"

// Trilinos forward declarations
class Epetra_MultiVector;
class Epetra_RowMatrix;
class Epetra_Operator;
class Ifpack_Preconditioner;
namespace Belos
{
  template<class ScalarType, class MV, class OP>
  class LinearProblem;
}

namespace ML_Epetra
{
  class MultiLevelPreconditioner;
}

namespace Teuchos
{
  class ParameterList;
}

// some typdefs for Belos
typedef double BelosScalarType;
typedef Epetra_MultiVector BelosMultiVector;
typedef Epetra_Operator BelosOperator;
typedef Belos::LinearProblem<BelosScalarType, BelosMultiVector, BelosOperator>
  BelosLinearProblem;

namespace dolfin
{

  // Forward declarations
  class EpetraKrylovSolver;
  class EpetraMatrix;
  class GenericVector;
  class VectorSpaceBasis;

  /// This class is a wrapper for configuring Epetra
  /// preconditioners. It does not own a preconditioner. It can take a
  /// EpetraKrylovSolver and set the preconditioner type and
  /// parameters.

  class TrilinosPreconditioner : public GenericPreconditioner, public Variable
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    explicit TrilinosPreconditioner(std::string method="default");

    /// Destructor
    virtual ~TrilinosPreconditioner();

    /// Set the precondtioner and matrix used in preconditioner
    virtual void set(BelosLinearProblem& problem,
                     const EpetraMatrix& P
                     );

    /// Set the Trilonos preconditioner parameters list
    void set_parameters(std::shared_ptr<const Teuchos::ParameterList> list);

    /// Set the Trilonos preconditioner parameters list (for use from
    /// Python)
    void set_parameters(Teuchos::RCP<Teuchos::ParameterList> list);

    /// Set basis for the null space of the operator. Setting this is
    /// critical to the performance of some preconditioners, e.g. ML.
    /// The vectors spanning the null space are copied.
    void set_nullspace(const VectorSpaceBasis& null_space);

    /// Set the coordinates of the operator (matrix) rows and geometric
    /// dimension d. This is can be used by required for certain
    /// preconditioners, e.g. ML. The input for this function can be
    /// generated using GenericDofMap::tabulate_all_dofs.
    virtual void set_coordinates(const std::vector<double>& x, std::size_t dim)
    {
      dolfin_error("TrilinosPreconditioner.h",
                   "set coordinates for preconditioner operator",
                   "Not supported by current preconditioner type");
    }

    /// Return preconditioner name
    std::string name() const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return a list of available preconditioners
    static std::vector<std::pair<std::string, std::string> > preconditioners();

    /// Default parameter values
    static Parameters default_parameters();


  private:

    /// Setup the ML precondtioner
    void set_ml(BelosLinearProblem& problem,
                const Epetra_RowMatrix& P);

    /// Named preconditioner
    std::string _preconditioner;

    // Available named preconditioners
    static const std::map<std::string, int> _preconditioners;

    // Available named preconditionersdescriptions
    static const std::vector<std::pair<std::string, std::string> >
      _preconditioners_descr;

    // The Preconditioner
    std::shared_ptr<Ifpack_Preconditioner> _ifpack_preconditioner;
    std::shared_ptr<ML_Epetra::MultiLevelPreconditioner> _ml_preconditioner;

    // Parameter list
    std::shared_ptr<const Teuchos::ParameterList> parameter_list;

    // Vectors spanning the null space
    std::shared_ptr<Epetra_MultiVector> _nullspace;

    // Teuchos::ParameterList pointer, used when initialized with a
    // Teuchos::RCP shared_ptr
    Teuchos::RCP<const Teuchos::ParameterList> parameter_ref_keeper;

  };

}

#endif

#endif
