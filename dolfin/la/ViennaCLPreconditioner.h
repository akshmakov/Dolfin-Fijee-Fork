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
// Modified by Cobigo Logg 2013-20xx
//
// First added:  2013-09-11
// Last changed: 

#ifndef __DOLFIN_VIENNACL_PRECONDITIONER_H
#define __DOLFIN_VIENNACL_PRECONDITIONER_H

#ifdef HAS_VIENNACL

#define VIENNACL_WITH_UBLAS 1

//
// ViennaCL includes
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/jacobi_precond.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"

#include "ublas.h"
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/parameter/Parameters.h>
#include "GenericPreconditioner.h"
#include "ViennaCLSolver.h"
#include <dolfin/log/log.h>


namespace dolfin
{

  //-----------------------------------------------------------------------------
  /// This class specifies the interface for ViennCL preconditioner
  class ViennaCLPreconditioner : public GenericPreconditioner, public Variable
  {
  public:

    /// Create general preconditioning parameters object
    explicit ViennaCLPreconditioner( std::string );

    /// Solver
    virtual 
      ublas_vector solve( const ublas_sparse_matrix& , const ublas_vector& , 
			  const ViennaCLSolverIterCG&) = 0;
    /// Solver
    virtual 
      ublas_vector solve( const ublas_sparse_matrix& , const ublas_vector&, 
			  const ViennaCLSolverIterBiCGStab&) = 0;
    /// Solver
    virtual 
      ublas_vector solve( const ublas_sparse_matrix& , const ublas_vector& , 
			  const ViennaCLSolverIterGMRES&) = 0;
    
    /// Set the (approximate) null space of the preconditioner operator
    /// (matrix). This is required for certain preconditioner types,
    /// e.g. smoothed aggregation multigrid
    virtual void set_nullspace(const VectorSpaceBasis& ) = 0;

    /// Set the coordinates of the operator (matrix) rows and geometric
    /// dimension d. This is can be used by required for certain
    /// preconditioners, e.g. ML. The input for this function can be
    /// generated using GenericDofMap::tabulate_all_dofs.
    virtual void set_coordinates(const std::vector<double>&, std::size_t) = 0;

    /// Rerturn a list of available preconditioners
    static std::vector<std::pair<std::string, std::string> > preconditioners();

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Default parameter values
    static Parameters default_parameters();

  private:

    /// Available names preconditioners
    static const std::map<std::string, int> _methods;

    /// Available preconditioner descriptions
    static const std::vector<std::pair<std::string, std::string> >
      _methods_descr;
 };
  //-----------------------------------------------------------------------------
  
  //
  // No preconditioning
  class ViennaCLPreconditionerNoPreconditioning : public ViennaCLPreconditioner
  {
  public:

    /// Create a no preconditioning object
    explicit ViennaCLPreconditionerNoPreconditioning( std::string type ) : ViennaCLPreconditioner(type) 
    {};
    
    /// Set the (approximate) null space of the preconditioner operator
    /// (matrix). This is required for certain preconditioner types,
    /// e.g. smoothed aggregation multigrid
    virtual void set_nullspace(const VectorSpaceBasis& )
    {
      dolfin_error("ViennaCLPreconditioner.h",
                   "set nullspace for preconditioner operator",
                   "Not supported by current preconditioner type");
    };

    /// Set the coordinates of the operator (matrix) rows and geometric
    /// dimension d. This is can be used by required for certain
    /// preconditioners, e.g. ML. The input for this function can be
    /// generated using GenericDofMap::tabulate_all_dofs.
    virtual void set_coordinates(const std::vector<double>&, std::size_t)
    {
      dolfin_error("ViennaPreconditioner.h",
		   "set coordinates for preconditioner operator",
		   "Not supported by current preconditioner type");
    };

    /// Solver
    virtual ublas_vector solve( const ublas_sparse_matrix& A, 
				const ublas_vector& b, 
				const ViennaCLSolverIterCG& solver);
 
    /// Solver
    virtual  ublas_vector solve( const ublas_sparse_matrix& A, 
				 const ublas_vector& b, 
				 const ViennaCLSolverIterBiCGStab& solver);

    /// Solver
    virtual ublas_vector solve( const ublas_sparse_matrix& A, 
				const ublas_vector& b, 
				const ViennaCLSolverIterGMRES& solver);
  };
  //-----------------------------------------------------------------------------

  //
  // Incomplete LU Factorization with Threshold (ILUT)
  class ViennaCLPreconditionerILUT : public ViennaCLPreconditioner
  {
  public:

    /// Create an incomplete LU Factorization with Threshold (ILUT) preconditioning object
    explicit ViennaCLPreconditionerILUT( std::string type ) : ViennaCLPreconditioner(type) 
    {};

    /// Set the (approximate) null space of the preconditioner operator
    /// (matrix). This is required for certain preconditioner types,
    /// e.g. smoothed aggregation multigrid
    virtual void set_nullspace(const VectorSpaceBasis& )
    {
      dolfin_error("ViennaCLPreconditioner.h",
                   "set nullspace for preconditioner operator",
                   "Not supported by current preconditioner type");
    };

    /// Set the coordinates of the operator (matrix) rows and geometric
    /// dimension d. This is can be used by required for certain
    /// preconditioners, e.g. ML. The input for this function can be
    /// generated using GenericDofMap::tabulate_all_dofs.
    virtual void set_coordinates(const std::vector<double>&, std::size_t)
    {
      dolfin_error("ViennaPreconditioner.h",
		   "set coordinates for preconditioner operator",
		   "Not supported by current preconditioner type");
    };

    /// Solver
    virtual 
      ublas_vector solve( const ublas_sparse_matrix& A, const ublas_vector& b, 
			  const ViennaCLSolverIterCG& solver);

    /// Solver
    virtual 
      ublas_vector solve( const ublas_sparse_matrix& A, const ublas_vector& b, 
			  const ViennaCLSolverIterBiCGStab& solver);
 
    /// Solver
    virtual 
      ublas_vector solve( const ublas_sparse_matrix& A, const ublas_vector& b, 
			  const ViennaCLSolverIterGMRES& solver);
  };
  //-----------------------------------------------------------------------------
 
  //
  // Incomplete LU Factorization with Static Pattern (ILU0)
  class ViennaCLPreconditionerILU0 : public ViennaCLPreconditioner
  {
  public:

    /// Create an incomplete LU Factorization with Static Pattern (ILU0) preconditioning object
    explicit ViennaCLPreconditionerILU0( std::string type ) : ViennaCLPreconditioner(type) 
    {};

    /// Set the (approximate) null space of the preconditioner operator
    /// (matrix). This is required for certain preconditioner types,
    /// e.g. smoothed aggregation multigrid
    virtual void set_nullspace(const VectorSpaceBasis& )
    {
      dolfin_error("ViennaCLPreconditioner.h",
                   "set nullspace for preconditioner operator",
                   "Not supported by current preconditioner type");
    };

    /// Set the coordinates of the operator (matrix) rows and geometric
    /// dimension d. This is can be used by required for certain
    /// preconditioners, e.g. ML. The input for this function can be
    /// generated using GenericDofMap::tabulate_all_dofs.
    virtual void set_coordinates(const std::vector<double>&, std::size_t)
    {
      dolfin_error("ViennaPreconditioner.h",
		   "set coordinates for preconditioner operator",
		   "Not supported by current preconditioner type");
    };

    /// Solver
    virtual ublas_vector solve( const ublas_sparse_matrix& A, 
				const ublas_vector& b, 
				const ViennaCLSolverIterCG& solver);
 
    /// Solver
    virtual ublas_vector solve( const ublas_sparse_matrix& A, 
				const ublas_vector& b, 
				const ViennaCLSolverIterBiCGStab& solver);

    /// Solver
    virtual ublas_vector solve( const ublas_sparse_matrix& A, 
				const ublas_vector& b, 
				const ViennaCLSolverIterGMRES& solver);
  };
  //-----------------------------------------------------------------------------
 
  //
  // To overcome the serial nature of ILUT and ILU0 applied to the full system matrix, 
  // a parallel variant is to apply ILU to diagonal blocks of the system matrix. 
  class ViennaCLPreconditionerBlockILU : public ViennaCLPreconditioner
  {
  public:

    /// Create a ILU to diagonal blocks of the system matrix preconditioning object
    explicit ViennaCLPreconditionerBlockILU( std::string type ) : ViennaCLPreconditioner(type)
    {};

    /// Set the (approximate) null space of the preconditioner operator
    /// (matrix). This is required for certain preconditioner types,
    /// e.g. smoothed aggregation multigrid
    virtual void set_nullspace(const VectorSpaceBasis& )
    {
      dolfin_error("ViennaCLPreconditioner.h",
                   "set nullspace for preconditioner operator",
                   "Not supported by current preconditioner type");
    };

    /// Set the coordinates of the operator (matrix) rows and geometric
    /// dimension d. This is can be used by required for certain
    /// preconditioners, e.g. ML. The input for this function can be
    /// generated using GenericDofMap::tabulate_all_dofs.
    virtual void set_coordinates(const std::vector<double>&, std::size_t)
    {
      dolfin_error("ViennaPreconditioner.h",
		   "set coordinates for preconditioner operator",
		   "Not supported by current preconditioner type");
    };

    /// Solver
    virtual ublas_vector solve( const ublas_sparse_matrix& A, 
				const ublas_vector& b, 
				const ViennaCLSolverIterCG& solver);

    /// Solver
    virtual ublas_vector solve( const ublas_sparse_matrix& A, 
				const ublas_vector& b, 
				const ViennaCLSolverIterBiCGStab& solver);

    /// Solver
    virtual ublas_vector solve( const ublas_sparse_matrix& A, 
				const ublas_vector& b, 
				const ViennaCLSolverIterGMRES& solver);
  };
  //-----------------------------------------------------------------------------
 
  //
  // Jacobi preconditioner is a simple diagonal preconditioner given by the 
  // reciprocals of the diagonal entries of the system matrix A
  class ViennaCLPreconditionerJacobi : public ViennaCLPreconditioner
  {
  public:

    /// Create a Jacobi preconditioner object
    explicit ViennaCLPreconditionerJacobi( std::string type ) : ViennaCLPreconditioner(type)
    {};

    /// Set the (approximate) null space of the preconditioner operator
    /// (matrix). This is required for certain preconditioner types,
    /// e.g. smoothed aggregation multigrid
    virtual void set_nullspace(const VectorSpaceBasis& )
    {
      dolfin_error("ViennaCLPreconditioner.h",
                   "set nullspace for preconditioner operator",
                   "Not supported by current preconditioner type");
    };

    /// Set the coordinates of the operator (matrix) rows and geometric
    /// dimension d. This is can be used by required for certain
    /// preconditioners, e.g. ML. The input for this function can be
    /// generated using GenericDofMap::tabulate_all_dofs.
    virtual void set_coordinates(const std::vector<double>&, std::size_t)
    {
      dolfin_error("ViennaPreconditioner.h",
		   "set coordinates for preconditioner operator",
		   "Not supported by current preconditioner type");
    };

    /// Solver
    virtual ublas_vector solve( const ublas_sparse_matrix& A, 
				const ublas_vector& b, 
				const ViennaCLSolverIterCG& solver);

    /// Solver
    virtual ublas_vector solve( const ublas_sparse_matrix& A, 
				const ublas_vector& b, 
				const ViennaCLSolverIterBiCGStab& solver);

    /// Solver
    virtual ublas_vector solve( const ublas_sparse_matrix& A, 
				const ublas_vector& b, 
				const ViennaCLSolverIterGMRES& solver);
  };
}

#endif

#endif
