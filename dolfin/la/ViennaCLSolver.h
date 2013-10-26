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

#ifndef __VIENNACL_SOLVER_H
#define __VIENNACL_SOLVER_H

#define VIENNACL_WITH_UBLAS 1

//
// ViennaCL includes
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"

#include "ublas.h"
#include <dolfin/log/log.h>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>
#include <dolfin/parameter/Parameters.h>
#include "KrylovSolver.h"


namespace dolfin
{
  // forward declaration
  class ViennaCLPreconditioner;

  /// This class specifies the interface for ViennCL solvers
  class ViennaCLSolver : public Variable
  {
  public:
    
    /// Destructor
    virtual
      ~ViennaCLSolver(){/* Do nothing */};
    
    /// Solver
    virtual 
      ublas_vector solve( const ublas_sparse_matrix&, const ublas_vector& ) = 0;

    /// Set solver parameters
    virtual 
      void set(Parameters& ) = 0;

    /// Number of iteration
    virtual 
      std::size_t number_iteration() = 0;
  };
  //-----------------------------------------------------------------------------
 
  // 
  // ViennaCL direct solvers
  class ViennaCLSolverDirect : public ViennaCLSolver
  {
  public:
    
    /// Destructor
    virtual
      ~ViennaCLSolverDirect(){/* Do nothing */};

    /// Solver
    virtual 
      ublas_vector solve( const ublas_sparse_matrix& A, const ublas_vector& b) = 0;

    /// Set solver parameters
    virtual 
      void set(Parameters& ) = 0;

    /// Number of iteration
    virtual 
      std::size_t number_iteration() = 0;
  };
  //-----------------------------------------------------------------------------

  // 
  // ViennaCL iteratif solvers
  class ViennaCLSolverIteratif : public ViennaCLSolver
  {
  public:
    /// Create iteratif with particular preconditioner
    ViennaCLSolverIteratif( std::string preconditioner );
      
    /// Destructor
    virtual
      ~ViennaCLSolverIteratif(){/* Do nothing */};

    /// Solver
    virtual 
      ublas_vector solve( const ublas_sparse_matrix& A, const ublas_vector& b) = 0;
    
    /// Set solver parameters
    virtual 
      void set(Parameters& ) = 0;

    /// Number of iteration
    virtual 
      std::size_t number_iteration() = 0;

  protected:
    /// Underlying preconditioner strategy
    boost::shared_ptr< ViennaCLPreconditioner > _preconditioner;
  };
  //-----------------------------------------------------------------------------
  
  //
  // Conjugate Gradient (CG) 
  class ViennaCLSolverIterCG : public ViennaCLSolverIteratif
  {
  public:
    /// Create Conjugate Gradient solver
  ViennaCLSolverIterCG( std::string preconditioner ):ViennaCLSolverIteratif(preconditioner){};
    
    /// Destructor
    virtual
      ~ViennaCLSolverIterCG(){/* Do nothing */};

    /// Solver
    virtual 
      ublas_vector solve( const ublas_sparse_matrix& A, const ublas_vector& b);

    /// Set solver parameters
    virtual 
      void set(Parameters& );

    /// Number of iteration
    virtual 
      std::size_t number_iteration(){ return _custom_tag->iters();};

    /// Returun solver tag
    viennacl::linalg::cg_tag& get_tag() const {return *_custom_tag;};

  private:
    /// Solver tag
    boost::shared_ptr< viennacl::linalg::cg_tag > _custom_tag;

  };
  //-----------------------------------------------------------------------------

  //
  // Stabilized Bi-CG (BiCGStab) 
  class ViennaCLSolverIterBiCGStab : public ViennaCLSolverIteratif
  {
  public:
    /// Create Stabilized Bi-CG solver
  ViennaCLSolverIterBiCGStab( std::string preconditioner ):ViennaCLSolverIteratif(preconditioner){};
    
    /// Destructor
    virtual
      ~ViennaCLSolverIterBiCGStab(){/* Do nothing */};

    /// Solver
    virtual 
      ublas_vector solve( const ublas_sparse_matrix& A, const ublas_vector& b);

    /// Set solver parameters
    virtual 
      void set(Parameters& );

    /// Number of iteration
    virtual 
      std::size_t number_iteration(){ return _custom_tag->iters();};

    /// Returun solver tag
    viennacl::linalg::bicgstab_tag& get_tag() const {return *_custom_tag;};

  private:
    /// Solver tag
    boost::shared_ptr< viennacl::linalg::bicgstab_tag > _custom_tag;
  };
  //-----------------------------------------------------------------------------
 
  //
  //  Generalized Minimum Residual (GMRES) 
  class ViennaCLSolverIterGMRES : public ViennaCLSolverIteratif
  {
  public:
    /// Create Generalized Minimum Residual solver
  ViennaCLSolverIterGMRES( std::string preconditioner ):ViennaCLSolverIteratif(preconditioner){};
    
    /// Destructor
    virtual
      ~ViennaCLSolverIterGMRES(){/* Do nothing */};

    /// Solver
    virtual 
      ublas_vector solve( const ublas_sparse_matrix& A, const ublas_vector& b);

    /// Set solver parameters
    virtual 
      void set(Parameters& );

    /// Number of iteration
    virtual 
      std::size_t number_iteration(){ return _custom_tag->iters();};

    /// Returun solver tag
    viennacl::linalg::gmres_tag& get_tag() const {return *_custom_tag;};

  private:
    /// Solver tag
    boost::shared_ptr< viennacl::linalg::gmres_tag > _custom_tag;
  };
}
//-----------------------------------------------------------------------------

#endif
