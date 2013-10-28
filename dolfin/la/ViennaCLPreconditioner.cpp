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
// Modified by Cobigo Logg 2013-
// 
// First added:  2013-09-09
// Last changed: 

#ifdef HAS_VIENNACL

#include <boost/assign/list_of.hpp>
#include <boost/lexical_cast.hpp>


#include "KrylovSolver.h"
#include <dolfin/parameter/Parameters.h>
#include "ViennaCLPreconditioner.h"


namespace dolfin
{

  // Mapping from preconditioner string to ViennaCL
  const std::map<std::string, int> ViennaCLPreconditioner::_methods
  = boost::assign::map_list_of
    ("default",     0)
    ("none",        1)
    ("ilut",        2)
    ("ilu0",        3)
    ("block_ilu",   4)
    ("jacobi",      5)
    /*  ("row_scaling", 6)*/;

  // Mapping from preconditioner string to description string
  const std::vector<std::pair<std::string, std::string> >
  ViennaCLPreconditioner::_methods_descr
  = boost::assign::pair_list_of
    ("default",          "default preconditioner")
    ("none",             "No preconditioner")
    ("ilut",             "Incomplete LU factorization with Threshold")
    ("ilu0",             "Incomplete LU factorization with Static Pattern")
    ("block_ilu",        "Incomplete LU factorization with Static Pattern parallel variant")
    ("jacobi",           "Jacobi preconditioner")
    /* ("row_scaling",      "Simple diagonal preconditioner given by the reciprocals of the norms of the rows of the system matrix")*/;

  //-----------------------------------------------------------------------------
  std::vector<std::pair<std::string, std::string> >
  ViennaCLPreconditioner::preconditioners()
  {
    return ViennaCLPreconditioner::_methods_descr;
  }

  //-----------------------------------------------------------------------------
  ViennaCLPreconditioner::ViennaCLPreconditioner( std::string type )
  {
    // Set parameter values
    parameters = default_parameters();
    
    // Check that the requested method is known
    if (_methods.count(type) == 0)
      {
	dolfin_error("ViennaCLPreconditioner.cpp",
		     "create ViennaCL preconditioner",
		     "Unknown preconditioner type (\"%s\")", type.c_str());
      }
}

//-----------------------------------------------------------------------------
  Parameters ViennaCLPreconditioner::default_parameters()
  {
    Parameters p(KrylovSolver::default_parameters()("preconditioner"));
    p.rename("viennacl_preconditioner");
    
    return p;
 }
  //-----------------------------------------------------------------------------
  std::string ViennaCLPreconditioner::str(bool verbose) const
  {
    std::stringstream s;
    if (verbose)
      warning("Verbose output for ViennaCLPreconditioner not implemented.");
    else
      s << "<ViennaCLPreconditioner>";
    
    return s.str();
  }
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerNoPreconditioning::solve(const ublas_sparse_matrix& A, 
						 const ublas_vector& b, 
						 const ViennaCLSolverIterCG& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag());
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag());
#endif      
    
    //
    return X; 
  }
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerNoPreconditioning::solve(const ublas_sparse_matrix& A, 
						 const ublas_vector& b, 
						 const ViennaCLSolverIterBiCGStab& solver)
  {
     //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag());
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag());
#endif      
    
    //
    return X; 
  }
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerNoPreconditioning::solve(const ublas_sparse_matrix& A, 
						 const ublas_vector& b, 
						 const ViennaCLSolverIterGMRES& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag());
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag());
#endif      
    
    //
    return X; 
  }
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerILUT::solve( const ublas_sparse_matrix& A, 
				     const ublas_vector& b, 
				     const ViennaCLSolverIterCG& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::linalg::ilut_precond<  viennacl::compressed_matrix<double>  > 
      precond(vcl_A, viennacl::linalg::ilut_tag());
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag(), precond);
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    viennacl::linalg::ilut_precond< ublas_sparse_matrix > precond(A, viennacl::linalg::ilut_tag());
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag(), precond);
#endif      
    
    //
    return X; 
  };
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerILUT::solve( const ublas_sparse_matrix& A, 
				     const ublas_vector& b, 
				     const ViennaCLSolverIterBiCGStab& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::linalg::ilut_precond<  viennacl::compressed_matrix<double>  > 
      precond(vcl_A, viennacl::linalg::ilut_tag());
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag(), precond);
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    viennacl::linalg::ilut_precond< ublas_sparse_matrix > precond(A, viennacl::linalg::ilut_tag());
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag(), precond);
#endif      
    
    //
    return X; 
  };
  //-----------------------------------------------------------------------------
 ublas_vector
  ViennaCLPreconditionerILUT::solve( const ublas_sparse_matrix& A, 
				     const ublas_vector& b, 
				     const ViennaCLSolverIterGMRES& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::linalg::ilut_precond<  viennacl::compressed_matrix<double>  > 
      precond(vcl_A, viennacl::linalg::ilut_tag());
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag(), precond);
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    viennacl::linalg::ilut_precond< ublas_sparse_matrix > precond(A, viennacl::linalg::ilut_tag());
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag(), precond);
#endif      
    
    //
    return X; 
  };
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerILU0::solve(const ublas_sparse_matrix& A, 
				    const ublas_vector& b, 
				    const ViennaCLSolverIterCG& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::linalg::ilu0_precond< viennacl::compressed_matrix<double> > 
      precond(vcl_A, viennacl::linalg::ilu0_tag());
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag(), precond);
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    viennacl::linalg::ilu0_precond< ublas_sparse_matrix > precond(A, viennacl::linalg::ilu0_tag());
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag(), precond);
#endif      
    
    //
    return X; 
  };
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerILU0::solve(const ublas_sparse_matrix& A, 
				    const ublas_vector& b, 
				    const ViennaCLSolverIterBiCGStab& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::linalg::ilu0_precond< viennacl::compressed_matrix<double> > 
      precond(vcl_A, viennacl::linalg::ilu0_tag());
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag(), precond);
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    viennacl::linalg::ilu0_precond< ublas_sparse_matrix > precond(A, viennacl::linalg::ilu0_tag());
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag(), precond);
#endif      
    
    //
    return X; 
  };
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerILU0::solve(const ublas_sparse_matrix& A, 
				    const ublas_vector& b, 
				    const ViennaCLSolverIterGMRES& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::linalg::ilu0_precond< viennacl::compressed_matrix<double> > 
      precond(vcl_A, viennacl::linalg::ilu0_tag());
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag(), precond);
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    viennacl::linalg::ilu0_precond< ublas_sparse_matrix > precond(A, viennacl::linalg::ilu0_tag());
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag(), precond);
#endif      
    
    //
    return X; 
  };
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerBlockILU::solve(const ublas_sparse_matrix& A, 
					const ublas_vector& b, 
					const ViennaCLSolverIterCG& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::linalg::block_ilu_precond< viennacl::compressed_matrix<double>, viennacl::linalg::ilu0_tag> 
      precond(vcl_A, viennacl::linalg::ilu0_tag());
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag(), precond);
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    viennacl::linalg::block_ilu_precond< ublas_sparse_matrix, viennacl::linalg::ilu0_tag> 
      precond(A, viennacl::linalg::ilu0_tag());
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag(), precond);
#endif      
    
    //
    return X; 
  };
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerBlockILU::solve(const ublas_sparse_matrix& A, 
					const ublas_vector& b, 
					const ViennaCLSolverIterBiCGStab& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::linalg::block_ilu_precond< viennacl::compressed_matrix<double>, viennacl::linalg::ilu0_tag> 
      precond(vcl_A, viennacl::linalg::ilu0_tag());
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag(), precond);
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    viennacl::linalg::block_ilu_precond< ublas_sparse_matrix, viennacl::linalg::ilu0_tag> 
      precond(A, viennacl::linalg::ilu0_tag());
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag(), precond);
#endif      
    
    //
    return X; 
  };
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerBlockILU::solve(const ublas_sparse_matrix& A, 
					const ublas_vector& b, 
					const ViennaCLSolverIterGMRES& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::linalg::block_ilu_precond< viennacl::compressed_matrix<double>, viennacl::linalg::ilu0_tag> 
      precond(vcl_A, viennacl::linalg::ilu0_tag());
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag(), precond);
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    viennacl::linalg::block_ilu_precond< ublas_sparse_matrix, viennacl::linalg::ilu0_tag> 
      precond(A, viennacl::linalg::ilu0_tag());
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag(), precond);
#endif      
    
    //
    return X; 
  };
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerJacobi::solve(const ublas_sparse_matrix& A, 
				      const ublas_vector& b, 
				      const ViennaCLSolverIterCG& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<double> > 
      precond(vcl_A, viennacl::linalg::jacobi_tag());
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag(), precond);
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    viennacl::linalg::jacobi_precond< ublas_sparse_matrix > precond(A, viennacl::linalg::jacobi_tag());
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag(), precond);
#endif      
    
    //
    return X; 
  };
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerJacobi::solve(const ublas_sparse_matrix& A, 
				      const ublas_vector& b, 
				      const ViennaCLSolverIterBiCGStab& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<double> > 
      precond(vcl_A, viennacl::linalg::jacobi_tag());
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag(), precond);
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    viennacl::linalg::jacobi_precond< ublas_sparse_matrix > precond(A, viennacl::linalg::jacobi_tag());
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag(), precond);
#endif      
    
    //
    return X; 
  };
  //-----------------------------------------------------------------------------
  ublas_vector
  ViennaCLPreconditionerJacobi::solve(const ublas_sparse_matrix& A, 
				      const ublas_vector& b, 
				      const ViennaCLSolverIterGMRES& solver)
  {
    //
    // Solve AX = b
    
    // solution
    ublas_vector X;
    //
    
#ifdef VIENNACL_WITH_OPENCL
    // Move data on OpenCL devices
    viennacl::vector<double> vcl_b( b.size() );
    viennacl::compressed_matrix<double> vcl_A( A.size1(),
					       A.size2() );
    //
    viennacl::copy(A, vcl_A);
    viennacl::copy(b, vcl_b);
    
    //
    viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<double> > 
      precond(vcl_A, viennacl::linalg::jacobi_tag());
    //
    viennacl::vector<double> vcl_X = viennacl::linalg::solve(vcl_A, vcl_b, solver.get_tag(), precond);
    
    // Move data back to the host
    X.resize( vcl_b.size() );
    viennacl::copy(vcl_X, X);
#else
    //
    viennacl::linalg::jacobi_precond< ublas_sparse_matrix > precond(A, viennacl::linalg::jacobi_tag());
    //
    X = viennacl::linalg::solve(A, b, solver.get_tag(), precond);
#endif      
    
    //
    return X; 
  };
  //-----------------------------------------------------------------------------
}
#endif
