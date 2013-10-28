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

#include <boost/assign/list_of.hpp>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/LogStream.h>
#include "ViennaCLKrylovSolver.h"
#include "KrylovSolver.h"

using namespace dolfin;


// Mapping from method string to PETSc
const std::map<std::string, int > ViennaCLKrylovSolver::_methods
= boost::assign::map_list_of
  ("default",  0)
  ("cg",       1)
  ("gmres",    2)
  ("bicgstab", 3);

// Mapping from method string to description
const std::vector<std::pair<std::string, std::string> >
ViennaCLKrylovSolver::_methods_descr = boost::assign::pair_list_of
  ("default",    "default Krylov method")
  ("cg",         "Conjugate gradient method")
  ("gmres",      "Generalized minimal residual method")
  ("bicgstab",   "Biconjugate gradient stabilized method");

//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
ViennaCLKrylovSolver::methods()
{
  return ViennaCLKrylovSolver::_methods_descr;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
ViennaCLKrylovSolver::preconditioners()
{
  return ViennaCLPreconditioner::preconditioners();
}
//-----------------------------------------------------------------------------
Parameters ViennaCLKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("viennacl_krylov_solver");
  return p;
}
//-----------------------------------------------------------------------------
ViennaCLKrylovSolver::ViennaCLKrylovSolver(std::string method,
					   std::string preconditioner)
  : _method(method), report(false)
{
  // Set parameter values
  parameters = default_parameters();

  // Set solver
  if (_method == "cg")
    _solver.reset( new ViennaCLSolverIterCG(preconditioner) );
  else if (_method == "gmres")
    _solver.reset( new ViennaCLSolverIterGMRES(preconditioner) );
  else if (_method == "bicgstab")
    _solver.reset( new ViennaCLSolverIterBiCGStab(preconditioner) );
  else if (_method == "default")
    _solver.reset( new ViennaCLSolverIterCG(preconditioner) );
}
// //-----------------------------------------------------------------------------
// ViennaCLKrylovSolver::ViennaCLKrylovSolver(ViennaCLPreconditioner& pc)
//   : _method("default"), _pc(reference_to_no_delete_pointer(pc)), report(false)
// {
//   // Set parameter values
//   parameters = default_parameters();
// }
// //-----------------------------------------------------------------------------
// ViennaCLKrylovSolver::ViennaCLKrylovSolver(std::string method,
// 					   ViennaCLPreconditioner& pc)
//   : _method(method), _pc(reference_to_no_delete_pointer(pc)), report(false)
// {
//   // Set parameter values
//   parameters = default_parameters();
// }
//-----------------------------------------------------------------------------
ViennaCLKrylovSolver::~ViennaCLKrylovSolver()
{ 
  // Do nothing 
}
//-----------------------------------------------------------------------------
std::size_t ViennaCLKrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  dolfin_assert(_solver);
  dolfin_assert(_A);

  // Try to first use operator as a uBLAS matrix
  if (has_type<const uBLASMatrix<ublas_sparse_matrix> >(*_A))
    {

      // Check dimensions
      std::size_t M = as_type< const uBLASMatrix<ublas_sparse_matrix > >(*_A).size(0);
      std::size_t N = as_type< const uBLASMatrix<ublas_sparse_matrix > >(*_A).size(1);
      if ( N != b.size() )
	{
	  dolfin_error("ViennaCLKrylovSolver.h",
		       "solve linear system using ViennaCL Krylov solver",
		       "Non-matching dimensions for linear system");
	}
  
      // Reinitialise x if necessary
      if (x.size() != b.size())
	{
	  x.resize(b.size());
	  x.zero();
	}
      
      // Set solver parameters
      _solver->set(parameters);
      // Write a message
      if (report)
	info("Solving linear system of size %d x %d (ViennaCL Krylov solver).", M, N);

      // solver
      // FIXME FIND THE CONVERGE RATIO
      bool converged = true /* false*/;
      std::size_t iterations = 0;
      // Pull the uBLAS matrix ; solve ; save the solution in x
      ublas_vector*   ptr_uBLAS_x = new ublas_vector();
      *ptr_uBLAS_x = _solver->solve( ( as_type< const uBLASMatrix< ublas_sparse_matrix > >(*_A) ).mat(), 
				     ( as_type< const uBLASVector >(b) ).vec() );
      boost::shared_ptr< ublas_vector > shared_ptr_uBLAS_x(ptr_uBLAS_x);
      x = uBLASVector( shared_ptr_uBLAS_x );
      iterations = _solver->number_iteration();
    
      // Check for convergence
      if (!converged)
	{
	  bool error_on_nonconvergence = parameters["error_on_nonconvergence"];
	  if (error_on_nonconvergence)
	    {
	      dolfin_error("ViennaCLKrylovSolver.h",
			   "solve linear system using ViennaCL Krylov solver",
			   "Solution failed to converge");
	    }
	  else
	    warning("ViennaCL Krylov solver failed to converge.");
	}
      else if (report)
	info("Krylov solver converged in %d iterations.", iterations);
      
      return iterations;
    }
  else
    dolfin_error("ViennaCLKrilovSolver.cpp",
		 "solve linear system",
		 "Not supporte orther linear algebra operator backend else than uBLAS.");

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t ViennaCLKrylovSolver::solve(const GenericLinearOperator& A,
					GenericVector& x,
					const GenericVector& b)
{
  // Set operator
  boost::shared_ptr<const GenericLinearOperator> Atmp(&A, NoDeleter());
  set_operator(Atmp);

  return solve(as_type<uBLASVector>(x), as_type<const uBLASVector>(b));
}
//-----------------------------------------------------------------------------
void ViennaCLKrylovSolver::select_preconditioner(std::string preconditioner)
{}
//-----------------------------------------------------------------------------
void ViennaCLKrylovSolver::read_parameters()
{
  // Set tolerances and other parameters
  rtol    = parameters["relative_tolerance"];
  atol    = parameters["absolute_tolerance"];
  div_tol = parameters["divergence_limit"];
  max_it  = parameters["maximum_iterations"];
  // TODO
  //  restart = parameters("gmres")["restart"];
  report  = parameters["report"];
}
//-----------------------------------------------------------------------------
