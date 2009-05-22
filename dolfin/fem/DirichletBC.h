// Copyright (C) 2007-2008 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian Oelgaard, 2007
//
// First added:  2007-04-10
// Last changed: 2008-12-12
//
// FIXME: This class needs some cleanup, in particular collecting
// FIXME: all data from different representations into a common
// FIXME: data structure (perhaps an std::vector<uint> with facet indices).

#ifndef __DIRICHLET_BC_H
#define __DIRICHLET_BC_H

#include <map>
#include <set>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include "BoundaryCondition.h"

namespace dolfin
{
  class Function;
  class FunctionSpace;
  class Facet;
  class GenericMatrix;
  class GenericVector;
  class SubDomain;
  template<class T> class MeshFunction;

  /// The BCMethod variable may be used to specify the type of method
  /// used to identify degrees of freedom on the boundary. Available
  /// methods are: topological approach (default), geometric approach,
  /// and pointwise approach. The topological approach is faster,
  /// but will only identify degrees of freedom that are located on a
  /// facet that is entirely on the boundary. In particular, the
  /// topological approach will not identify degrees of freedom
  /// for discontinuous elements (which are all internal to the cell).
  /// A remedy for this is to use the geometric approach. To apply
  /// pointwise boundary conditions e.g. pointloads, one will have to
  /// use the pointwise approach which in turn is the slowest of the
  /// three possible methods.
  /// The three possibilties are:
  ///   "topological"
  ///   "geometric"
  ///   "pointwise"

  /// This class specifies the interface for setting (strong)
  /// Dirichlet boundary conditions for partial differential
  /// equations,
  ///
  ///    u = g on G,
  ///
  /// where u is the solution to be computed, g is a function
  /// and G is a sub domain of the mesh.
  ///
  /// A DirichletBC is specified by the Function g, the FunctionSpace
  /// (trial space) and boundary indicators on (a subset of) the mesh
  /// boundary.
  ///
  /// The boundary indicators may be specified in a number of
  /// different ways.
  ///
  /// The simplest approach is to specify the boundary by a SubDomain
  /// object, using the inside() function to specify on which facets
  /// the boundary conditions should be applied.
  ///
  /// Alternatively, the boundary may be specified by a MeshFunction
  /// labeling all mesh facets together with a number that specifies
  /// which facets should be included in the boundary.
  ///
  /// The third option is to attach the boundary information to the
  /// mesh. This is handled automatically when exporting a mesh from
  /// for example VMTK.

  class DirichletBC : public BoundaryCondition
  {
  public:

    /// Create boundary condition for subdomain
    DirichletBC(const FunctionSpace& V,
                const Function& g,
                const SubDomain& sub_domain,
                std::string method="topological");

    /// Create boundary condition for subdomain specified by index
    DirichletBC(const FunctionSpace& V,
                const Function& g,
                const MeshFunction<uint>& sub_domains, uint sub_domain,
                std::string method="topological");

    /// Create boundary condition for boundary data included in the mesh
    DirichletBC(const FunctionSpace& V,
                const Function& g,
                uint sub_domain,
                std::string method="topological");

    /// Destructor
    ~DirichletBC();

    /// Apply boundary condition to a matrix
    void apply(GenericMatrix& A) const;

    /// Apply boundary condition to a vector
    void apply(GenericVector& b) const;

    /// Apply boundary condition to a linear system
    void apply(GenericMatrix& A, GenericVector& b) const;

    /// Apply boundary condition to a vector for a nonlinear problem
    void apply(GenericVector& b, const GenericVector& x) const;

    /// Apply boundary condition to a linear system for a nonlinear problem
    void apply(GenericMatrix& A, GenericVector& b, const GenericVector& x) const;

    /// Make row associated with boundary conditions zero, useful for non-diagonal matrices in a block matrix.
    void zero(GenericMatrix& A) const;

    /// Get Dirichlet values and indicators
    void get_bc(uint* indicators, double* values) const;

    /// Check if given function is compatible with boundary condition (checking only vertex values)
    bool is_compatible(Function& v) const;

  private:

    // Apply boundary conditions
    void apply(GenericMatrix* A, GenericVector* b, const GenericVector* x) const;

    // Check input data to constructor
    void check() const;

    // Check input data to apply()
    void check(GenericMatrix* A, GenericVector* b, const GenericVector* x) const;

    // Initialize sub domain markers from sub domain
    void init_from_sub_domain(const SubDomain& sub_domain);

    // Initialize sub domain markers from MeshFunction
    void init_from_mesh_function(const MeshFunction<uint>& sub_domains, uint sub_domain);

    // Initialize sub domain markers from mesh
    void init_from_mesh(uint sub_domain);

    // Compute dofs and values for application of boundary conditions
    void compute_bc(std::map<uint, double>& boundary_values,
                    BoundaryCondition::LocalData& data) const;

    // Compute boundary values for facet (topological approach)
    void compute_bc_topological(std::map<uint, double>& boundary_values,
                                BoundaryCondition::LocalData& data) const;

    // Compute boundary values for facet (geometrical approach)
    void compute_bc_geometric(std::map<uint, double>& boundary_values,
                              BoundaryCondition::LocalData& data) const;

    // Compute boundary values for facet (pointwise approach)
    void compute_bc_pointwise(std::map<uint, double>& boundary_values,
                              BoundaryCondition::LocalData& data) const;

    // Check if the point is in the same plane as the given facet
    bool on_facet(double* coordinates, Facet& facet) const;

    // The function
    const Function& g;

    // Search method
    std::string method;

    // Possible search methods
    static const std::set<std::string> methods;

    // User defined sub domain
    const SubDomain* user_sub_domain;

    // Boundary facets, stored as pairs (cell, local facet number)
    std::vector< std::pair<uint, uint> > facets;

  };

}

#endif
