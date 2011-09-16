// Copyright (C) 2011 Anders Logg and Garth N. Wells
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
// First added:  2006-08-30
// Last changed: 2011-09-15

#ifndef __MESH_VALUE_COLLECTION_H
#define __MESH_VALUE_COLLECTION_H

#include <map>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>
#include "MeshEntity.h"
#include "Cell.h"
#include "MeshFunction.h"

namespace dolfin
{

  /// The MeshValueCollection class can be used to store data
  /// associated with a subset of the entities of a mesh of a given
  /// topological dimension. It differs from the MeshFunction class in
  /// two ways. First, data does not need to be associated with all
  /// entities (only a subset). Second, data is associated with
  /// entities through the corresponding cell index and local entity
  /// number (relative to the cell), not by global entity index, which
  /// means that data may be stored robustly to file.

  template <class T> class MeshValueCollection : public Variable
  {
  public:

    /// Create empty mesh value collection of given dimension on given mesh
    ///
    /// *Arguments*
    ///     dim (uint)
    ///         The mesh entity dimension for the mesh value collection.
    MeshValueCollection(uint dim);

    /// Destructor
    ~MeshValueCollection()
    {}

    /// Return topological dimension
    ///
    /// *Returns*
    ///     uint
    ///         The dimension.
    uint dim() const;

    /// Return size (number of entities in subset)
    ///
    /// *Returns*
    ///     uint
    ///         The size.
    uint size() const;

    /// Set marker value for given entity defined by a cell index and
    /// a local entity index
    ///
    /// *Arguments*
    ///     cell_index (uint)
    ///         The index of the cell.
    ///     local_entity (uint)
    ///         The local index of the entity relative to the cell.
    ///     marker_value (T)
    ///         The value of the marker.
    void set_value(uint cell_index, uint local_entity, const T& marker_value);

    /// Set value for given entity index
    ///
    /// *Arguments*
    ///     entity_index (uint)
    ///         Index of the entity.
    ///     value (T).
    ///         The value.
    ///     mesh (Mesh)
    ///         The mesh.
    void set_value(uint entity_index, const T& value, const Mesh& mesh);

    /// Get all values
    ///
    /// *Returns*
    ///     std::map<std::pair<uint, uint>, T>
    ///         A map from positions to values.
    std::map<std::pair<uint, uint>, T>& values();

    /// Get all values (const version)
    ///
    /// *Returns*
    ///     std::map<std::pair<uint, uint>, T>
    ///         A map from positions to values.
    const std::map<std::pair<uint, uint>, T>& values() const;

    /// Clear all values
    void clear();

    /// Return informal string representation (pretty-print)
    ///
    /// *Arguments*
    ///     verbose (bool)
    ///         Flag to turn on additional output.
    ///
    /// *Returns*
    ///     std::string
    ///         An informal representation.
    std::string str(bool verbose) const;

  private:

    // The values
    std::map<std::pair<uint, uint>, T> _values;

    /// Topological dimension
    uint _dim;

  };

  //---------------------------------------------------------------------------
  // Implementation of MeshValueCollection
  //---------------------------------------------------------------------------
  template <class T>
  MeshValueCollection<T>::MeshValueCollection(uint dim)
    : Variable("m", "unnamed MeshValueCollection"), _dim(dim)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <class T>
  uint MeshValueCollection<T>::dim() const
  {
    return _dim;
  }
  //---------------------------------------------------------------------------
  template <class T>
  uint MeshValueCollection<T>::size() const
  {
    return _values.size();
  }
  //---------------------------------------------------------------------------
  template <class T>
  void MeshValueCollection<T>::set_value(uint cell_index,
                                         uint local_entity,
                                         const T& value)
  {
    std::pair<uint, uint> pos(std::make_pair(cell_index, local_entity));
    _values[pos] = value;
  }
  //---------------------------------------------------------------------------
  template <class T>
  void MeshValueCollection<T>::set_value(uint entity_index,
                                         const T& value,
                                         const Mesh& mesh)
  {
    // Get mesh connectivity d --> D
    const uint D = mesh.topology().dim();
    mesh.init(_dim, D);
    const MeshConnectivity& connectivity = mesh.topology()(_dim, D);

    // Find the cell
    assert(connectivity.size() > 0);
    assert(connectivity.size(entity_index) > 0);
    MeshEntity entity(mesh, _dim, entity_index);
    Cell cell(mesh, connectivity(entity_index)[0]); // choose first

    // Find the local entity index
    const uint local_entity = cell.index(entity);

    // Add value
    std::pair<uint, uint> pos(std::make_pair(cell.index(), local_entity));
    _values[pos] = value;
  }
  //---------------------------------------------------------------------------
  template <class T>
  std::map<std::pair<uint, uint>, T>& MeshValueCollection<T>::values()
  {
    return _values;
  }
  //---------------------------------------------------------------------------
  template <class T>
  const std::map<std::pair<uint, uint>, T>& MeshValueCollection<T>::values() const
  {
    return _values;
  }
  //---------------------------------------------------------------------------
  template <class T>
  void MeshValueCollection<T>::clear()
  {
    _values.clear();
  }
  //---------------------------------------------------------------------------
  template <class T>
  std::string MeshValueCollection<T>::str(bool verbose) const
  {
    std::stringstream s;

    if (verbose)
    {
      s << str(false) << std::endl << std::endl;
      warning("Verbose output of MeshValueCollection must be implemented manually.");
    }
    else
      s << "<MeshValueCollection of topological dimension " << dim()
        << " containing " << size() << " values>";

    return s.str();
  }
  //---------------------------------------------------------------------------

}

#endif