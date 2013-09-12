# - Try to find ViennaCL
# Once done this will define
#
#  VIENNACL_FOUND        - system has VIENNACL
#  VIENNACL_INCLUDE_DIRS - include directories for VIENNACL
#  VIENNACL_LIBRARIES    - libraries for VIENNACL
#  VIENNACL_DEFINITIONS  - compiler flags for VIENNACL

#=============================================================================
# Copyright (C) 2010-2011 Anders Logg, Johannes Ring and Garth N. Wells
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

message(STATUS "Checking for package 'VIENNACL'")

#
set(CGAL_FIND_VERSION "")

# Check for header file
find_path(VIENNACL_INCLUDE_DIRS /viennacl/linalg/cg.hpp
  HINTS ${VIENNACL_DIR} $ENV{VIENNACL_DIR}
  PATH_SUFFIXES include
  DOC "Directory where the VIENNACL header is located"
  )

# Set variables
set(VIENNACL_INCLUDE_DIRS ${VIENNACL_INCLUDE_DIRS} ${VIENNACL_3RD_PARTY_INCLUDE_DIRS})

# Libraries
set(VIENNACL_LIBRARIES ${DOLFIN_VIENNACL_LIBRARIES} ${VIENNACL_3RD_PARTY_LIBRARIES})
find_library(VIENNACL_LIBRARIES
  NAMES OpenCL
  HINTS ${VIENNACL_LIBRARY_DIRS}
  DOC "The OpenCL libraries"
  )

# Try compiling and running test program
if (DOLFIN_SKIP_BUILD_TESTS)
  set(VIENNACL_TEST_RUNS TRUE)
elseif (VIENNACL_INCLUDE_DIRS AND VIENNACL_LIBRARIES)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${VIENNACL_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${VIENNACL_LIBRARIES})
  set(CMAKE_REQUIRED_FLAGS ${VIENNACL_CXX_FLAGS_INIT})

  # Build and run test program
  include(CheckCXXSourceRuns)
  check_cxx_source_runs("
#ifndef NDEBUG
 #define NDEBUG
#endif

#ifndef VIENNACL_WITH_OPENCL
  #define VIENNACL_WITH_OPENCL
#endif

#include \"viennacl/scalar.hpp\"
#include \"viennacl/vector.hpp\"
#include \"viennacl/matrix.hpp\"
#include \"viennacl/compressed_matrix.hpp\"

#include <iostream>
#include <vector>


#define BENCHMARK_VECTOR_SIZE   100000


template<typename ScalarType>
int run_benchmark()
{
   
  std::vector<ScalarType> std_vec1(BENCHMARK_VECTOR_SIZE);
  
  viennacl::ocl::get_queue().finish();
  
  viennacl::scalar<ScalarType> vcl_s1;
  
  viennacl::vector<ScalarType> vcl_vec1(BENCHMARK_VECTOR_SIZE);
  viennacl::vector<ScalarType> vcl_vec2(BENCHMARK_VECTOR_SIZE);
 
  viennacl::matrix<ScalarType> vcl_matrix(BENCHMARK_VECTOR_SIZE/100, BENCHMARK_VECTOR_SIZE/100);
  
  viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix(BENCHMARK_VECTOR_SIZE, BENCHMARK_VECTOR_SIZE);
  
  
  std_vec1[0] = 1.0;
  for (int i=1; i<BENCHMARK_VECTOR_SIZE; ++i)
    std_vec1[i] = std_vec1[i-1] * ScalarType(1.000001);

  viennacl::copy(std_vec1, vcl_vec1);
  
  double std_accumulate = 0;
  double vcl_accumulate = 0;

  for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
    std_accumulate += std_vec1[i];
 
  vcl_accumulate = vcl_vec1[0];
  viennacl::ocl::get_queue().finish();
  vcl_accumulate = 0;
  for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
    vcl_accumulate += vcl_vec1[i];
 
  return 0;
}

int main()
{
 
  std::cout << viennacl::ocl::current_device().info() << std::endl;
  
  run_benchmark<float>();
  if( viennacl::ocl::current_device().double_support() )
  {
    run_benchmark<double>();
  }
  return 0;
}
" VIENNACL_TEST_RUNS)
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VIENNACL
  "VIENNACL could not be found. Be sure to set VIENNACL_DIR"
  VIENNACL_LIBRARIES VIENNACL_INCLUDE_DIRS VIENNACL_TEST_RUNS)
