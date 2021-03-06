# - Try to find SLEPC
# Once done this will define
#
#  SLEPC_FOUND        - system has SLEPc
#  SLEPC_INCLUDE_DIR  - include directories for SLEPc
#  SLEPC_LIBARIES     - libraries for SLEPc
#  SLEPC_DIR          - directory where SLEPc is built
#  SLEPC_VERSION      - version of SLEPc
#
# Assumes that PETSC_DIR and PETSC_ARCH has been set by
# alredy calling find_package(PETSc)

#=============================================================================
# Copyright (C) 2010-2012 Garth N. Wells, Anders Logg and Johannes Ring
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

message(STATUS "Checking for package 'SLEPc'")

# Set debian_arches (PETSC_ARCH for Debian-style installations)
foreach (debian_arches linux kfreebsd)
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(DEBIAN_FLAVORS ${debian_arches}-gnu-c-debug ${debian_arches}-gnu-c-opt ${DEBIAN_FLAVORS})
  else()
    set(DEBIAN_FLAVORS ${debian_arches}-gnu-c-opt ${debian_arches}-gnu-c-debug ${DEBIAN_FLAVORS})
  endif()
endforeach()

# List of possible locations for SLEPC_DIR
set(slepc_dir_locations "")
list(APPEND slepc_dir_locations "/usr/lib/slepcdir/3.4.2")
list(APPEND slepc_dir_locations "/usr/lib/slepcdir/3.2")
list(APPEND slepc_dir_locations "/usr/lib/slepcdir/3.1")
list(APPEND slepc_dir_locations "/usr/lib/slepcdir/3.0.0")
list(APPEND slepc_dir_locations "/opt/local/lib/petsc")    # Macports
list(APPEND slepc_dir_locations "/usr/local/lib/slepc")
list(APPEND slepc_dir_locations "$ENV{HOME}/slepc")

# Add other possible locations for SLEPC_DIR
set(_SYSTEM_LIB_PATHS "${CMAKE_SYSTEM_LIBRARY_PATH};${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES}")
string(REGEX REPLACE ":" ";" libdirs ${_SYSTEM_LIB_PATHS})
foreach (libdir ${libdirs})
  get_filename_component(slepc_dir_location "${libdir}/" PATH)
  list(APPEND slepc_dir_locations ${slepc_dir_location})
endforeach()

# Try to figure out SLEPC_DIR by finding slepc.h
find_path(SLEPC_DIR include/slepc.h
  HINTS ${SLEPC_DIR} $ENV{SLEPC_DIR}
  PATHS ${slepc_dir_locations}
  DOC "SLEPc directory")

# Report result of search for SLEPC_DIR
if (DEFINED SLEPC_DIR)
  message(STATUS "SLEPC_DIR is ${SLEPC_DIR}")
else()
  message(STATUS "SLEPC_DIR is empty")
endif()

# Get variables from SLEPc configuration
if (SLEPC_DIR)

  find_library(SLEPC_LIBRARY
    NAMES slepc
    HINTS ${SLEPC_DIR}/lib $ENV{SLEPC_DIR}/lib  ${SLEPC_DIR}/${PETSC_ARCH}/lib $ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}/lib 
    NO_DEFAULT_PATH
    DOC "The SLEPc library"
    )
  find_library(SLEPC_LIBRARY
    NAMES slepc
    DOC "The SLEPc library"
    )
  mark_as_advanced(SLEPC_LIBRARY)

  # Create a temporary Makefile to probe the SLEPcc configuration
  set(slepc_config_makefile ${PROJECT_BINARY_DIR}/Makefile.slepc)
  file(WRITE ${slepc_config_makefile}
"# This file was autogenerated by FindSLEPc.cmake
SLEPC_DIR  = ${SLEPC_DIR}
PETSC_ARCH = ${PETSC_ARCH}
PETSC_DIR = ${PETSC_DIR}
include ${SLEPC_DIR}/conf/slepc_common
show :
	-@echo -n \${\${VARIABLE}}
")

  # Define macro for getting SLEPc variables from Makefile
  macro(SLEPC_GET_VARIABLE var name)
    set(${var} "NOTFOUND" CACHE INTERNAL "Cleared" FORCE)
    execute_process(COMMAND ${CMAKE_MAKE_PROGRAM} --no-print-directory -f ${slepc_config_makefile} show VARIABLE=${name}
      OUTPUT_VARIABLE ${var}
      RESULT_VARIABLE slepc_return)
  endmacro()

  # Call macro to get the SLEPc variables
  slepc_get_variable(SLEPC_INCLUDE SLEPC_INCLUDE)
  slepc_get_variable(SLEPC_EXTERNAL_LIB SLEPC_EXTERNAL_LIB)

  # Remove temporary Makefile
  file(REMOVE ${slepc_config_makefile})

  # Extract include paths and libraries from compile command line
  include(ResolveCompilerPaths)
  resolve_includes(SLEPC_INCLUDE_DIRS "${SLEPC_INCLUDE}")
  resolve_libraries(SLEPC_EXTERNAL_LIBRARIES "${SLEPC_EXTERNAL_LIB}")

  # Add variables to CMake cache and mark as advanced
  set(SLEPC_INCLUDE_DIRS ${SLEPC_INCLUDE_DIRS} CACHE STRING "SLEPc include paths." FORCE)
  set(SLEPC_LIBRARIES ${SLEPC_LIBRARY} CACHE STRING "SLEPc libraries." FORCE)
  mark_as_advanced(SLEPC_INCLUDE_DIRS SLEPC_LIBRARIES)
endif()

if (DOLFIN_SKIP_BUILD_TESTS)
  set(SLEPC_TEST_RUNS TRUE)
  set(SLEPC_VERSION "UNKNOWN")
  set(SLEPC_VERSION_OK TRUE)
elseif (SLEPC_LIBRARIES AND SLEPC_INCLUDE_DIRS)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${SLEPC_INCLUDE_DIRS} ${PETSC_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${SLEPC_LIBRARIES} ${PETSC_LIBRARIES})

  # Add MPI variables if MPI has been found
  if (MPI_C_FOUND)
    set(CMAKE_REQUIRED_INCLUDES  ${CMAKE_REQUIRED_INCLUDES} ${MPI_C_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${MPI_C_LIBRARIES})
    set(CMAKE_REQUIRED_FLAGS     "${CMAKE_REQUIRED_FLAGS} ${MPI_C_COMPILE_FLAGS}")
  endif()

  # Check SLEPc version
  set(SLEPC_CONFIG_TEST_VERSION_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/slepc_config_test_version.cpp")
  file(WRITE ${SLEPC_CONFIG_TEST_VERSION_CPP} "
#include <iostream>
#include \"slepcversion.h\"

int main() {
  std::cout << SLEPC_VERSION_MAJOR << \".\"
	    << SLEPC_VERSION_MINOR << \".\"
	    << SLEPC_VERSION_SUBMINOR;
  return 0;
}
")

  try_run(
    SLEPC_CONFIG_TEST_VERSION_EXITCODE
    SLEPC_CONFIG_TEST_VERSION_COMPILED
    ${CMAKE_CURRENT_BINARY_DIR}
    ${SLEPC_CONFIG_TEST_VERSION_CPP}
    CMAKE_FLAGS
      "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
    COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
    RUN_OUTPUT_VARIABLE OUTPUT
    )

  if (SLEPC_CONFIG_TEST_VERSION_EXITCODE EQUAL 0)
    set(SLEPC_VERSION ${OUTPUT} CACHE TYPE STRING)
    mark_as_advanced(SLEPC_VERSION)
  endif()

  if (SLEPc_FIND_VERSION)
    # Check if version found is >= required version
    if (NOT "${SLEPC_VERSION}" VERSION_LESS "${SLEPc_FIND_VERSION}")
      set(SLEPC_VERSION_OK TRUE)
    endif()
  else()
    # No specific version requested
    set(SLEPC_VERSION_OK TRUE)
  endif()
  mark_as_advanced(SLEPC_VERSION_OK)

  # Run SLEPc test program
  set(SLEPC_TEST_LIB_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/slepc_test_lib.cpp")
  file(WRITE ${SLEPC_TEST_LIB_CPP} "
#include \"petsc.h\"
#include \"slepceps.h\"
int main()
{
  PetscErrorCode ierr;
  int argc = 0;
  char** argv = NULL;
  ierr = SlepcInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  EPS eps;
  ierr = EPSCreate(PETSC_COMM_SELF, &eps); CHKERRQ(ierr);
  //ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
#if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 1
  ierr = EPSDestroy(eps); CHKERRQ(ierr);
#else
  ierr = EPSDestroy(&eps); CHKERRQ(ierr);
#endif
  ierr = SlepcFinalize(); CHKERRQ(ierr);
  return 0;
}
")

  try_run(
    SLEPC_TEST_LIB_EXITCODE
    SLEPC_TEST_LIB_COMPILED
    ${CMAKE_CURRENT_BINARY_DIR}
    ${SLEPC_TEST_LIB_CPP}
    CMAKE_FLAGS
      "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
      "-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
    COMPILE_OUTPUT_VARIABLE SLEPC_TEST_LIB_COMPILE_OUTPUT
    RUN_OUTPUT_VARIABLE SLEPC_TEST_LIB_OUTPUT
    )

  if (SLEPC_TEST_LIB_COMPILED AND SLEPC_TEST_LIB_EXITCODE EQUAL 0)
    message(STATUS "Performing test SLEPC_TEST_RUNS - Success")
    set(SLEPC_TEST_RUNS TRUE)
  else()
    message(STATUS "Performing test SLEPC_TEST_RUNS - Failed")

    # Test program does not run - try adding SLEPc 3rd party libs and test again
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${SLEPC_EXTERNAL_LIBRARIES})

    try_run(
      SLEPC_TEST_3RD_PARTY_LIBS_EXITCODE
      SLEPC_TEST_3RD_PARTY_LIBS_COMPILED
      ${CMAKE_CURRENT_BINARY_DIR}
      ${SLEPC_TEST_LIB_CPP}
      CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
	"-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
      COMPILE_OUTPUT_VARIABLE SLEPC_TEST_3RD_PARTY_LIBS_COMPILE_OUTPUT
      RUN_OUTPUT_VARIABLE SLEPC_TEST_3RD_PARTY_LIBS_OUTPUT
      )

    if (SLEPC_TEST_3RD_PARTY_LIBS_COMPILED AND SLEPC_TEST_3RD_PARTY_LIBS_EXITCODE EQUAL 0)
      message(STATUS "Performing test SLEPC_TEST_3RD_PARTY_LIBS_RUNS - Success")
      set(SLEPC_LIBRARIES ${SLEPC_LIBRARIES} ${SLEPC_EXTERNAL_LIBRARIES}
	CACHE STRING "SLEPc libraries." FORCE)
      set(SLEPC_TEST_RUNS TRUE)
    else()
      message(STATUS "Performing test SLEPC_TEST_3RD_PARTY_LIBS_RUNS - Failed")
    endif()
  endif()
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SLEPc
  "SLEPc could not be found. Be sure to set SLEPC_DIR, PETSC_DIR, and PETSC_ARCH."
  SLEPC_LIBRARIES SLEPC_DIR SLEPC_INCLUDE_DIRS SLEPC_TEST_RUNS
  SLEPC_VERSION SLEPC_VERSION_OK)
