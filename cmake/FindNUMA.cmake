find_path(
  NUMA_INCLUDE_DIRS
  NAMES numa.h
  HINTS ${NUMA_INCLUDE_DIR} ${NUMA_ROOT_DIR} ${NUMA_ROOT_DIR}/include)

find_library(
  NUMA_LIBRARIES
  NAMES numa
  HINTS ${NUMA_LIB_DIR} ${NUMA_ROOT_DIR} ${NUMA_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_INCLUDE_DIRS
                                  NUMA_LIBRARIES)
mark_as_advanced(NUMA_INCLUDE_DIR NUMA_LIBRARIES)
