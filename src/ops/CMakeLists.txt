# Find HSA runtime library
find_library(HSA_RUNTIME_LIBRARY NAMES hsa-runtime64)

add_library(
  mori_ops
  dispatch_combine/dispatch_combine.cpp dispatch_combine/internode_v1.cpp
  dispatch_combine/proxy.cpp dispatch_combine/utils.cpp
  dispatch_combine/dma_transfer.cpp)
target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/)
target_link_libraries(
  mori_ops
  mori_application
  mori_shmem
  mori_logging
  hip::host
  hip::device
  ${NUMA_LIBRARIES}
  ${HSA_RUNTIME_LIBRARY})
target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/include
                                           ${NUMA_INCLUDE_DIRS})
