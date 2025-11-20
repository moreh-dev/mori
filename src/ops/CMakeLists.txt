add_library(
  mori_ops
  dispatch_combine/dispatch_combine.cpp dispatch_combine/internode_v1.cpp
  dispatch_combine/proxy.cpp dispatch_combine/utils.cpp)
target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/)
target_link_libraries(
  mori_ops
  mori_application
  mori_shmem
  mori_logging
  hip::host
  hip::device
  ${NUMA_LIBRARIES})
target_include_directories(mori_ops PUBLIC ${CMAKE_SOURCE_DIR}/include
                                           ${NUMA_INCLUDE_DIRS})
