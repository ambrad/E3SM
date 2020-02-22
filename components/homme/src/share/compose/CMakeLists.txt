include_directories (${CMAKE_CURRENT_BINARY_DIR})
if (ENABLE_HORIZ_OPENMP)
  add_definitions(-DHORIZ_OPENMP)
endif ()
if (ENABLE_COLUMN_OPENMP)
  add_definitions(-DCOLUMN_OPENMP)
endif ()
add_library (${COMPOSE_LIBRARY}
  compose_test.cpp
  compose_cedr.cpp
  compose_slmm.cpp
  compose_slmm_util.cpp
  compose_slmm_departure_point.cpp
  compose_slmm_advecter.cpp
  compose_slmm_islmpi.cpp
  compose_slmm_islmpi_adp.cpp
  compose_slmm_islmpi_comm.cpp
  compose_slmm_islmpi_pack.cpp
  compose_slmm_islmpi_q.cpp
  compose_slmm_islmpi_qextrema.cpp
  compose_slmm_islmpi_step.cpp)
link_to_kokkos(${COMPOSE_LIBRARY})
