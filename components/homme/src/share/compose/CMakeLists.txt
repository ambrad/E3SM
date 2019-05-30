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
  compose_slmm_advecter.cpp)
link_to_kokkos(${COMPOSE_LIBRARY})
