#ifndef INCLUDE_COMPOSE_HPP
#define INCLUDE_COMPOSE_HPP

#ifdef _OPENMP
# include <omp.h>
#endif

// Options

// Uncomment this to look for MPI-related memory leaks.
#define COMPOSE_DEBUG_MPI

// Uncomment this to mimic GPU threading on host to debug race conditions on a
// regular CPU.
#define COMPOSE_MIMIC_GPU

// Optionally define this for testing.
#define COMPOSE_PORT

// Do not modify below here.

#if ! defined COMPOSE_MIMIC_GPU
# if defined HORIZ_OPENMP
#  define COMPOSE_HORIZ_OPENMP
# endif
# if defined COLUMN_OPENMP
#  define COMPOSE_COLUMN_OPENMP
# endif
#endif

#if defined HOMMEXX_VECTOR_SIZE
# define COMPOSE_WITH_HOMMEXX
#endif

// Define COMPOSE_PORT if we're on or mimicking the GPU or being used by
// Hommexx.
#if ! defined COMPOSE_PORT
# if defined COMPOSE_MIMIC_GPU || defined KOKKOS_ENABLE_CUDA || defined COMPOSE_WITH_HOMMEXX
#  define COMPOSE_PORT
# endif
#endif

#if defined COMPOSE_PORT
# if defined COMPOSE_HORIZ_OPENMP || defined COMPOSE_COLUMN_OPENMP
"This should not happen."
# endif
#endif

#endif
