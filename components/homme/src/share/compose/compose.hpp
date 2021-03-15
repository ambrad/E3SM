#ifndef INCLUDE_COMPOSE_HPP
#define INCLUDE_COMPOSE_HPP

#ifdef HAVE_CONFIG_H
# include "config.h.c"
#endif

#include <Kokkos_Core.hpp>

#ifdef _OPENMP
# include <omp.h>
#endif

// Options

#ifdef NDEBUG
//# undef NDEBUG
#endif

#ifndef NDEBUG
# define COMPOSE_BOUNDS_CHECK
#endif

#define COMPOSE_TIMERS
#ifdef COMPOSE_TIMERS
# include "gptl.h"
#endif

// Look for MPI-related memory leaks.
//#define COMPOSE_DEBUG_MPI

#ifndef HORIZ_OPENMP
# ifndef KOKKOS_ENABLE_CUDA
// Mimic GPU threading on host to debug race conditions on a regular CPU.
//#  define COMPOSE_MIMIC_GPU
# endif
// Optionally define this for testing the port code.
# define COMPOSE_PORT
#endif

// Do not modify below here.

#if ! defined COMPOSE_MIMIC_GPU
# if defined HORIZ_OPENMP
#  define COMPOSE_HORIZ_OPENMP
# endif
# if defined COLUMN_OPENMP
#  define COMPOSE_COLUMN_OPENMP
# endif
#endif

// Define COMPOSE_PORT if we're on or mimicking the GPU or being used by
// Hommexx. COMPOSE_PORT is used to provide separate code paths when necessary.
#if ! defined COMPOSE_PORT
# if defined COMPOSE_MIMIC_GPU || defined KOKKOS_ENABLE_CUDA || defined COMPOSE_WITH_HOMMEXX
#  define COMPOSE_PORT
# endif
#endif

#if defined COMPOSE_PORT
# if defined COMPOSE_HORIZ_OPENMP || defined COMPOSE_COLUMN_OPENMP
"This should not happen."
# endif
# if defined COMPOSE_MIMIC_GPU || defined KOKKOS_ENABLE_CUDA
// If defined, then certain buffers need explicit mirroring and copying.
#  define COMPOSE_PORT_SEPARATE_VIEWS
// If defined, do pass1 routines on host. This is for performance checking.
//#  define COMPOSE_PACK_NOSCAN
# endif
#endif

#if defined COMPOSE_BOUNDS_CHECK && defined NDEBUG
# pragma message "NDEBUG but COMPOSE_BOUNDS_CHECK"
#endif

#endif
