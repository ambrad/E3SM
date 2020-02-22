#ifndef INCLUDE_COMPOSE_HPP
#define INCLUDE_COMPOSE_HPP

#ifdef HORIZ_OPENMP
# define COMPOSE_HORIZ_OPENMP
#endif

#ifdef COLUMN_OPENMP
# define COMPOSE_COLUMN_OPENMP
#endif

#if ! defined COMPOSE_HORIZ_OPENMP && ! defined COMPOSE_COLUMN_OPENMP
# define COMPOSE_MIMIC_GPU
#endif
#if defined COMPOSE_MIMIC_GPU
# if defined COMPOSE_COLUMN_OPENMP
#  undef COMPOSE_COLUMN_OPENMP
# endif
#endif

#if defined COMPOSE_MIMIC_GPU || defined KOKKOS_ENABLE_CUDA || defined HOMMEXX_VECTOR_SIZE
# define COMPOSE_PORT
#endif

#ifdef COMPOSE_PORT
# if defined COMPOSE_HORIZ_OPENMP || defined COMPOSE_COLUMN_OPENMP
"This should not happen."
# endif
#endif

#endif
