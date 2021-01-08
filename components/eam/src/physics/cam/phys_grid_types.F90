module phys_grid_types

   use shr_kind_mod,     only: r8 => shr_kind_r8
   implicit none

! chunk data structures
   type chunk
     integer  :: ncols                 ! number of vertical columns
     integer, allocatable :: gcol(:)   ! global physics column indices
     integer, allocatable :: lon(:)    ! global longitude indices
     integer, allocatable :: lat(:)    ! global latitude indices
     integer  :: owner                 ! id of process where chunk assigned
     integer  :: lcid                  ! local chunk index
     integer  :: dcols                 ! number of columns in common with co-located dynamics blocks
     real(r8) :: estcost               ! estimated computational cost (normalized)
   end type chunk

   type lchunk
     integer  :: ncols                 ! number of vertical columns
     integer  :: cid                   ! global chunk index
     integer,  allocatable :: gcol(:)  ! global physics column indices
     real(r8), allocatable :: area(:)  ! column surface area (from dynamics)
     real(r8), allocatable :: wght(:)  ! column integration weight (from dynamics)
     real(r8) :: cost                  ! measured computational cost (seconds)
   end type lchunk

   type knuhc
     integer  :: chunkid               ! chunk id
     integer  :: col                   ! column index in chunk
   end type knuhc

contains

end module phys_grid_types
