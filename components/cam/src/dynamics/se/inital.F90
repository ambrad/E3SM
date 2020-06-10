module inital

! Dynamics initialization

implicit none
private

public :: cam_initial

!=========================================================================
contains
!=========================================================================

subroutine cam_initial(dyn_in, dyn_out, NLFileName)

   use dyn_comp,             only: dyn_init1, dyn_init2, dyn_import_t, dyn_export_t
   use phys_grid,            only: phys_grid_init
   use chem_surfvals,        only: chem_surfvals_init
   use cam_initfiles,        only: initial_file_get_id
   use startup_initialconds, only: initial_conds
   use cam_logfile,          only: iulog
   use perf_mod

   ! modules from SE
   use parallel_mod, only : par

   type(dyn_import_t), intent(out) :: dyn_in
   type(dyn_export_t), intent(out) :: dyn_out
   character(len=*),   intent(in)  :: NLFileName
   !----------------------------------------------------------------------

   call t_startf('amb3 dyn_init1')
   call dyn_init1(initial_file_get_id(), NLFileName, dyn_in, dyn_out)
   call t_stopf('amb3 dyn_init1')

   ! Define physics data structures
   if(par%masterproc  ) write(iulog,*) 'Running phys_grid_init()'
   call t_startf('amb3 phys_grid_init')
   call phys_grid_init( )
   call t_stopf('amb3 phys_grid_init')

   ! Initialize ghg surface values before default initial distributions
   ! are set in inidat.
   call t_startf('amb3 chem_surfvals_init')
   call chem_surfvals_init()
   call t_stopf('amb3 chem_surfvals_init')

   if(par%masterproc  ) write(iulog,*) 'Reading initial data'
   call t_startf('amb3 initial_conds')
   call initial_conds(dyn_in)
   call t_stopf('amb3 initial_conds')
   if(par%masterproc  ) write(iulog,*) 'Done Reading initial data'

   call t_startf('amb3 dyn_init2')
   call dyn_init2(dyn_in)
   call t_stopf('amb3 dyn_init2')

end subroutine cam_initial

end module inital
