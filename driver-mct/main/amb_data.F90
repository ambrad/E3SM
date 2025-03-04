module amb_data_mod
  use mct_mod
  implicit none

  type(mct_aVect) , pointer :: fractions_ax(:)   ! Fractions on atm grid, cpl processes
  type(mct_aVect) , pointer :: fractions_lx(:)   ! Fractions on lnd grid, cpl processes
  type(mct_aVect) , pointer :: fractions_ix(:)   ! Fractions on ice grid, cpl processes
  type(mct_aVect) , pointer :: fractions_ox(:)   ! Fractions on ocn grid, cpl processes
  type(mct_aVect) , pointer :: fractions_gx(:)   ! Fractions on glc grid, cpl processes
  type(mct_aVect) , pointer :: fractions_rx(:)   ! Fractions on rof grid, cpl processes
  type(mct_aVect) , pointer :: fractions_wx(:)   ! Fractions on wav grid, cpl processes
  type(mct_aVect) , pointer :: fractions_zx(:)   ! Fractions on iac grid, cpl processes

end module amb_data_mod
