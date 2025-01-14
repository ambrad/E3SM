set(TEST_NAME planar-transport-a-etm)
set(EXEC_NAME theta-l-nlev128-native)
set(NAMELIST_FILES ${HOMME_ROOT}/test/reg_test/namelists/planar-transport-a-etm.nl)
set(NUM_CPUS 16)

set(NC_OUTPUT_FILES planar_transport_a1.nc)

set(TCEN_ERROR_ANCHOR "planar_conv")
set(TCEN_FILENAME "${TEST_NAME}_1.out")
set(TCEN_UPPER_BOUNDS "3.9e-3;3.9e-3;4.1e-3")
configure_file(
  ${HOMME_SOURCE_DIR}/cmake/TransportCheckErrorNorms.cmake.in
  ${HOMME_BINARY_DIR}/tests/${TEST_NAME}/check.cmake
  @ONLY)
