# The name of this test (should be the basename of this file)
SET(TEST_NAME thetah-sl-testconv-3e)
# The specifically compiled executable that this test uses
SET(EXEC_NAME theta-l-nlev30)

SET(NUM_CPUS 16)

SET(NAMELIST_FILES ${HOMME_ROOT}/test/reg_test/namelists/thetah-sl-testconv-3e.nl)

# compare all of these files against baselines:
SET(NC_OUTPUT_FILES dcmip2012_test1_3e_conv1.nc)

set(TCEN_ERROR_ANCHOR "test1_conv")
set(TCEN_FILENAME "${TEST_NAME}_1.out")
set(TCEN_UPPER_BOUNDS "1.0e-1;1.1e-1;5.0e-2;9.0e-2")
configure_file(
  ${HOMME_SOURCE_DIR}/cmake/TransportCheckErrorNorms.cmake.in
  ${HOMME_BINARY_DIR}/tests/${TEST_NAME}/check.cmake
  @ONLY)
