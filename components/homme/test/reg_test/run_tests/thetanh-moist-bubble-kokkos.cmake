# The name of this test (should be the basename of this file)
SET(TEST_NAME thetanh-moist-bubble-kokkos)
# The specifically compiled executable that this test uses
SET(EXEC_NAME theta-l-nlev20-native-kokkos)

SET(NUM_CPUS 4)

SET(NAMELIST_FILES ${HOMME_ROOT}/test/reg_test/namelists/thetanh-moist-bubble.nl)

# compare all of these files against baselines:
SET(NC_OUTPUT_FILES planar_rising_bubble1.nc)
