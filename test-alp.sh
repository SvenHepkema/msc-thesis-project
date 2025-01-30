#KERNELS_TO_VERIFY="stateless stateful stateless_branchless stateful_branchless"
#PATCHERS_TO_VERIFY="stateless stateful naive naive_branchless prefetch_position prefetch_all prefetch_all_branchless"
KERNELS_TO_VERIFY="stateless"
PATCHERS_TO_VERIFY="prefetch_position"
UNPACK_N_VECTORS="1 4"
UNPACK_N_VALUES="1 4"
EXPERIMENTS_TO_VERIFY="alp_decompress alp_query"
#TYPES_TO_VERIFY="8 16 32 64"
TYPES_TO_VERIFY="32"
VECTOR_COUNT=256

LOG_FILE=/tmp/log
parallel --progress --joblog $LOG_FILE ./bin/executable {3} {1} {6} {4} {5} {2} random vbw-0-16 $VECTOR_COUNT 0 ::: $KERNELS_TO_VERIFY ::: $TYPES_TO_VERIFY ::: $EXPERIMENTS_TO_VERIFY ::: $UNPACK_N_VECTORS ::: $UNPACK_N_VALUES ::: $PATCHERS_TO_VERIFY 
./print-joblog.sh
