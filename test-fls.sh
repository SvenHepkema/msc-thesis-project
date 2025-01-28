KERNELS_TO_VERIFY="stateless stateful stateless_branchless stateful_branchless"
UNPACK_N_VECTORS="1 4"
UNPACK_N_VALUES="1 4"
EXPERIMENTS_TO_VERIFY="fls_decompress fls_query"
#TYPES_TO_VERIFY="8 16 32 64"
TYPES_TO_VERIFY="32"
VECTOR_COUNT=1024

LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable {3} {1} none {4} {5} {2} random vbw-0-{2} $VECTOR_COUNT 0 ::: $KERNELS_TO_VERIFY ::: $TYPES_TO_VERIFY ::: $EXPERIMENTS_TO_VERIFY ::: $UNPACK_N_VECTORS ::: $UNPACK_N_VALUES 2> /dev/null
./print-joblog.sh
