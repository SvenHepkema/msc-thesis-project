KERNELS_TO_VERIFY="stateless stateful-cache stateful-local-1 stateful-local-2 stateful-local-4 stateful-register-1 stateful-register-2 stateful-register-4 stateful-register-branchless-1 stateful-register-branchless-2 stateful-register-branchless-4 stateless_branchless stateful_branchless"
UNPACK_N_VECTORS="1 4"
UNPACK_N_VALUES="1 4"
EXPERIMENTS_TO_VERIFY="fls_decompress fls_query"
#TYPES_TO_VERIFY="8 16 32 64"
TYPES_TO_VERIFY="32"
VECTOR_COUNT=1024

LOG_FILE=/tmp/log
parallel --progress --joblog $LOG_FILE ./bin/executable {3} {1} none {4} {5} {2} random vbw-0-{2} $VECTOR_COUNT 0 ::: $KERNELS_TO_VERIFY ::: $TYPES_TO_VERIFY ::: $EXPERIMENTS_TO_VERIFY ::: $UNPACK_N_VECTORS ::: $UNPACK_N_VALUES 
./print-joblog.sh
