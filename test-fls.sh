KERNELS_TO_VERIFY="stateless_1_1 stateful_1_1 stateless_branchless_1_1 stateful_branchless_1_1 stateless_4_1 stateful_4_1 stateless_branchless_4_1 stateful_branchless_4_1"
#TYPES_TO_VERIFY="8 16 32 64"
TYPES_TO_VERIFY="32"
VECTOR_COUNT=1024

LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable fls_decompress {1} {2} random vbw-0-{2} $VECTOR_COUNT 0 ::: $KERNELS_TO_VERIFY ::: $TYPES_TO_VERIFY 2> /dev/null
./print-joblog.sh
