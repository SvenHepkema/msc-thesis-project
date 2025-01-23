KERNELS_TO_VERIFY="test_stateless_1_1 test_stateful_1_1 test_stateless_branchless_1_1 test_stateful_branchless_1_1 test_stateless_4_1 test_stateful_4_1 test_stateless_branchless_4_1 test_stateful_branchless_4_1"
#TYPES_TO_VERIFY="8 16 32 64"
TYPES_TO_VERIFY="32"
VECTOR_COUNT=1024

LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable verify_gpu_bp {1} {2} random vbw-0-{2} $VECTOR_COUNT 0 ::: $KERNELS_TO_VERIFY ::: $TYPES_TO_VERIFY 2> /dev/null
./print-joblog.sh
