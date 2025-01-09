COMPRESSIONS_TO_VERIFY="verify_bench_alp_stateful verify_bench_alp_stateless verify_bench_alp_stateful_extended "
VECTOR_COUNT=100

LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable {1} {2} value_bit_width $VECTOR_COUNT 0 ::: $COMPRESSIONS_TO_VERIFY ::: 32 64 2> /dev/null
./print-joblog.sh
