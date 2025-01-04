COMPRESSIONS_TO_VERIFY="verify_bench_alp_stateful verify_bench_alp_stateless verify_bench_alp_stateful_extended "
VECTOR_COUNT=10000

LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable {1} {2} random $VECTOR_COUNT 0 ::: $COMPRESSIONS_TO_VERIFY ::: 32 64 2> /dev/null
./print-joblog.sh
