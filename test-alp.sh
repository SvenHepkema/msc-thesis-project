COMPRESSIONS_TO_VERIFY="alp gpu_alp alprd gpu_alprd"
VECTOR_COUNT=1024

LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable {1} {2} random $VECTOR_COUNT 0 ::: $COMPRESSIONS_TO_VERIFY ::: 32 64 2> /dev/null
./print-joblog.sh
