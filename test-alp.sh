COMPRESSIONS_TO_VERIFY="alp gpu_alp alprd gpu_alprd"
VECTOR_COUNT=1024

LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable {2} {1} $VECTOR_COUNT {3} {4} 0 ::: 32 64 ::: $COMPRESSIONS_TO_VERIFY ::: 0 1 2> /dev/null
./print-joblog.sh
