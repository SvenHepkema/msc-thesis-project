COMPRESSIONS_TO_VERIFY="gpu_bp gpu_unffor"
VECTOR_COUNT=1024

LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable {2} {1} $VECTOR_COUNT {3} {4} 0 ::: 8 16 32 64 ::: $COMPRESSIONS_TO_VERIFY :::  0 1 2> /dev/null
./print-joblog.sh
