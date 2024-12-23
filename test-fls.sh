COMPRESSIONS_TO_VERIFY="bp ffor gpu_bp gpu_bp_state gpu_bp_multivec gpu_unffor"
VECTOR_COUNT=1024

LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable {1} {2} {3} $VECTOR_COUNT 0 ::: $COMPRESSIONS_TO_VERIFY ::: 8 16 32 64 ::: index random 2> /dev/null
./print-joblog.sh
