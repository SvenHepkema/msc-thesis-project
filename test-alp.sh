COMPRESSIONS_TO_VERIFY="alp gpu_alp alprd gpu_alprd"
LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable 1024 {1} {2} {3} {4} 0 ::: 32 64 ::: $COMPRESSIONS_TO_VERIFY ::: 0 1 2> /dev/null
./print-joblog.sh
