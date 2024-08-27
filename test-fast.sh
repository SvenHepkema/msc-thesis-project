COMPRESSIONS_TO_VERIFY="gpu_bp"
LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/fast 1024 {1} {2} {3} {4} 0 ::: default ::: $COMPRESSIONS_TO_VERIFY ::: 0 1 2> /dev/null
./print-joblog.sh
