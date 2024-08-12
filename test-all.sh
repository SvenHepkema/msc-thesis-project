COMPRESSIONS_TO_VERIFY="bp ffor azim_bp gpu_bp"
LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable 1024 {1} {2} {3} {4} 0 ::: 8 16 32 64 ::: $COMPRESSIONS_TO_VERIFY :::  0 1 2> /dev/null
./print-joblog.sh
