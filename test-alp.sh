COMPRESSIONS_TO_VERIFY="alp"
LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable alp 1024 {1} {2} {3} {4} 0 ::: 64 ::: $COMPRESSIONS_TO_VERIFY :::  0 1 2> /dev/null
./print-joblog.sh
