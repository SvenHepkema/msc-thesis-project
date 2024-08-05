LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/executable 1024 {1} {2} {3} {4} 0 ::: 8 16 32 64 ::: bp ffor :::  0 1 ::: 0 1 2> /dev/null
cat $LOG_FILE | tail -n +2 | perl -ane '$F[6] and print'
