LOG_FILE=/tmp/log
parallel --joblog $LOG_FILE ./bin/fast 1024 {1} {2} {3} {4} 0 ::: default ::: bp gpu_bp ffor :::  0 1 ::: default 2> /dev/null
cat $LOG_FILE | tail -n +2 | perl -ane '$F[6] and print'
