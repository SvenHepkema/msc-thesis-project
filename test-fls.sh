TYPES="u32 u64"
KERNELS="decompress query"
UNPACK_N_VECS="1 4"
UNPACK_N_VALS="1"
UNPACKERS="stateless stateless-branchless stateful-cache stateful-local-1 stateful-local-2 stateful-local-4 stateful-register-1 stateful-register-2 stateful-register-4 stateful-register-branchless-1 stateful-register-branchless-2 stateful-register-branchless-4 stateful-branchless"
VECTOR_COUNT=256

LOG_FILE=/tmp/log
parallel --progress --joblog $LOG_FILE ./bin/test {1} {2} {3} {4} {5} none 0 32 0 0 $VECTOR_COUNT 0 ::: $TYPES ::: $KERNELS ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS ::: $UNPACKERS
./print-joblog.sh
