TYPES="f32 f64"
KERNELS="decompress query"
UNPACK_N_VECS="1 4"
UNPACK_N_VALS="1"
UNPACKERS="stateless stateless-branchless stateful-cache stateful-local-1 stateful-local-2 stateful-local-4 stateful-register-1 stateful-register-2 stateful-register-4 stateful-register-branchless-1 stateful-register-branchless-2 stateful-register-branchless-4 stateful-branchless"
PATCHERS="stateless stateful naive naive-branchless prefetch-position prefetch-all prefetch-all-branchless"
VECTOR_COUNT=256

LOG_FILE=/tmp/log
parallel --progress --joblog $LOG_FILE ./bin/test {1} {2} {3} {4} {5} {6} 0 32 0 30 $VECTOR_COUNT 0 ::: $TYPES ::: $KERNELS ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS ::: $UNPACKERS ::: $PATCHERS
./print-joblog.sh
