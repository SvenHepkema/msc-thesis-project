TYPES="f32 f64"
KERNELS="decompress query"
UNPACK_N_VECS="1 4"
UNPACK_N_VALS="1"
UNPACKERS="stateless stateless-branchless stateful-cache stateful-local-1 stateful-local-2 stateful-local-4 stateful-shared-1 stateful-shared-2 stateful-shared-4 stateful-register-1 stateful-register-2 stateful-register-4 stateful-register-branchless-1 stateful-register-branchless-2 stateful-register-branchless-4 stateful-branchless"
PATCHERS="stateless stateful naive naive-branchless prefetch-position prefetch-all prefetch-all-branchless"
VECTOR_COUNT=256

LOG_FILE=/tmp/log
echo "============================================="
echo "Old-fls decompressor"
echo "============================================="
parallel --progress --joblog $LOG_FILE ./bin/test f32 {1} 1 32 old-fls {2} 0 16 0 10 $VECTOR_COUNT 0 ::: $KERNELS ::: $PATCHERS
./print-joblog.sh
echo "============================================="
echo "Dummy patcher"
echo "============================================="
parallel --progress --joblog $LOG_FILE ./bin/test {1} {2} {3} {4} {5} dummy 0 16 0 0 $VECTOR_COUNT 0 ::: $TYPES ::: $KERNELS ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS ::: $UNPACKERS 
./print-joblog.sh
echo "============================================="
echo "Main decompressors"
echo "============================================="
parallel --progress --joblog $LOG_FILE ./bin/test {1} {2} {3} {4} {5} {6} 0 16 0 10 $VECTOR_COUNT 0 ::: $TYPES ::: $KERNELS ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS ::: $UNPACKERS ::: $PATCHERS
./print-joblog.sh
