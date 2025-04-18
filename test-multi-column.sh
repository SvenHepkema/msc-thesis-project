U_TYPES="u32 u64"
F_TYPES="f32 f64"
UNPACK_N_VECS="1 4"
UNPACK_N_VALS="1"
UNPACKERS="stateful-register-2 stateful-register-branchless-2 stateful-branchless"
PATCHERS="dummy naive naive-branchless prefetch-all prefetch-all-branchless"
VECTOR_COUNT=256

LOG_FILE=/tmp/log
echo "============================================="
echo "Old-fls decompressor"
echo "============================================="
parallel --progress --joblog $LOG_FILE ./bin/test u32 query-multi-column 1 32 old-fls none 0 32 0 0 $VECTOR_COUNT {} ::: 0
./print-joblog.sh
parallel --progress --joblog $LOG_FILE ./bin/test f32 query-multi-column 1 32 old-fls {1} 0 16 0 10 $VECTOR_COUNT 0 ::: $PATCHERS
./print-joblog.sh

echo "============================================="
echo "Main kernels"
echo "============================================="
parallel --progress --joblog $LOG_FILE ./bin/test {1} query-multi-column {2} {3} {4} none 0 32 0 0 $VECTOR_COUNT 0 ::: $U_TYPES ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS ::: $UNPACKERS 
./print-joblog.sh
parallel --progress --joblog $LOG_FILE ./bin/test {1} query-multi-column {2} {3} {4} {5} 0 16 0 10 $VECTOR_COUNT 0 ::: $F_TYPES ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS ::: $UNPACKERS ::: $PATCHERS
./print-joblog.sh
