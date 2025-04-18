TYPES="32 64"
KERNELS="decompress query"
UNPACK_N_VECS="1 4"
UNPACK_N_VALS="1"
UNPACKERS="stateless stateless-branchless stateful-cache stateful-local-1 stateful-local-2 stateful-local-4 stateful-register-1 stateful-register-2 stateful-register-4 stateful-register-branchless-1 stateful-register-branchless-2 stateful-register-branchless-4 stateful-branchless"
VECTOR_COUNT=256

LOG_FILE=/tmp/log
echo "============================================="
echo "Dummy decompressor"
echo "============================================="
parallel --progress --joblog $LOG_FILE ./bin/test u{1} {2} {3} {4} dummy none {1} {1} 0 0 $VECTOR_COUNT 0 ::: $TYPES ::: $KERNELS ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS 
./print-joblog.sh
echo "============================================="
echo "Old-fls decompressor"
echo "============================================="
parallel --progress --joblog $LOG_FILE ./bin/test u32 {1} 1 32 old-fls none 0 32 0 0 $VECTOR_COUNT 0 ::: $KERNELS 
./print-joblog.sh
echo "============================================="
echo "Switch-case decompressors"
echo "============================================="
parallel --progress --joblog $LOG_FILE ./bin/test u{1} {2} 1 {3} switch-case none 0 {1} 0 0 $VECTOR_COUNT 0 ::: $TYPES ::: $KERNELS ::: $UNPACK_N_VALS 
echo "============================================="
echo "Main decompressors"
echo "============================================="
parallel --progress --joblog $LOG_FILE ./bin/test u{1} {2} {3} {4} {5} none 0 {1} 0 0 $VECTOR_COUNT 0 ::: $TYPES ::: $KERNELS ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS ::: $UNPACKERS
./print-joblog.sh
