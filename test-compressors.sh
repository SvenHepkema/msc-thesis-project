FLOAT_FILES=$(ls data-input/floats/*.bin)
DOUBLE_FILES=$(ls data-input/doubles/*.bin)
COMPARISON="decompression decompression_query"
COMPRESSORS="ALP GALP Bitcomp BitcompSparse LZ4 zstd Deflate GDeflate Snappy"
VECTOR_COUNT=256

LOG_FILE=/tmp/log
echo "============================================="
echo "Thrust"
echo "============================================="
parallel --progress --joblog $LOG_FILE ./bin/bench-compressors f32 decompression_query Thrust {1} $VECTOR_COUNT ::: $FLOAT_FILES
./print-joblog.sh
parallel --progress --joblog $LOG_FILE ./bin/bench-compressors f64 decompression_query Thrust {1} $VECTOR_COUNT ::: $DOUBLE_FILES
./print-joblog.sh
echo "============================================="
echo "Floats"
echo "============================================="
parallel --progress --joblog $LOG_FILE ./bin/bench-compressors f32 {1} {2} {3} $VECTOR_COUNT ::: $COMPARISON ::: $COMPRESSORS ::: $FLOAT_FILES
./print-joblog.sh

echo "============================================="
echo "Double"
echo "============================================="
parallel --progress --joblog $LOG_FILE ./bin/bench-compressors f64 {1} {2} {3} $VECTOR_COUNT ::: $COMPARISON ::: $COMPRESSORS ::: $DOUBLE_FILES
./print-joblog.sh
