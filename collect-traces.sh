if [ "$#" -ne 5 ]; then
    echo "Illegal number of parameters"
		echo "Requires <function> <bitwidth> <count> <repeat> <output_file_name>"
		exit 0
fi

FUNCTION=$1
BITWIDTH=$2
COUNT=$3
REPEAT=$4
OUTPUT_FILE=$5

TMP_FILE=/tmp/results.txt
rm $TMP_FILE
touch $TMP_FILE

for i in $(seq 1 $REPEAT);
do
		echo "Starting run" $i
		nvprof --print-gpu-trace ./bin/executable $FUNCTION $BITWIDTH random $COUNT 0 2>&1 | grep void | awk '{print $2}' >> $TMP_FILE
done

./data-analysis/results-reader.py $TMP_FILE $OUTPUT_FILE -r $REPEAT -v value-bit-width
