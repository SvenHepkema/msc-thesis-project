OUTPUT_FILE=$1
REPEAT=$2
CORRECTNESS_TEST=$3
COMMAND=${@:4}

TMP_FILE=/tmp/results.txt
rm $TMP_FILE
touch $TMP_FILE

echo $COMMAND

if [ "$CORRECTNESS_TEST" -eq 1 ]; then
	$COMMAND
	if [ $? -ne 0 ]; then
			echo "Did not exit with code 0."
			exit
	fi
fi

for i in $(seq 1 $REPEAT);
do
		echo "Starting run" $i
		nvprof --print-gpu-trace $COMMAND 2>&1 | grep void | awk '{print $2}' >> $TMP_FILE
done

./data-analysis/results-reader.py $TMP_FILE $OUTPUT_FILE -r $REPEAT -v value-bit-width
