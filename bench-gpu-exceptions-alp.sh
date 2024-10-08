if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
		echo "Requires <output_file_name> <repeat>"
		exit 0
fi

TMP_FILE=/tmp/results.txt
rm $TMP_FILE
touch $TMP_FILE

for i in $(seq 1 $2);
do
		echo "Starting run" $i
		nvprof --print-gpu-trace ./bin/executable gpu_alp_exceptions 32 random 1024 0 2>&1 | grep void | awk '{print $2}' >> $TMP_FILE
done

./data-analysis/results-reader.py $TMP_FILE $1 -r $2 -v exception-count
