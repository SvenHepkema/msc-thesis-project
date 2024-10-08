if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
		exit 0
fi

TMP_FILE=/tmp/results.txt
rm $TMP_FILE
touch $TMP_FILE

for i in $(seq 1 $1);
do
		echo "Starting run" $i
		nvprof --print-gpu-trace ./bin/executable gpu_alp_exceptions 32 random 1024 0 2>&1 | grep void | awk '{print $2}' >> $TMP_FILE
done
./data-analysis/graph-generator.py $TMP_FILE -r $1

