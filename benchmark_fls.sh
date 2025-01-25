KERNELS_TO_VERIFY="stateless_1_1 stateful_1_1 stateless_branchless_1_1 stateful_branchless_1_1 stateless_4_1 stateful_4_1 stateless_branchless_4_1 stateful_branchless_4_1 stateless_1_4 stateful_1_4 stateless_branchless_1_4 stateful_branchless_1_4"
EXPERIMENTS="fls_decompress fls_query"
N_VEC=10000
OUTPUT_DIR="data"
parallel -j 1 ./collect-traces.py  -tr 3 -m global_hit_rate,dram_read_bytes,dram_write_bytes,ipc -o $OUTPUT_DIR/{1} -c \"./bin/executable {2} {1} 32 random vbw-0-32 $N_VEC 0\" ::: $KERNELS_TO_VERIFY ::: $EXPERIMENTS
ls $OUTPUT_DIR | parallel chown $(whoami) $OUTPUT_DIR/{}
