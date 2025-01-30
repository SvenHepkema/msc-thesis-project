KERNELS_TO_VERIFY="stateless stateful stateless_branchless stateful_branchless noninterleaved"
UNPACK_N_VECTORS="1 4"
UNPACK_N_VALUES="1"
DATATYPE_WIDTH=32
EXPERIMENTS="fls_decompress fls_query fls_compute"
N_VEC=10000
OUTPUT_DIR="fls-data"
METRICS="global_load_requests,global_hit_rate,l2_tex_hit_rate,inst_issued,stall_memory_dependency,stall_memory_throttle,dram_read_bytes,dram_write_bytes,ipc"

mkdir $OUTPUT_DIR
parallel --progress -j 1 ./collect-traces.py  -tr 3 -m $METRICS -o $OUTPUT_DIR/{2}-{1}-{3}-{4} -c \"./bin/executable {2} {1} none {3} {4} $DATATYPE_WIDTH random vbw-0-32 $N_VEC 0\" ::: $KERNELS_TO_VERIFY ::: $EXPERIMENTS ::: $UNPACK_N_VECTORS ::: $UNPACK_N_VALUES
ls $OUTPUT_DIR | parallel chown $(whoami) $OUTPUT_DIR/{}
