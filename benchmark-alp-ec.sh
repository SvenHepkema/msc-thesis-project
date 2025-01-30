KERNELS_TO_VERIFY="stateless stateful stateless_branchless stateful_branchless"
PATCHERS_TO_VERIFY="stateless stateful naive naive_branchless prefetch_position prefetch_all prefetch_all_branchless"
UNPACK_N_VECTORS="1 4"
UNPACK_N_VALUES="1 4"
DATATYPE_WIDTH=32
EXPERIMENTS="alp_query"
N_VEC=10000
OUTPUT_DIR="alp-data-ec"
METRICS="global_load_requests,global_hit_rate,l2_tex_hit_rate,inst_issued,stall_memory_dependency,stall_memory_throttle,dram_read_bytes,dram_write_bytes,ipc"

mkdir $OUTPUT_DIR
parallel --progress -j 1 ./collect-traces.py  -tr 3 -m $METRICS -o $OUTPUT_DIR/{2}-{1}-{5}-{3}-{4} -c \"./bin/executable {2} {1} {5} {3} {4} $DATATYPE_WIDTH random ec-0-30 $N_VEC 0\" ::: $KERNELS_TO_VERIFY ::: $EXPERIMENTS ::: $UNPACK_N_VECTORS ::: $UNPACK_N_VALUES ::: $PATCHERS_TO_VERIFY
ls $OUTPUT_DIR | parallel chown $(whoami) $OUTPUT_DIR/{}
