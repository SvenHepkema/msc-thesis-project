INC := -I $(CUDA_LIBRARY_PATH)/include -I.
LIB := -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand 
CLANG_FLAGS = -std=c++17 -g $(INC) $(LIB) $(WARNINGS)
WARNINGS = -Weverything -Wno-c++98-compat-local-type-template-args -Wno-c++98-compat -Wno-padded

# For the fast compilations:
DATA_TYPE=int8_t
VALUE_BIT_WIDTH=3

COMPUTE_CAPABILITY = 61
CUDA_FLAGS = -ccbin /usr/bin/clang++-14 -O3 --resource-usage -opt-info inline  -arch=sm_$(COMPUTE_CAPABILITY) -I $(CUDA_LIBRARY_PATH)/include -I. -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand -lcuda -lineinfo $(INC) $(LIB)

obj/gpu.o: src/gpu/fastlanes-global.cu
	nvcc $(CUDA_FLAGS) -c -o $@ $<

fast: src/main.cpp obj/gpu.o
	clang++ $^ -O3 -o bin/$@ $(CLANG_FLAGS) -DDATA_TYPE=$(DATA_TYPE) -DVBW=$(VALUE_BIT_WIDTH)

executable: src/main.cpp obj/gpu.o
	clang++ $^ -O3 -o bin/$@ $(CLANG_FLAGS)

ub-sanitizer: src/main.cpp obj/gpu.o
	clang++ $^ -fsanitize=undefined -O3 -fno-omit-frame-pointer -o bin/$@ $(CLANG_FLAGS) -g -DDATA_TYPE=$(DATA_TYPE) -DVBW=$(VALUE_BIT_WIDTH)

address-sanitizer: src/main.cpp obj/gpu.o
	clang++ $^ -fsanitize=address -O3 -fno-omit-frame-pointer -o bin/$@ $(CLANG_FLAGS) -g -DDATA_TYPE=$(DATA_TYPE) -DVBW=$(VALUE_BIT_WIDTH)

clean:
	rm -f bin/*
	rm -f obj/*
