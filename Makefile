INC := -I $(CUDA_LIBRARY_PATH)/include -I.
LIB := -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand 
CUDA_OBJ_FLAGS = $(INC) $(LIB) 
CLANG_FLAGS = -std=c++17 -g $(WARNINGS)
WARNINGS = -Weverything -Wno-c++98-compat-local-type-template-args -Wno-c++98-compat-pedantic -Wno-c++98-compat -Wno-padded

# For the fast compilations:
DATA_TYPE=uint64_t
VALUE_BIT_WIDTH=63

COMPUTE_CAPABILITY = 61
CUDA_FLAGS = -ccbin /usr/bin/clang++-14 -O3 --resource-usage  -arch=sm_$(COMPUTE_CAPABILITY) -I $(CUDA_LIBRARY_PATH)/include -I. -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand -lcuda -lineinfo $(INC) $(LIB) --expt-relaxed-constexpr


CPU_OBJ := $(patsubst src/cpu/%.cpp, obj/cpu-%.o, $(wildcard src/cpu/*.cpp))
AZIM_OBJ := $(patsubst src/azim/%.cpp, obj/azim-%.o, $(wildcard src/azim/*.cpp))

# OBJ Files
obj/cpu-%.o: src/cpu/%.cpp
	clang++ $^ -O3 -c -o $@ $(CLANG_FLAGS)

obj/azim-%.o: src/azim/%.cpp
	clang++ $^ -O3 -c -o $@ $(CLANG_FLAGS)

obj/gpu.o: src/gpu/fastlanes-global.cu
	nvcc $(CUDA_FLAGS) -c -o $@ $<

# Executables

SOURCE_FILES=src/main.cpp obj/gpu.o $(AZIM_OBJ) $(CPU_OBJ)

fast: $(SOURCE_FILES)
	clang++ $^ -O3 -o bin/$@ $(CLANG_FLAGS) $(CUDA_OBJ_FLAGS) -DDATA_TYPE=$(DATA_TYPE) -DVBW=$(VALUE_BIT_WIDTH)

executable: $(SOURCE_FILES)
	clang++ $^ -O3 -o bin/$@ $(CLANG_FLAGS) $(CUDA_OBJ_FLAGS) 

ub-sanitizer: $(SOURCE_FILES)
	clang++ $^ -fsanitize=undefined -O3 -fno-omit-frame-pointer -o bin/$@ $(CLANG_FLAGS) $(CUDA_OBJ_FLAGS)  -g -DDATA_TYPE=$(DATA_TYPE) -DVBW=$(VALUE_BIT_WIDTH)

address-sanitizer: $(SOURCE_FILES)
	clang++ $^ -fsanitize=address -O3 -fno-omit-frame-pointer -o bin/$@ $(CLANG_FLAGS) $(CUDA_OBJ_FLAGS) -g -DDATA_TYPE=$(DATA_TYPE) -DVBW=$(VALUE_BIT_WIDTH)

clean:
	rm -f bin/*
	rm -f obj/*
