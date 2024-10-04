INC := -I $(CUDA_LIBRARY_PATH)/include -I.
LIB := -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand 
CUDA_OBJ_FLAGS = $(INC) $(LIB) 
OPTIMIZATION_LEVEL = -O3
CLANG_FLAGS = -std=c++17 -g $(WARNINGS)
WARNINGS = -Weverything -Wno-c++98-compat-local-type-template-args -Wno-c++98-compat-pedantic -Wno-c++98-compat -Wno-padded -Wno-float-equal -Wno-global-constructors -Wno-exit-time-destructors 

COMPUTE_CAPABILITY = 61
CUDA_FLAGS = -ccbin /usr/bin/clang++-14 $(OPTIMIZATION_LEVEL) --resource-usage  -arch=sm_$(COMPUTE_CAPABILITY) -I $(CUDA_LIBRARY_PATH)/include -I. -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand -lcuda -lineinfo $(INC) $(LIB) --expt-relaxed-constexpr


FLS_OBJ := $(patsubst src/fls/%.cpp, obj/fls-%.o, $(wildcard src/fls/*.cpp))
ALP_OBJ := $(patsubst src/alp/%.cpp, obj/alp-%.o, $(wildcard src/alp/*.cpp))

# OBJ Files
obj/fls-%.o: src/fls/%.cpp
	clang++ $^  -c -o $@ $(CLANG_FLAGS)

obj/alp-%.o: src/alp/%.cpp
	clang++ $^  -c -o $@ $(CLANG_FLAGS)

obj/gpu-fls.o: src/gpu-fls/gpu-bindings-fls.cu
	nvcc $(CUDA_FLAGS) -c -o $@ $<

obj/gpu-alp.o: src/gpu-alp/alp-test-kernels-setup.cu obj/gpu-fls.o
	nvcc $(CUDA_FLAGS) -c -o $@ $<

# Executables

HEADER_FILES=$(wildcard src/*.h) $(wildcard src/cpu/*.cuh) $(wildcard src/gpu/*.cuh)
SOURCE_FILES=src/main.cpp obj/gpu-fls.o obj/gpu-alp.o $(FLS_OBJ) $(ALP_OBJ)

executable: $(SOURCE_FILES) $(HEADER_FILES)
	clang++ $(SOURCE_FILES) $(OPTIMIZATION_FLAG) -o bin/$@ $(CLANG_FLAGS) $(CUDA_OBJ_FLAGS) 

ub-sanitizer: $(SOURCE_FILES) $(HEADER_FILES)
	clang++ $(SOURCE_FILES) -O2 -fsanitize=undefined  -fno-omit-frame-pointer -o bin/$@ $(CLANG_FLAGS) $(CUDA_OBJ_FLAGS)  -g 

address-sanitizer: $(SOURCE_FILES) $(HEADER_FILES)
	clang++ $(SOURCE_FILES) -O3 -fsanitize=address  -fno-omit-frame-pointer -o bin/$@ $(CLANG_FLAGS) $(CUDA_OBJ_FLAGS) -g 

clean:
	rm -f bin/*
	rm -f obj/*
