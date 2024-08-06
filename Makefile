CLANG_FLAGS = -std=c++17

fast: src/main.cpp 
	clang++ $< -O0 -o bin/$@ $(CLANG_FLAGS) -DFAST_COMPILATION -DVBW=64

executable: src/main.cpp 
	clang++ $< -O3 -o bin/$@ $(CLANG_FLAGS)

ub-sanitizer: src/main.cpp 
	clang++ $< -fsanitize=undefined -O3 -o bin/$@ $(CLANG_FLAGS) -g -DFAST_COMPILATION -DVBW=63
