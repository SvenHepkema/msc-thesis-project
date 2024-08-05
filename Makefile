CLANG_FLAGS = -std=c++17

fast: src/main.cpp 
	clang++ $< -O3 -o bin/$@ $(CLANG_FLAGS) -DFAST_COMPILATION

executable: src/main.cpp 
	clang++ $< -O3 -o bin/$@ $(CLANG_FLAGS)
