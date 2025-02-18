#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <iostream>
#include <cstdlib>

struct is_equal_to {
    int value;
    is_equal_to(int v) : value(v) {}
    __host__ __device__ bool operator()(int x) { return x == value; }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <array_size>" << std::endl;
        return 1;
    }
    int size = std::atoi(argv[1]);

    int some_value = -4;
    thrust::host_vector<int> h_vec(size, 0);
    thrust::device_vector<int> d_vec = h_vec;
    bool result = thrust::any_of(thrust::device, d_vec.begin(), d_vec.end(), is_equal_to(some_value));
    std::cout << (result ? "Array contains the value" : "Value not found in array") << std::endl;
    return 0;
}
