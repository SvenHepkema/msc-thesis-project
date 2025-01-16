#include <cstddef>
#include <cstdint>
#include <string>

#include "../gpu-fls/fls-test-kernels-bindings.hpp"

#ifndef RUNSPEC_HPP
#define RUNSPEC_HPP

namespace runspec {

struct RunSpecification {
  const size_t count;
  const std::string dataset_name;
  const kernels::KernelSpecification spec;

  RunSpecification() : count(1024), dataset_name("random"), spec(kernels::KernelSpecification()) {}

  RunSpecification(const size_t a_count, const std::string a_dataset_name,
                   const std::string kernel)
      : count(a_count), dataset_name(a_dataset_name),
        spec(kernels::kernel_options.at(kernel)) {}
};

}

#endif // RUNSPEC_HPP
