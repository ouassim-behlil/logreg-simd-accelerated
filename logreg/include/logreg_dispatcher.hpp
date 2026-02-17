// logreg_dispatcher.hpp file

#ifndef LOGREG_DISPATCHER_H
# define LOGREG_DISPATCHER_H

# include "cpu_arch.hpp"
# include "cpu_features.hpp"
# include <stdint.h>

// External function pointer for the selected dot product implementation
extern float (*dot_product)(const float* a, const float* b, uint64_t n);

void	init_kernels();
#endif