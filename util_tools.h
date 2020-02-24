#ifndef UTIL_TOOLS_H_
#define UTIL_TOOLS_H_

#include <vector>
#include <cmath>
#include <torch/extension.h>

namespace ffrnn {

extern void SANITY_CHECK(at::Tensor points);

}  // namespace ffrnn

#endif  // UTIL_TOOLS_H_