#include "util_tools.h"
#include "types.h"

namespace ffrnn {

__hd__ void SANITY_CHECK(at::Tensor points) {
    assert(at::size(points, 1) == 3);
}

}  // namespace ffrnn
