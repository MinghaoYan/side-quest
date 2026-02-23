#include <torch/extension.h>
torch::Tensor custom_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("custom_cuda", torch::wrap_pybind_function(custom_cuda), "custom_cuda");
}