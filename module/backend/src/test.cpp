#include "../inc/test.h"

namespace py = pybind11;

// sum of squares: sum(x[i]^2)
double Test::sum_squares(py::array_t<double, py::array::c_style | py::array::forcecast> x) {
    auto buf = x.request();
    const auto* ptr = static_cast<const double*>(buf.ptr);
    const size_t n = static_cast<size_t>(buf.size);

    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double v = ptr[i];
        s += v * v;
    }
    return s;
}

PYBIND11_MODULE(Test, m) {
    m.doc() = "Fast math kernels implemented in C++ (pybind11)";
    m.def("sum_squares", &Test::sum_squares, "Compute sum of squares of a 1D array");
}
