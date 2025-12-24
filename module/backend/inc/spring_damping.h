#pragma once

#include "common.h"

namespace py = pybind11;

namespace SpringDamping {

// sum of squares: sum(x[i]^2)
double sum_squares(py::array_t<double, py::array::c_style | py::array::forcecast> x);

}
