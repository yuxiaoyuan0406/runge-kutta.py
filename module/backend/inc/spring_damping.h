#pragma once

#include "common.h"

namespace py = pybind11;

namespace SpringDamping {
class SpringDampingBackend {
public:
    SpringDampingBackend (const double mass, const double spring_coef,
                          const double damping_coef)
        : mass (mass), spring_coef (spring_coef), damping_coef (damping_coef) {}

    py::array_t<double> state_equation (
        py::array_t<double, py::array::c_style | py::array::forcecast> state,
        double a_ext);

private:
    double mass;
    double spring_coef;
    double damping_coef;
};
}
