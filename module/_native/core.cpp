#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace spring_damping {

    py::array_t<double> state_equation(py::array_t<double, py::array::c_style | py::array::forcecast> state, double t);

}

PYBIND11_MODULE(_core, m) {
    m.def("spring_damping_state_equation", &spring_damping::state_equation, "Spring Damping State Equation");
}
