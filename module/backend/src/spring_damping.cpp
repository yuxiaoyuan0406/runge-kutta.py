#include "../inc/spring_damping.h"
#include "pybind11/buffer_info.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include <cstring>
#include <stdexcept>

namespace py = pybind11;

namespace SpringDamping {
py::array_t<double>
SpringDampingBackend::state_equation (
    py::array_t<double, py::array::c_style | py::array::forcecast> state,
    double t, double a_ext) {
    auto in = state.request ();
    if (in.ndim != 1) {
        throw std::runtime_error ("state must be a 1D array with shape (2,)");
    }
    if (in.shape[0] != 2) {
        throw std::runtime_error ("state must have shape (2,)");
    }

    py::array_t<double> out (in.shape);
    auto outb = out.request ();

    const double* s = static_cast<const double*> (in.ptr);
    double* y = static_cast<double*> (outb.ptr);

    y[0] = s[1];
    y[1] = a_ext - (spring_coef * s[0] + damping_coef * s[1]) / mass;

    return out;
}
} // namespace SpringDamping

PYBIND11_MODULE (SpringDamping, m, py::mod_gil_not_used ()) {
    // m.doc () = "Spring-Damping system backend implemented in C++ (pybind11)";
    py::class_<SpringDamping::SpringDampingBackend> (m, "SpringDampingBackend")
        .def (py::init<double, double, double>(), py::arg ("mass"),
              py::arg ("spring_coef"), py::arg ("damping_coef"))
        .def ("state_equation",
              &SpringDamping::SpringDampingBackend::state_equation,
              py::arg ("state"), py::arg ("t"), py::arg ("a_ext"));
}
