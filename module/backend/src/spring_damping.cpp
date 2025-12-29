#include "../inc/spring_damping.h"
#include "pybind11/attr.h"
#include "pybind11/buffer_info.h"
#include "pybind11/cast.h"
#include "pybind11/gil.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include <cstring>
#include <stdexcept>

namespace py = pybind11;

namespace SpringDamping {
py::array_t<double>
SpringDampingBackend::state_equation (
    py::array_t<double, py::array::c_style | py::array::forcecast> state,
    double a_ext) {
    auto in = state.request ();
    // if (in.ndim != 1) {
    //     throw std::runtime_error ("state must be a 1D array with shape (2,)");
    // }
    if (in.shape[0] != 2 || in.ndim != 1) {
        throw std::runtime_error ("state must have shape (2,)");
    }

    py::array_t<double> out (in.shape);
    // auto outb = out.request ();

    // const double* s = static_cast<const double*> (in.ptr);
    auto s = state.unchecked<1>();
    // double* y = static_cast<double*> (outb.ptr);
    auto y = out.mutable_unchecked<1>();

    y[0] = s[1];
    y[1] = a_ext - (spring_coef * s[0] + damping_coef * s[1]) / mass;

    return out;
}

py::array_t<double>
ode4 (py::array_t<double, py::array::c_style | py::array::forcecast> _k) {
    auto buf = _k.request ();

    // _k 期望是 (4,2): k1..k4，每一行是 [dx, dv]
    if (buf.ndim != 2 || buf.shape[0] != 4 || buf.shape[1] != 2) {
        throw std::runtime_error ("_k must have shape (4, 2)");
    }

    py::array_t<double> out ({ 2 });
    auto k = _k.unchecked<2> ();
    auto y = out.mutable_unchecked<1> ();

    y (0) = (k (0, 0) + 2.0 * k (1, 0) + 2.0 * k (2, 0) + k (3, 0)) / 6.0;
    y (1) = (k (0, 1) + 2.0 * k (1, 1) + 2.0 * k (2, 1) + k (3, 1)) / 6.0;

    return out;
}

} // namespace SpringDamping

PYBIND11_MODULE (SpringDamping, m) {
    m.doc () = "Spring-Damping system backend implemented in C++ (pybind11)";
    py::class_<SpringDamping::SpringDampingBackend> (m, "SpringDampingBackend")
        .def (py::init<double, double, double> (), py::arg ("mass"),
              py::arg ("spring_coef"), py::arg ("damping_coef"))
        .def ("state_equation",
              &SpringDamping::SpringDampingBackend::state_equation,
              py::arg ("state"), py::arg ("a_ext"));
    m.def ("ode4", &SpringDamping::ode4, py::arg ("_k"));
}
