// bindings/py_logreg.cpp  –  pybind11 ↔ NumPy bridge
//
// Every NumPy buffer that enters the C++ layer is either already
// 32-byte aligned (common on modern NumPy ≥ 1.20) or is copied
// transparently by the LogisticRegression class into aligned
// scratch storage.  The Python user never has to think about it.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "LogisticRegression.hpp"
#include "logreg_dispatcher.hpp"
#include "simd_fn.hpp"

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

// ---------------------------------------------------------------
//  Module definition
// ---------------------------------------------------------------
PYBIND11_MODULE(logreg, m)
{
    m.doc() = "SIMD-accelerated binary logistic regression";

    // Auto-detect best SIMD kernels once at import time.
    init_kernels();

    py::class_<LogisticRegression>(m, "LogisticRegression",
        "Binary logistic regression classifier.\n\n"
        "Internally uses SIMD-accelerated dot products and sigmoid.\n"
        "All data is automatically copied to 32-byte-aligned buffers\n"
        "when necessary, so NumPy arrays of any alignment are accepted.")

        // ---- constructor ------------------------------------------------
        .def(py::init<int, float, int>(),
             py::arg("n_features"),
             py::arg("lr")     = 0.1f,
             py::arg("epochs") = 1000,
             "Create a logistic regression model.\n\n"
             "Parameters\n"
             "----------\n"
             "n_features : int\n"
             "    Number of input features (columns of X).\n"
             "lr : float, optional\n"
             "    Learning rate (default 0.1).\n"
             "epochs : int, optional\n"
             "    Number of full passes over the training set (default 1000).")

        // ---- train ------------------------------------------------------
        .def("train",
             [](LogisticRegression& self,
                py::array_t<float,   py::array::c_style | py::array::forcecast> X,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> Y)
             {
                 auto xbuf = X.request();
                 auto ybuf = Y.request();

                 if (xbuf.ndim != 2)
                     throw std::runtime_error(
                         "X must be 2-D [n_samples x n_features]");
                 if (ybuf.ndim != 1)
                     throw std::runtime_error(
                         "Y must be 1-D [n_samples]");
                 if (xbuf.shape[1] != self.get_n_features())
                     throw std::runtime_error(
                         "X.shape[1] does not match n_features");
                 if (xbuf.shape[0] != ybuf.shape[0])
                     throw std::runtime_error(
                         "X and Y must have the same number of samples");

                 self.train(
                     static_cast<const float*>(xbuf.ptr),
                     static_cast<const int*>(ybuf.ptr),
                     static_cast<int>(xbuf.shape[0]));
             },
             py::arg("X"), py::arg("Y"),
             "Train on X [n_samples x n_features] and Y [n_samples] in {0,1}.")

        // ---- predict (single sample) ------------------------------------
        .def("predict",
             [](const LogisticRegression& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> x)
             {
                 auto buf = x.request();
                 if (buf.ndim != 1 ||
                     buf.shape[0] != self.get_n_features())
                     throw std::runtime_error(
                         "x must be 1-D with length n_features");

                 return self.predict(
                     static_cast<const float*>(buf.ptr));
             },
             py::arg("x"),
             "Return P(y=1 | x) for a single sample.")

        // ---- predict_class (single sample) ------------------------------
        .def("predict_class",
             [](const LogisticRegression& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> x)
             {
                 auto buf = x.request();
                 if (buf.ndim != 1 ||
                     buf.shape[0] != self.get_n_features())
                     throw std::runtime_error(
                         "x must be 1-D with length n_features");

                 return self.predict_class(
                     static_cast<const float*>(buf.ptr));
             },
             py::arg("x"),
             "Return the predicted class (0 or 1) for a single sample.")

        // ---- predict_batch ----------------------------------------------
        .def("predict_batch",
             [](const LogisticRegression& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> X)
             {
                 auto xbuf = X.request();
                 if (xbuf.ndim != 2)
                     throw std::runtime_error(
                         "X must be 2-D [n_samples x n_features]");
                 if (xbuf.shape[1] != self.get_n_features())
                     throw std::runtime_error(
                         "X.shape[1] does not match n_features");

                 const int n = static_cast<int>(xbuf.shape[0]);

                 py::array_t<float> out(n);
                 self.predict_batch(
                     static_cast<const float*>(xbuf.ptr),
                     static_cast<float*>(out.request().ptr),
                     n);
                 return out;
             },
             py::arg("X"),
             "Return P(y=1 | x_i) for each row of X as a 1-D array.")

        // ---- predict_class_batch ----------------------------------------
        .def("predict_class_batch",
             [](const LogisticRegression& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> X)
             {
                 auto xbuf = X.request();
                 if (xbuf.ndim != 2)
                     throw std::runtime_error(
                         "X must be 2-D [n_samples x n_features]");
                 if (xbuf.shape[1] != self.get_n_features())
                     throw std::runtime_error(
                         "X.shape[1] does not match n_features");

                 const int n = static_cast<int>(xbuf.shape[0]);

                 py::array_t<int32_t> out(n);
                 self.predict_class_batch(
                     static_cast<const float*>(xbuf.ptr),
                     static_cast<int*>(out.request().ptr),
                     n);
                 return out;
             },
             py::arg("X"),
             "Return predicted class (0 or 1) for each row of X as a 1-D array.")

        // ---- properties -------------------------------------------------
        .def_property_readonly("n_features",
             &LogisticRegression::get_n_features,
             "Number of input features the model was created with.");
}
