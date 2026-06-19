#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ThreadSafeQueue.h"
#include "llama_main.h"
namespace py = pybind11;

using StringQueue = ThreadSafeQueue<std::string>;

PYBIND11_MODULE(_llama, m) {
    m.doc() = "Python bindings for llama and queue";

   py::class_<StringQueue>(m, "StringQueue")
    .def(py::init<>())
    .def("push", [](StringQueue &q, const std::string &s){
        q.push(s);                                         // calls the lvalue overload
    })
    .def("push_move", [](StringQueue &q, std::string s){
        q.push(std::move(s));                              // calls the rvalue overload
    })
    .def("try_pop", [](StringQueue &q) -> py::object {
        auto maybe = q.try_pop();
        // if (maybe)
        //     return py::cast(*maybe);
        if (maybe)
            return py::bytes(*maybe);
        return py::none();  // return Python None
    })
    .def("pop", &StringQueue::pop)
    .def("snapshot", &StringQueue::snapshot)
    .def("empty", &StringQueue::empty);


    m.def(
        "main",
        &wrapped_main,
        py::arg("input_queue"),
        py::arg("output_queue"),    // 1️⃣ the ThreadSafeQueue<std::string> you created in Python
        py::arg("args")      // 2️⃣ the list of CLI‐style arguments
    );

}
