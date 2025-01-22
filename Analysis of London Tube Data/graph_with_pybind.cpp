#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "test.cpp"  // Include your existing code here

namespace py = pybind11;

PYBIND11_MODULE(graph_module, m) {
    m.doc() = "C++ graph module with DFS functionality";

    // Bind the Graph class
    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("addEdge", &Graph::addEdge)
        .def("printGraph", &Graph::printGraph);

    // Bind the depth_first_search function
    m.def("depth_first_search", &depth_first_search, 
          "Perform DFS on the graph",
          py::arg("graph"), py::arg("initial"), py::arg("goal"), py::arg("check_reverse") = false);
}
