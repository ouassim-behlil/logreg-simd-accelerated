cmake_minimum_required(VERSION 3.15)
project(logreg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Compiler flags per platform ----
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# ---- Source files (shared between C++ exe and Python module) ----
set(LIB_SOURCES
    logreg/LogisticRegression.cpp
    logreg/dispatcher.cpp
    logreg/dot_product.cpp
    logreg/vect_sigmoid.cpp
    utils/aligned_alloc.cpp
)

# ---- Static library (used by both targets) ----
add_library(logreg_core STATIC ${LIB_SOURCES})
target_include_directories(logreg_core PUBLIC logreg/include)

# ---- C++ executable ----
add_executable(main main.cpp)
target_link_libraries(main PRIVATE logreg_core)

# ---- Python module (optional – only if pybind11 is found) ----
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(logreg_py bindings/py_logreg.cpp)
    target_link_libraries(logreg_py PRIVATE logreg_core)
    set_target_properties(logreg_py PROPERTIES OUTPUT_NAME logreg)
    message(STATUS "pybind11 found – Python module will be built")
else()
    message(STATUS "pybind11 not found – skipping Python module")
endif()
