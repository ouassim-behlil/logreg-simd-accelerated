
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O3 -march=native
INCLUDES = -Ilogreg/include

# Source files (C++ executable)
SOURCES = main.cpp \
		  logreg/LogisticRegression.cpp \
		  logreg/dispatcher.cpp \
		  logreg/dot_product.cpp \
		  logreg/vect_sigmoid.cpp \
		  utils/aligned_alloc.cpp

# Object files
OBJS = $(SOURCES:.cpp=.o)

# Executable
TARGET = main

# ---- Python extension (pybind11) ----
PYBIND11_INCLUDES = $(shell python3 -m pybind11 --includes)
PYTHON_EXT_SUFFIX = $(shell python3-config --extension-suffix)
PY_MODULE         = logreg$(PYTHON_EXT_SUFFIX)

PY_SOURCES = bindings/py_logreg.cpp \
             logreg/LogisticRegression.cpp \
             logreg/dispatcher.cpp \
             logreg/dot_product.cpp \
             logreg/vect_sigmoid.cpp \
             utils/aligned_alloc.cpp

.PHONY: all clean python

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Build the Python extension module in-place
python:
	$(CXX) -shared -fPIC $(CXXFLAGS) $(INCLUDES) $(PYBIND11_INCLUDES) \
	    $(PY_SOURCES) -o $(PY_MODULE)

clean:
	rm -f $(OBJS) $(TARGET) logreg*.so
