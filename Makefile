
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O3 -march=native
INCLUDES = -Ilogreg/include

# Source files
SOURCES = main.cpp \
		  logreg/dispatcher.cpp \
		  logreg/dot_product.cpp \
		  logreg/vect_exp.cpp

# Object files
OBJS = $(SOURCES:.cpp=.o)

# Executable
TARGET = main

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
