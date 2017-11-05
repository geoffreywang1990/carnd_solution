project(markov_localization)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(sources src/main.cpp 
	    src/bayesianFilter.cpp
            src/bayesianFilter.h
            src/help_functions.h
            src/map.h
            src/measurement_package.h)
 
add_executable(markov_localization src/main.cpp ${sources})
