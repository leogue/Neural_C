cmake_minimum_required(VERSION 4.2)
project(Neural_C C)

set(CMAKE_C_STANDARD 11)

enable_testing()

add_executable(Neural main.c
        src/matrix.c
        include/matrix.h)

target_include_directories(Neural PRIVATE include)
target_link_libraries(Neural PRIVATE m)

add_executable(NeuralTests
        tests/test_matrix.c
        src/matrix.c
        include/matrix.h)

target_include_directories(NeuralTests PRIVATE include)
target_link_libraries(NeuralTests PRIVATE m)

add_test(NAME NeuralTests COMMAND NeuralTests)
