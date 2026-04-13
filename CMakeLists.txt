cmake_minimum_required(VERSION 4.2)
project(Neural_C C)

set(CMAKE_C_STANDARD 11)

enable_testing()

add_executable(Neural main.c
        src/matrix.c
        include/matrix.h
        include/layer.h
        src/layer.c
        include/network.h
        src/network.c)

target_include_directories(Neural PRIVATE include)
target_link_libraries(Neural PRIVATE m)

add_executable(NeuralTests
        tests/test_matrix.c
        src/matrix.c
        include/matrix.h)

add_executable(LayerTests
        tests/test_layer.c
        src/matrix.c
        src/layer.c
        include/matrix.h
        include/layer.h
        include/network.h
        src/network.c)

target_include_directories(NeuralTests PRIVATE include)
target_link_libraries(NeuralTests PRIVATE m)
target_include_directories(LayerTests PRIVATE include)
target_link_libraries(LayerTests PRIVATE m)

add_test(NAME NeuralTests COMMAND NeuralTests)
add_test(NAME LayerTests COMMAND LayerTests)
