cmake_minimum_required(VERSION 3.0)
project(SampleOnnxMNIST CXX C)

set(APP_NAME SampleOnnxMNIST)
add_executable(${APP_NAME} sampleOnnxMNIST.cpp)

install(TARGETS ${APP_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)