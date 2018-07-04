cmake_minimum_required(VERSION 2.8)

project(rgb_imfit)
set(CMAKE_C_FLAGS "-g -Wall")
set(USE_GLEW true CACHE BOOL "Use GLEW extension loader library")

if(APPLE)
  include_directories(/opt/local/include)
  link_directories(/opt/local/lib)
endif(APPLE)

find_package(glfw3 3 REQUIRED)
find_package(OpenGL REQUIRED)

include(FindPkgConfig)

if (USE_GLEW)
  pkg_check_modules(GLEW glew)
endif()

if (GLEW_FOUND)
  add_definitions(-DST_GLFW_USE_GLEW)
endif(GLEW_FOUND)

if (CURL_FOUND)
  add_definitions(-DST_GLFW_USE_CURL)
endif(CURL_FOUND)

set(SOURCES
  src/rgb_imfit.c
  src/require.c src/buffer.c src/image.c
  src/gl_util.c src/framebuffer.c src/pcg_tiny.c)  

add_executable(rgb_imfit ${SOURCES})

target_link_libraries(rgb_imfit glfw
  ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} png jpeg m)


