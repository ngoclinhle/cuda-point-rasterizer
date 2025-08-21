# Minimum CMake version
cmake_minimum_required(VERSION 3.18)

# Project name and languages
project(pcd_rasterizer LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


include(FetchContent)


# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)

# Fetch Eigen 3.4
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(Eigen3)

pkg_check_modules(PDAL REQUIRED pdal)


find_package(OpenGL REQUIRED)


find_package(glfw3 REQUIRED)


FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.4  # Use stable version
)
FetchContent_MakeAvailable(imgui)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PDAL_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
add_compile_definitions(EIGEN_NO_DEBUG)
add_compile_definitions(EIGEN_DEFAULT_DENSE_INDEX_TYPE=int)
add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ImGui source files
set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
)

# ImGui backends for GLFW and OpenGL3
set(IMGUI_BACKENDS
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# Create ImGui library
add_library(imgui STATIC ${IMGUI_SOURCES} ${IMGUI_BACKENDS})
target_include_directories(imgui PUBLIC 
    ${imgui_SOURCE_DIR} 
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# Create main executable
add_executable( ${PROJECT_NAME} 
    src/main.cpp
    src/main_app.cpp
    src/point_cloud.cpp
    src/rasterization.cu
    src/camera.cpp
    src/camera_controller.cpp
    src/renderer.cpp
    src/ui.cpp
)

target_include_directories( ${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PDAL_INCLUDE_DIRS}
    ${Eigen3_SOURCE_DIR}
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75)

target_link_libraries(${PROJECT_NAME}
    ${PDAL_LIBRARIES}
    ${OPENGL_LIBRARIES}
    glfw
    ${CUDA_LIBRARIES}
    imgui
    Eigen3::Eigen
)

# Compiler definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PDAL_DEFINITIONS})

# Compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE ${PDAL_CFLAGS_OTHER})

# CUDA-specific settings
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ImGui test executable
add_executable(imgui_test tests/imgui_test.cpp)
target_link_libraries(imgui_test PRIVATE imgui glfw OpenGL::GL dl)
target_compile_features(imgui_test PRIVATE cxx_std_17)

# PDAL test executable
add_executable(test_pdal tests/test_pdal.cpp)

target_include_directories(test_pdal PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PDAL_INCLUDE_DIRS}
)

target_link_libraries(test_pdal
    ${PDAL_LIBRARIES}
)

target_compile_definitions(test_pdal PRIVATE ${PDAL_DEFINITIONS})
target_compile_options(test_pdal PRIVATE ${PDAL_CFLAGS_OTHER})

# Renderer unit test executable
add_executable(test_renderer tests/test_renderer.cpp
    src/point_cloud.cpp
    src/rasterization.cu
    src/camera.cpp
    src/renderer.cpp
)

target_include_directories(test_renderer PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PDAL_INCLUDE_DIRS}
    ${Eigen3_SOURCE_DIR}
)

set_property(TARGET test_renderer PROPERTY CUDA_ARCHITECTURES 75)

target_link_libraries(test_renderer
    ${PDAL_LIBRARIES}
    OpenGL::GL
    glfw
    CUDA::cudart
    Eigen3::Eigen
)

target_compile_definitions(test_renderer PRIVATE ${PDAL_DEFINITIONS})
target_compile_options(test_renderer PRIVATE ${PDAL_CFLAGS_OTHER}) 