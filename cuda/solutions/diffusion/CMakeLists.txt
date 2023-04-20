cmake_minimum_required(VERSION 3.13)
project (diffusion2d CUDA C CXX)

include (GNUInstallDirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

# since we use C++11 in this example.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_SRCS
  diffusion2d.cu
)
include_directories(../include)

add_executable(diffusion2d ${CUDA_SRCS})
set_target_properties(diffusion2d PROPERTIES CUDA_ARCHITECTURES "60;75;86")

set(INSITU None CACHE STRING "Enable in-situ support")
set_property(CACHE INSITU PROPERTY STRINGS None Catalyst Ascent)

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/nvvm/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/extras/Debugger/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/extras/CUPTI/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/math_libs/11.2/include")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-L/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/math_libs/11.2/lib64")

if(INSITU STREQUAL "None")
  target_link_libraries(diffusion2d PRIVATE cublas )
elseif(INSITU STREQUAL "Catalyst")
  find_package(catalyst 2.0 REQUIRED PATHS "${EBROOTCATALYST}/lib/cmake/catalyst-2.0")
  target_compile_definitions(diffusion2d PUBLIC USE_CATALYST=1)
  target_link_libraries(diffusion2d PRIVATE cublas catalyst::catalyst)
elseif(INSITU STREQUAL "Ascent")
  find_package(Ascent REQUIRED PATHS "/apps/daint/UES/Ascent/ascent-install/lib/cmake/ascent")
  #  target_compile_options(miniapp PRIVATE ${CRAY_HPC_SDK_INCLUDE_OPTS})
  target_compile_definitions(diffusion2d PRIVATE USE_ASCENT=1)
  target_link_libraries(diffusion2d PRIVATE cublas ascent::ascent )
endif()
