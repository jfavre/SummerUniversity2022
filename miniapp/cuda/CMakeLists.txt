cmake_minimum_required(VERSION 3.13)
project (MiniAppCuda CUDA C CXX)

include (GNUInstallDirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

# since we use C++11 in this example.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_SRCS
stats.cu  data.cu  operators.cu  linalg.cu    main.cu
)


#include(FindCUDA/select_compute_arch)
#CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
#string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
#string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
#string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
#SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
#set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
      
find_package(OpenMP)

set(INSITU None CACHE STRING "Enable in-situ support")
set_property(CACHE INSITU PROPERTY STRINGS None Catalyst Ascent)

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/nvvm/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/extras/Debugger/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/extras/CUPTI/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/math_libs/11.2/include")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-L/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/math_libs/11.2/lib64")

if(INSITU STREQUAL "None")
  add_executable(miniapp ${CUDA_SRCS})
#  target_compile_options(miniapp PRIVATE "${CRAY_HPC_SDK_INCLUDE_OPTS}")
  target_link_libraries(miniapp PRIVATE cublas OpenMP::OpenMP_CXX)
elseif(INSITU STREQUAL "Catalyst")
  find_package(catalyst REQUIRED PATHS "${ParaView_DIR}/catalyst")
  # we only need C++ compilers if we're building the Adaptor
  enable_language(CXX)

  target_compile_definitions(miniapp PUBLIC USE_CATALYST=1)
  target_link_libraries(miniapp PRIVATE catalyst::catalyst)
elseif(INSITU STREQUAL "Ascent")
  find_package(Ascent REQUIRED PATHS "/apps/daint/UES/Ascent/ascent-install/lib/cmake/ascent")
  add_executable(miniapp ${CUDA_SRCS} AscentAdaptor.cu)
  #  target_compile_options(miniapp PRIVATE ${CRAY_HPC_SDK_INCLUDE_OPTS})
  target_compile_definitions(miniapp PRIVATE USE_ASCENT=1)
  target_link_libraries(miniapp PRIVATE cublas OpenMP::OpenMP_CXX ascent::ascent )
endif()
