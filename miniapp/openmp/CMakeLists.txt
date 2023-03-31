cmake_minimum_required(VERSION 3.13)
project (MiniApp C CXX)

include (GNUInstallDirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

# since we use C++11 in this example.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(C_SRCS
stats.cpp
data.cpp
operators.cpp
linalg.cpp
main.cpp
)

find_package(OpenMP)

set(INSITU None CACHE STRING "Enable in-situ support")
set_property(CACHE INSITU PROPERTY STRINGS None Catalyst Ascent)

if(INSITU STREQUAL "None")
  add_executable(miniapp ${C_SRCS})
  target_link_libraries(miniapp PUBLIC OpenMP::OpenMP_CXX)
elseif(INSITU STREQUAL "Catalyst")
  add_executable(miniapp ${C_SRCS} CatalystAdaptor.cpp)
  find_package(catalyst REQUIRED PATHS "${EBROOTCATALYST}/lib/cmake/catalyst-2.0")
  target_compile_definitions(miniapp PRIVATE USE_CATALYST=1)
  target_link_libraries(miniapp PRIVATE OpenMP::OpenMP_CXX catalyst::catalyst)
elseif(INSITU STREQUAL "Ascent")
  find_package(Ascent REQUIRED PATHS "/apps/daint/UES/Ascent/ascent-install/lib/cmake/ascent")
  add_executable(miniapp ${C_SRCS} AscentAdaptor.cpp)
  target_compile_definitions(miniapp PRIVATE USE_ASCENT=1)
  target_link_libraries(miniapp PRIVATE OpenMP::OpenMP_CXX ascent::ascent )
endif()
