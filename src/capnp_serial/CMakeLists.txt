project("Cpnp serialized datastructures" CXX)
cmake_minimum_required(VERSION 3.1)

find_package(CapnProto REQUIRED)
add_definitions(${CAPNP_DEFINITIONS})

set(CAPNPC_OUTPUT_DIR ${CMAKE_BINARY_DIR}/capnp)
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})

capnp_generate_cpp(SHEDDER_CAPNP_SRCS SHEDDER_CAPNP_HDRS mapping.capnp)
add_library(mappingcap STATIC ${SHEDDER_CAPNP_SRCS} ${SHEDDER_CAPNP_HDRS})
target_link_libraries(mappingcap PUBLIC ${CAPNP_LIBRARIES_LITE})
target_include_directories(mappingcap PUBLIC ${CAPNPC_OUTPUT_DIR} ${CAPNP_INCLUDE_DIRS})

capnp_generate_cpp(MSGS_CAPNP_SRCS MSGS_CAPNP_HDRS messages.capnp)
add_library(messagescap STATIC ${MSGS_CAPNP_SRCS} ${MSGS_CAPNP_HDRS})
add_dependencies(messagescap mappingcap)
target_link_libraries(messagescap PUBLIC ${CAPNP_LIBRARIES_LITE})
target_include_directories(messagescap PUBLIC ${CAPNPC_OUTPUT_DIR} ${CAPNP_INCLUDE_DIRS})
