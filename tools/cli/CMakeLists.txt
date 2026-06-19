set(TARGET llama-cli)
add_executable(${TARGET} cli.cpp)
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

include_directories(../server)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

if(LLAMA_BUILD_CLI_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(llama_cpp_cli NB_STATIC
        cli.cpp
        llama_cpp_cli_nanobind.cpp
        llama_cli_engine.cpp
    )
    target_link_libraries(llama_cpp_cli PRIVATE server-context llama-common ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(llama_cpp_cli PRIVATE cxx_std_17)
    include_directories(../server)
endif()
