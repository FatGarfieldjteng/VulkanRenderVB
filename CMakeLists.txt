cmake_minimum_required(VERSION 3.22)
project(VulkanRenderVB VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# --- Volk (Vulkan meta-loader) ---
FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        vulkan-sdk-1.3.290.0
)

# --- GLFW ---
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
if(UNIX AND NOT APPLE)
    set(GLFW_BUILD_WAYLAND ON CACHE BOOL "" FORCE)
    set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
endif()
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)

# --- spdlog ---
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.14.1
)

# --- GLM ---
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)

# --- VMA (Vulkan Memory Allocator) ---
FetchContent_Declare(VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.1.0
)

# --- tinygltf (header-only glTF 2.0 loader, bundles stb_image) ---
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.9.3
)

# --- Dear ImGui (docking branch) ---
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        docking
)

FetchContent_MakeAvailable(volk glfw spdlog glm VulkanMemoryAllocator tinygltf imgui)

# --- NRD (NVIDIA Real-time Denoisers) ---
set(NRD_NRI ON CACHE BOOL "" FORCE)
set(NRD_NORMAL_ENCODING "0" CACHE STRING "" FORCE)  # RGBA8 to match our prepack
set(NRD_EMBEDS_DXIL_SHADERS OFF CACHE BOOL "" FORCE)
set(NRD_EMBEDS_DXBC_SHADERS OFF CACHE BOOL "" FORCE)
set(NRD_EMBEDS_SPIRV_SHADERS ON CACHE BOOL "" FORCE)
set(NRD_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
FetchContent_Declare(nrd
    GIT_REPOSITORY https://github.com/NVIDIA-RTX/NRD.git
    GIT_TAG        v4.17.1
)
FetchContent_MakeAvailable(nrd)

# --- Shader compilation ---
find_program(GLSLANG_VALIDATOR glslangValidator
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
          "/usr/bin"
)
if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found. Install Vulkan SDK and set VULKAN_SDK environment variable.")
endif()

file(GLOB SHADER_SOURCES
    "${CMAKE_SOURCE_DIR}/shaders/*.vert"
    "${CMAKE_SOURCE_DIR}/shaders/*.frag"
    "${CMAKE_SOURCE_DIR}/shaders/*.comp"
)

file(GLOB RT_SHADER_SOURCES
    "${CMAKE_SOURCE_DIR}/shaders/*.rgen"
    "${CMAKE_SOURCE_DIR}/shaders/*.rmiss"
    "${CMAKE_SOURCE_DIR}/shaders/*.rchit"
    "${CMAKE_SOURCE_DIR}/shaders/*.rahit"
    "${CMAKE_SOURCE_DIR}/shaders/*.rint"
    "${CMAKE_SOURCE_DIR}/shaders/*.rcall"
)

file(GLOB SHADER_INCLUDES
    "${CMAKE_SOURCE_DIR}/shaders/*.glsl"
)

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")

foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SHADER_OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT  ${SHADER_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUTPUT_DIR}"
        COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.2 "${SHADER}" -o "${SHADER_OUTPUT}"
        DEPENDS ${SHADER}
        COMMENT "Compiling shader ${SHADER_NAME}"
    )
    list(APPEND SHADER_OUTPUTS ${SHADER_OUTPUT})
endforeach()

foreach(SHADER ${RT_SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SHADER_OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT  ${SHADER_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUTPUT_DIR}"
        COMMAND ${GLSLANG_VALIDATOR} -V --target-env spirv1.4 "${SHADER}" -o "${SHADER_OUTPUT}"
        DEPENDS ${SHADER} ${SHADER_INCLUDES}
        COMMENT "Compiling RT shader ${SHADER_NAME}"
    )
    list(APPEND SHADER_OUTPUTS ${SHADER_OUTPUT})
endforeach()

add_custom_target(Shaders ALL DEPENDS ${SHADER_OUTPUTS})

# --- Main executable ---
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h")

set(IMGUI_SOURCES
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/imgui_demo.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp"
)

add_executable(${PROJECT_NAME} ${SOURCES} ${IMGUI_SOURCES})
add_dependencies(${PROJECT_NAME} Shaders)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${vulkanmemoryallocator_SOURCE_DIR}/include"
    "${tinygltf_SOURCE_DIR}"
    "${imgui_SOURCE_DIR}"
    "${imgui_SOURCE_DIR}/backends"
)

# NRI is fetched by NRD as a dependency; path is in build/_deps/nri-src
set(NRI_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/nri-src/Include")
if(NOT EXISTS "${NRI_INCLUDE_DIR}/NRI.h")
    set(NRI_INCLUDE_DIR "${nri_SOURCE_DIR}/Include")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${nrd_SOURCE_DIR}/Include
    ${nrd_SOURCE_DIR}/Integration
    ${NRI_INCLUDE_DIR}
    ${NRI_INCLUDE_DIR}/Extensions
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    volk
    glfw
    glm::glm
    spdlog::spdlog
    NRI
    NRD
    NRDIntegration
)

if(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads dl)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:VRB_DEBUG>
    VMA_STATIC_VULKAN_FUNCTIONS=0
    VMA_DYNAMIC_VULKAN_FUNCTIONS=1
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    GLFW_INCLUDE_NONE
    IMGUI_IMPL_VULKAN_NO_PROTOTYPES
)

if(WIN32)
    # VS debugger working directory = build dir (so it finds shaders/)
    set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

    # Copy NRI.dll to executable dir (NRI is built as shared lib by NRD; loader needs it next to .exe)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:NRI>"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/"
        COMMENT "Copying NRI.dll to output directory"
    )

    # Copy compiled shaders to executable dir (so app finds them when run from Release/)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${SHADER_OUTPUT_DIR}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
        COMMENT "Copying shaders to output directory"
    )

    # Organize source files in VS solution filters mirroring the directory tree
    source_group(TREE "${CMAKE_SOURCE_DIR}/src" PREFIX "Source" FILES ${SOURCES})
endif()
