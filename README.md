# VulkanRenderVB

A Vulkan 1.3 real-time rendering engine built from scratch in C++17, featuring PBR shading, GPU-driven rendering, a render graph, and a full post-processing stack.

## Supported Platforms

| Platform | Window System | Status |
|----------|--------------|--------|
| Windows 10/11 | Win32 | Supported |
| Ubuntu 22.04 / 24.04 | Wayland or X11 | Supported |
| WSL2 (Windows Subsystem for Linux) | Wayland / X11 (WSLg) | Supported |

## Features

### Rendering

- **PBR (Physically Based Rendering)** — Metallic-roughness workflow with albedo, normal, metallic/roughness, and emissive maps
- **Image-Based Lighting (IBL)** — Diffuse irradiance + specular prefiltered environment maps + BRDF LUT
- **Cascaded Shadow Maps** — Multi-cascade directional light shadows
- **Forward Shading** — HDR forward rendering with dynamic rendering (no render pass objects)
- **MSAA** — Runtime-selectable multisample anti-aliasing (1x / 2x / 4x / 8x) with sample-rate shading

### GPU-Driven Rendering

- **Indirect Draw** — `vkCmdDrawIndexedIndirect` with per-object SSBO
- **Mesh Pool** — Shared vertex/index buffers for all meshes
- **Frustum Culling** — Compute shader AABB-based frustum culling
- **Hi-Z Occlusion Culling** — Two-pass hierarchical Z-buffer occlusion culling with mip-chain reduction

### Post-Processing

- **Auto Exposure** — Histogram-based adaptive exposure with configurable EV range and adaptation speed
- **SSAO (GTAO)** — Ground Truth Ambient Occlusion with depth-aware bilateral blur
- **Bloom** — 6-level mip-chain downsample (13-tap + Karis average) / upsample (9-tap tent filter)
- **Tone Mapping** — ACES Filmic and AgX with tweakable parameters
- **Color Grading** — 3D LUT, vignette, film grain, chromatic aberration

### Render Graph

- Automatic resource dependency tracking and barrier insertion
- Barrier batching for optimal synchronization
- Pass ordering based on dependencies

### Debug & Profiling

- **Dear ImGui** integration (docking branch) for runtime parameter tuning
- **GPU Profiler** — Per-pass GPU timestamp queries
- **Pipeline Statistics** — Hardware counter visualization
- **Debug Visualization** — Depth, normals, AO, bloom, shadow cascades
- **Object Labeling** — VK_EXT_debug_utils markers for GPU captures

### Asset Loading

- **glTF 2.0** — `.gltf` and `.glb` via tinygltf
- **DDS Textures** — BC1/BC3/BC4/BC5/BC7 compressed textures via bcdec
- **PNG fallback** — Automatic PNG-to-DDS fallback for missing compressed textures
- **Tangent generation** — Automatic MikkTSpace-style tangent computation when absent

## Dependencies

All dependencies are fetched automatically via CMake `FetchContent`:

| Library | Version | Purpose |
|---------|---------|---------|
| [Volk](https://github.com/zeux/volk) | vulkan-sdk-1.3.290.0 | Vulkan meta-loader |
| [GLFW](https://github.com/glfw/glfw) | 3.4 | Window and input |
| [spdlog](https://github.com/gabime/spdlog) | 1.14.1 | Logging |
| [GLM](https://github.com/g-truc/glm) | 1.0.1 | Mathematics |
| [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | 3.1.0 | Vulkan memory allocation |
| [tinygltf](https://github.com/syoyo/tinygltf) | 2.9.3 | glTF loading (includes stb_image) |
| [Dear ImGui](https://github.com/ocornut/imgui) | docking | Debug UI |

## Prerequisites

### Windows

- **Vulkan SDK** 1.3+ installed (with `glslangValidator` in PATH)
- **CMake** 3.20+
- **Visual Studio 2026** (or 2022+) with C++17 support

### Ubuntu Linux

- **Ubuntu** 22.04 or 24.04
- **CMake** 3.20+
- **GCC** 11+ or **Clang** 14+ with C++17 support
- **Vulkan development packages** and **glslang-tools**
- **Wayland and X11 development libraries**

Install all required packages manually, or use the one-click setup script (see below):

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git pkg-config \
    libwayland-dev wayland-protocols libxkbcommon-dev \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    vulkan-tools vulkan-utility-libraries-dev vulkan-validationlayers \
    glslang-tools libvulkan-dev mesa-vulkan-drivers
```

### One-Click Setup Script

On a fresh Ubuntu machine, the setup script installs all dependencies, downloads Sponza assets, and builds the project:

```bash
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh
```

## Building

### Windows — Generate Visual Studio Project

```bash
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
```

This generates a Visual Studio 2026 solution at `build/VulkanRenderVB.sln`. Open it and build the `VulkanRenderVB` target.

For older Visual Studio versions, replace the generator string:

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
```

Shaders are compiled automatically as a build step — `glslangValidator` converts all GLSL shaders in `shaders/` to SPIR-V in `build/shaders/`.

### Ubuntu Linux — Build from Source

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Run from the build directory (so it finds `shaders/`):

```bash
cd build
./VulkanRenderVB
```

### WSL2 — Build and Run on Windows Subsystem for Linux

WSL2 with WSLg provides a native Wayland compositor and GPU passthrough, allowing the renderer to run with full GPU acceleration from within Windows.

**1. Install Ubuntu in WSL2** (from PowerShell):

```powershell
wsl --install -d Ubuntu-24.04
```

**2. Install dependencies** (inside WSL2):

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git pkg-config \
    libwayland-dev wayland-protocols libxkbcommon-dev \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    vulkan-tools vulkan-utility-libraries-dev vulkan-validationlayers \
    glslang-tools libvulkan-dev \
    mesa-vulkan-drivers
```

Or simply run `./setup_ubuntu.sh` which handles everything (dependencies, assets, build).

**3. Verify Vulkan works** (inside WSL2):

```bash
vulkaninfo --summary
```

You should see your GPU listed. If not, ensure your Windows GPU driver is up to date (the WSL2 Vulkan driver is provided by your Windows GPU driver).

**4. Build and run**:

```bash
cd /mnt/d/tengj/git/VulkanRenderVB    # or wherever your project is
cmake -S . -B build-wsl -DCMAKE_BUILD_TYPE=Release
cmake --build build-wsl -j$(nproc)
cd build-wsl
./VulkanRenderVB
```

The application window will appear on your Windows desktop via WSLg.

## Download Sponza Asset

The engine expects assets in the `assets/` directory.

### Windows

```bash
mkdir assets
cd assets
git clone https://github.com/KhronosGroup/glTF-Sample-Assets.git
xcopy /E /I "glTF-Sample-Assets\Models\Sponza\glTF" "Sponza"
rmdir /S /Q glTF-Sample-Assets
```

### Linux / WSL2

```bash
mkdir -p assets
cd assets
git clone https://github.com/KhronosGroup/glTF-Sample-Assets.git
cp -r glTF-Sample-Assets/Models/Sponza/glTF Sponza
rm -rf glTF-Sample-Assets
```

Or download manually from [Khronos glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Sponza) and place the files so that `assets/Sponza/Sponza.gltf` exists.

The engine searches for scenes in this order:
1. `assets/Sponza/Sponza.gltf`
2. `assets/Sponza.glb`
3. `assets/Bistro/Bistro.gltf`
4. `assets/DamagedHelmet.glb`
5. `assets/DamagedHelmet/DamagedHelmet.gltf`
6. `assets/model.glb`

If no model is found, a procedural fallback scene (ground plane + cubes) is generated.

## Command-Line Options

```
VulkanRenderVB [options]
  --scene <path>       Override scene path (glTF/glb)
  --benchmark          Run benchmark mode
  --frames <N>         Number of benchmark frames (default: 200)
  --no-gpu             Disable GPU-driven rendering
  --no-occlusion       Disable occlusion culling
```

## Project Structure

```
VulkanRenderVB/
├── src/
│   ├── Core/              Application, Window, Input, Logger, ThreadPool
│   ├── RHI/               Vulkan device, swapchain, command buffers, sync
│   ├── Resource/          Buffers, images, pipelines, shaders, descriptors
│   ├── RenderGraph/       Render graph, pass scheduling, barriers
│   │   └── Passes/        ForwardPass, ShadowPass, PostProcessPass, ...
│   ├── PostProcess/       AutoExposure, SSAO, Bloom, ToneMapping, ColorGrading
│   ├── GPU/               IndirectRenderer, MeshPool, HiZBuffer, ComputeCulling
│   ├── Scene/             ECS (Registry, ComponentPools), Camera
│   ├── Asset/             ModelLoader (glTF + DDS)
│   ├── Lighting/          CascadedShadowMap
│   ├── IBL/               IBLProcessor
│   ├── VisualUI/          DebugUI, ImGuiPass, GPUProfiler
│   └── Math/              AABB
├── shaders/               GLSL shaders (.vert, .frag, .comp)
├── CMakeLists.txt
└── README.md
```
