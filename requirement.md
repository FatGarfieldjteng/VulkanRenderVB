# Modern Vulkan Real-Time Rendering Engine — Feature Requirements

## 1. Technical Requirements

- **C++17 Standard** — All source code should conform to the C++17 standard; prefer the C++ standard library where possible
- **Running Environment** — Windows platform only; use CMake to generate a Visual Studio 2026 project and manage third-party dependencies
- **Log System** — Use spdlog to log errors and warnings to the console window
- **Coding Style Guide** — Class names start with a capital letter (e.g. `VulkanBuffer`). Member functions use PascalCase. Member variables are prefixed with `m` and use camelCase (e.g. `mVertexCount`). Non-member variables use camelCase. Member variables should be initialized at declaration whenever possible. Functions should have descriptive comments
- **Volk** — Use Volk for dynamic Vulkan function loading, avoiding the need to link against the Vulkan loader at compile time
- **GLFW** — Use GLFW for window creation, input handling, and message processing

## 2. Core Vulkan Infrastructure

- **Vulkan Instance & Device Management** — Instance creation, physical/logical device selection, queue family discovery, feature/extension negotiation (target Vulkan 1.3+ to get dynamic rendering in core)
- **Memory Allocator** — Integration with VMA (Vulkan Memory Allocator) or a custom sub-allocator with support for dedicated allocations, memory pools, and defragmentation
- **Swapchain Management** — Creation, recreation on resize, presentation mode selection (FIFO, Mailbox, Immediate), HDR surface format support
- **Command Buffer Management** — Per-frame command pool/buffer cycling, secondary command buffers for parallel recording, transient command pools for one-shot uploads
- **Synchronization** — Timeline semaphores (core in 1.2), fences for CPU-GPU sync, pipeline barriers, frame-in-flight management (double/triple buffering)
- **Dynamic Rendering** — Use `VK_KHR_dynamic_rendering` (core in 1.3) instead of legacy render passes/framebuffers for simpler, more flexible rendering

## 3. Resource Management

- **Buffer & Image Abstraction** — Typed wrappers around `VkBuffer`/`VkImage` with automatic layout transitions, upload staging, and lifetime tracking
- **Descriptor Management** — Bindless descriptor model using `VK_EXT_descriptor_indexing` with large descriptor arrays for textures and buffers, eliminating per-draw descriptor set switches
- **Shader Management** — SPIR-V compilation pipeline (GLSL/HLSL via glslang/DXC), shader reflection for automatic pipeline layout generation, shader hot-reloading for development
- **Pipeline Management** — Pipeline caching (`VkPipelineCache`), background compilation, pipeline derivatives, and a material-oriented pipeline abstraction
- **Transfer / Upload System** — Async transfer queue utilization, staging ring buffer for per-frame uploads, GPU timeline-based release of staging memory

## 4. Render Graph / Frame Graph

A modern engine needs a render graph system to:

- Declare render passes and their resource dependencies (inputs/outputs)
- Automatically compute execution order via topological sort
- Insert memory barriers and layout transitions automatically
- Enable transient (aliased) resource allocation to minimize memory usage
- Support async compute pass scheduling alongside graphics passes
- Facilitate easy addition/removal of rendering features without manual synchronization

## 5. Scene Management & GPU-Driven Rendering

- **Scene Graph / ECS** — Entity-Component-System or a flat scene representation with spatial hierarchy
- **GPU-Driven Indirect Rendering** — Upload all mesh data to large GPU buffers; use `vkCmdDrawIndexedIndirect` / `vkCmdDrawIndexedIndirectCount` to draw everything in minimal draw calls
- **GPU Frustum & Occlusion Culling** — Compute shader-based culling using the previous frame's depth buffer (Hi-Z occlusion culling) and frustum tests, outputting indirect draw commands
- **Mesh LOD System** — Automatic LOD selection on the GPU, potentially with seamless transitions (cross-fade or morphing)
- **Virtual Geometry (Nanite-style)** — Cluster-based mesh representation with GPU-driven hierarchical culling and software rasterization for small triangles (advanced, optional)
- **Instance Batching & Merging** — Automatic instancing of identical meshes, mesh merging for static geometry

## 6. Lighting & Shadows

- **Punctual Lights** — Point, spot, and directional lights with physically-based attenuation
- **Clustered / Tiled Forward+ Shading** — Subdivide the view frustum into clusters/tiles and assign lights per cluster for efficient many-light rendering
- **Stochastic Lighting (MegaLights-style)** — For scenes with thousands of lights, use stochastic sampling with temporal accumulation
- **Shadow Mapping** — Cascaded Shadow Maps (CSM) for directional lights, cube shadow maps for point lights, single-pass shadow maps for spot lights
- **Virtual Shadow Maps** — Clipmap-based virtual shadow map system for high-resolution shadows across large scenes (advanced)
- **Volumetric Lighting / Fog** — Froxel-based volumetric fog with light scattering
- **Area Lights** — LTC (Linearly Transformed Cosines) based area light shading

## 7. Materials & Shading

- **Physically Based Rendering (PBR)** — Standard metallic-roughness or specular-glossiness workflow, energy-conserving BRDFs (GGX/Smith), image-based lighting (IBL) with split-sum approximation
- **Material System** — Data-driven material definitions, material parameter buffers, support for material layering and blending
- **Texture Streaming** — Virtual texturing or explicit mip streaming to manage GPU memory for large texture sets
- **Normal / Parallax / Displacement Mapping** — Tangent-space normal maps, parallax occlusion mapping, optional tessellation-based displacement
- **Subsurface Scattering** — Screen-space or ray-traced SSS for skin, wax, foliage
- **Anisotropic Shading** — For hair, brushed metal, etc.
- **Clear Coat, Sheen, Thin Film** — Extended material models for automotive paint, fabric, soap bubbles

## 8. Global Illumination

- **Screen-Space GI (SSGI)** — Screen-space ray marching for approximate indirect diffuse
- **Irradiance Probes / DDGI** — Dynamic Diffuse Global Illumination using ray-traced irradiance and visibility probes on a 3D grid, updated incrementally each frame
- **Radiance Cascades / Voxel GI** — Cascaded radiance probes or voxel cone tracing for multi-bounce GI
- **Reflection Probes** — Baked and real-time cubemap probes with parallax correction
- **Light Probe Interpolation** — Spherical harmonics or octahedral encoding for efficient probe storage and blending

## 9. Ray Tracing (Vulkan RT)

- **Acceleration Structure Management** — BLAS/TLAS build, update, and compaction using `VK_KHR_acceleration_structure`
- **Ray-Traced Shadows** — 1 spp stochastic shadows with denoising for soft area light shadows
- **Ray-Traced Reflections** — Hybrid SSR + RT reflections fallback
- **Ray-Traced Ambient Occlusion** — Accurate multi-bounce AO
- **Ray Queries in Fragment Shaders** — `VK_KHR_ray_query` for inline ray tracing in rasterization shaders (better mobile/bandwidth profile)
- **Path Tracing Reference Mode** — Full path tracer for ground-truth comparison and baking

## 10. Post-Processing

- **Tone Mapping** — ACES, AgX, or Khronos PBR Neutral tone mapping, HDR display output support
- **Bloom** — Physically-based bloom with downscale/upscale chain and energy conservation
- **Depth of Field** — Bokeh DoF with near/far CoC computation
- **Motion Blur** — Per-object and camera motion blur using velocity buffers
- **Screen-Space Reflections (SSR)** — Hierarchical ray marching in screen space
- **Screen-Space Ambient Occlusion** — GTAO, HBAO+, or similar
- **Temporal Anti-Aliasing (TAA)** — Jittered projection with history reprojection, neighborhood clamping, and velocity-based rejection
- **Upscaling** — FSR 2/3 or DLSS integration for temporal upscaling from lower internal resolution
- **Color Grading & LUTs** — 3D LUT-based color grading pipeline
- **Chromatic Aberration, Vignette, Film Grain** — Cinematic post effects
- **Auto Exposure / Eye Adaptation** — Histogram-based or average luminance-based

## 11. Compute & Async

- **Async Compute Pipeline** — Overlap compute work (culling, particle simulation, blur passes) with graphics on separate queue
- **GPU Particle System** — Compute shader-based particle emission, simulation (forces, collisions), sorting for transparency, and rendering
- **GPU Skinning & Animation** — Compute-based skeletal skinning, blend shapes, animation blending on GPU
- **Hi-Z Buffer Generation** — Compute-based hierarchical depth mipchain for occlusion culling

## 12. Terrain & Environment

- **Terrain Rendering** — Clipmap or quadtree-based terrain with GPU tessellation, virtual texturing for terrain splat maps
- **Sky & Atmosphere** — Physically-based atmospheric scattering (Bruneton or Hillaire model), procedural sky with sun/moon, aerial perspective
- **Water / Ocean** — FFT-based ocean simulation (compute), screen-space reflections/refractions, caustics
- **Vegetation** — Instanced foliage with wind animation, impostor LODs, alpha-to-coverage

## 13. Transparency & Order-Independent Transparency

- **Alpha Blending** — Sorted transparent pass (back-to-front)
- **Weighted Blended OIT** — Approximate OIT for particle-like transparency
- **Per-Pixel Linked Lists or Adaptive OIT** — For correct arbitrary transparency (when needed)
- **Stochastic Transparency** — Random threshold alpha testing with temporal accumulation

## 14. Debug & Profiling Tools

- **GPU Timestamp Queries** — Per-pass timing with rolling average display
- **Pipeline Statistics Queries** — Vertex/fragment/compute invocation counts
- **Vulkan Validation Layer Integration** — Debug messenger callbacks, GPU-assisted validation
- **Debug Visualization Modes** — Wireframe overlay, normals, UVs, mip level, overdraw heatmap, light complexity, shadow cascade visualization
- **ImGui Integration** — In-engine debug UI for tweaking parameters, viewing stats
- **RenderDoc / Nsight Integration** — Frame capture support, labeling of objects/passes with `VK_EXT_debug_utils`

## 15. Platform & Window

- **Window Abstraction** — GLFW for windowing (see Section 1)
- **Input System** — Keyboard, mouse, gamepad with action mapping
- **Multi-Monitor / Multi-Window** — Optional support for editor viewports
- **Fullscreen & DPI Scaling** — Proper handling of display modes and Hi-DPI

## 16. Asset Pipeline

- **Model Loading** — glTF 2.0 as primary format (via cgltf or tinygltf), FBX import
- **Texture Compression** — Offline compression to BC7/BC5/ASTC, KTX2 container with Basis Universal
- **Mesh Optimization** — Meshoptimizer for vertex cache, overdraw, and fetch optimization; meshlet generation for mesh shading
- **Asset Cooking / Caching** — Offline asset processing pipeline with content hashing for incremental rebuilds

---

## Suggested Implementation Order

| Phase | Focus | Key Deliverable |
|-------|-------|-----------------|
| **Phase 1** | Core Vulkan infra, swapchain, command buffers, sync, memory | Triangle on screen |
| **Phase 2** | Resource management, descriptor system, shader pipeline | Textured mesh rendering |
| **Phase 3** | Scene management, camera, basic forward PBR, directional light + CSM | Lit PBR scene |
| **Phase 4** | Render graph, deferred or Forward+ pipeline, multiple lights | Many-light scene |
| **Phase 5** | GPU-driven rendering, compute culling, indirect draws | Massive scene perf |
| **Phase 6** | Post-processing stack (TAA, bloom, tone mapping, SSAO) | Polished image |
| **Phase 7** | Global illumination (DDGI or probe-based), RT shadows/reflections | Realistic lighting |
| **Phase 8** | Advanced features (virtual shadow maps, atmosphere, terrain, water) | Full environment |
| **Phase 9** | Debug tools, profiling, editor UI | Developer experience |
