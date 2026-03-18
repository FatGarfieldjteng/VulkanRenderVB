# RT Shadows & RT Reflections — Design and Implementation

## Overall Architecture

The system uses **ray queries** in compute shaders rather than a full ray tracing pipeline (`VK_KHR_ray_tracing_pipeline`). This is a simpler approach: compute shaders fire rays inline using `rayQueryEXT`, which avoids the complexity of shader binding tables, ray generation shaders, and hit/miss shaders.

Ray queries involve two layers:

- **Vulkan API layer**: The `VK_KHR_ray_query` extension (along with `VK_KHR_acceleration_structure` and `VK_KHR_buffer_device_address`) is enabled at device creation time. This makes the ray query functionality available on the GPU.
- **GLSL shader layer**: Compute shaders declare `#extension GL_EXT_ray_query : require` to access the `rayQueryEXT` type and related built-in functions (`rayQueryInitializeEXT`, `rayQueryProceedEXT`, etc.). The GLSL is compiled to SPIR-V using the `SPV_KHR_ray_query` capability.

The `VK_KHR_` (Vulkan API) vs `GL_EXT_` (GLSL) naming difference is a Khronos convention — both refer to the same feature at different levels of the stack.

The rendering flow is a **screen-space post-process** that runs after the forward pass:

```
Forward Pass (rasterize scene → HDR + Depth)
    ↓
RT Shadow Trace  →  RT Shadow Denoise (×3)
RT Reflection Trace  →  RT Reflection Denoise (×3)
    ↓
Composite (blend RT results into HDR image)
```

This is orchestrated by `RayTracingPass`, which is a node in the engine's RenderGraph.

---

## 1. Acceleration Structures (`AccelStructure`)

This is the spatial data structure the GPU uses for fast ray-geometry intersection.

**BLAS (Bottom-Level Acceleration Structure)** — one per unique mesh in `MeshPool`:

- Iterates all `MeshDrawCommand` entries from `MeshPool`
- For each mesh, fills a `VkAccelerationStructureGeometryKHR` with triangle data: vertex buffer address + offset, index buffer address + offset, vertex stride, format (`R32G32B32_SFLOAT`), and `maxVertex`
- Queries build sizes via `vkGetAccelerationStructureBuildSizesKHR`
- Allocates a single scratch buffer (max of all sizes, 128-byte aligned) shared across sequential builds
- Builds all BLAS sequentially in one `ImmediateSubmit` command buffer, with memory barriers between each build, and writes compaction size queries
- After the submit completes, reads compaction query results and runs `CompactBLAS()` — copies each BLAS into a smaller buffer, then destroys the originals

**TLAS (Top-Level Acceleration Structure)** — one instance per renderable entity:

- Iterates the ECS `Registry` for all entities with Transform + Mesh + Material
- For each entity, gets the BLAS device address and creates a `VkAccelerationStructureInstanceKHR` with the entity's world transform (converted from GLM column-major to Vulkan row-major)
- Uploads the instance array to a device-local buffer
- Builds with `PREFER_FAST_TRACE | ALLOW_UPDATE` flags

**UpdateTLAS** provides an incremental update path (`MODE_UPDATE_KHR`) — reusing the same TLAS structure, only updating instance transforms. Falls back to a full rebuild if the TLAS hasn't been built yet.

---

## 2. RT Shadows (`RTShadows` + `rt_shadows.comp`)

**Purpose**: For each screen pixel, cast a shadow ray toward the light. If it hits geometry, the pixel is in shadow.

### GPU Resources

- **Two `R16_SFLOAT` images** (`mShadowImage[0]` and `mShadowImage[1]`): ping-pong pair for denoising. A single-channel half-float storing `0.0` (shadowed) or `1.0` (lit).
- **Descriptor sets**: One for the trace pass (TLAS + output image + depth texture), three for denoise iterations (each ping-ponging input/output + depth).

### Trace Shader (`rt_shadows.comp`)

The shader runs at full resolution, one thread per pixel (8×8 workgroups):

1. **Depth read**: `texelFetch(depthTex, pixel, 0).r`. Sky pixels (depth >= 1.0) are immediately written as `1.0` (fully lit).

2. **World position reconstruction**: Converts pixel coordinate + depth to NDC, then multiplies by `invViewProj`:

```glsl
vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
vec4 world = invViewProj * clip;
return world.xyz / world.w;
```

3. **Soft shadow jitter**: Uses Interleaved Gradient Noise (IGN) to jitter the ray direction around the light direction. Builds a tangent frame from `L`, then offsets by `lightRadius`:

```glsl
vec3 jitteredL = normalize(L + T * cos(angle) * radius + B * sin(angle) * radius);
```

4. **Ray query**: Fires a ray from `worldPos + L * 0.05` (biased along light direction to avoid self-intersection) with `tMin = 0.01`, `tMax = 200.0`, using `TerminateOnFirstHit | Opaque` flags for maximum performance:

```glsl
rayQueryInitializeEXT(rq, tlas,
    gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
    0xFF, worldPos + L * 0.05, 0.01, jitteredL, 200.0);
```

5. **Result**: If the committed intersection is not `None`, shadow = 0.0 (occluded). Otherwise 1.0 (lit). Written to `shadowOutput`.

### C++ Dispatch (`RTShadows::Dispatch`)

- On first call (or after resize), updates all descriptor sets via `mDescriptorsDirty` flag
- Transitions both ping-pong images from `UNDEFINED` → `GENERAL` (discarding previous content each frame)
- Pushes `TracePushConstants` (invViewProj, lightDir with radius in `.w`, cameraPos, resolution)
- Dispatches the compute shader
- Inserts a memory barrier (write → read) for the denoise pass

---

## 3. RT Reflections (`RTReflections` + `rt_reflections.comp`)

**Purpose**: For each screen pixel, cast a reflection ray and store the reflected color + hit confidence.

### GPU Resources

- **Two `R16G16B16A16_SFLOAT` images**: ping-pong for denoising. RGBA where `.rgb` = reflected color, `.a` = hit factor (used as blend weight in compositing).
- Same descriptor layout pattern as shadows.

### Trace Shader (`rt_reflections.comp`)

1. **Depth read + sky skip**: Same as shadows.

2. **Normal reconstruction** (min-abs-delta method): Reads depth at 4 neighboring pixels. For each axis, picks the neighbor whose depth difference from center is smallest, avoiding silhouette edges:

```glsl
vec3 ddx = (abs(dR - d) < abs(d - dL))
    ? ReconstructWorldPos(cR, dR) - P
    : P - ReconstructWorldPos(cL, dL);
vec3 ddy = (abs(dD - d) < abs(d - dU))
    ? ReconstructWorldPos(cD, dD) - P
    : P - ReconstructWorldPos(cU, dU);
return normalize(cross(ddy, ddx));
```

3. **GGX VNDF Importance Sampling** (Heitz 2018): Instead of a perfect mirror reflection, the reflection direction is perturbed based on a `roughness` parameter:
   - Transforms the view vector into a tangent-space frame built from the normal
   - Samples a microfacet half-vector `H` using the GGX visible normal distribution
   - Computes `R = reflect(-V, H)`
   
   This gives rougher surfaces more scattered reflections (blurrier) and smooth surfaces sharper reflections.

4. **Ray query**: Fires from `worldPos + N * 0.05` along `R` with `tMin = 0.01`, `tMax = 200.0`. Uses only `OpaqueEXT` (no early termination — needs the closest hit for color).

5. **Fresnel-weighted hit factor**: On hit, applies Schlick's approximation:

```glsl
float fresnel = 0.04 + 0.96 * pow(1.0 - NdotV, 5.0);
reflHit = clamp(fresnel * 6.0, 0.15, 1.0);
```

   Surfaces viewed head-on get ~15-24% reflectivity; grazing angles get up to 100%. The reflected color is a constant ambient placeholder `vec3(0.4, 0.45, 0.5)` attenuated by distance: `exp(-hitT * 0.02)`. This is **not** actual scene color at the hit point (that would require material evaluation at the hit position, which ray queries don't easily support).

6. **Miss**: `reflColor = vec3(0)`, `reflHit = 0.0` — no reflection contribution.

---

## 4. A-Trous Wavelet Denoiser

Both shadows and reflections use the same denoising approach — an **edge-aware spatial filter** run in 3 iterations with increasing step sizes (1, 2, 4).

### How it works (`rt_shadow_denoise.comp` / `rt_reflect_denoise.comp`)

Each iteration reads from one ping-pong image and writes to the other:

- **5×5 kernel** with weights `[1.0, 2/3, 1/6]` — a separable approximation of a Gaussian
- **Step size** (1 → 2 → 4) makes each iteration cover a wider spatial radius without increasing the kernel size. This is the "à trous" (with holes) technique.
- **Edge-stopping functions** prevent blurring across geometric boundaries:
  - **Depth weight**: `exp(-|centerDepth - sampleDepth| / depthSigma)` — suppresses blending across depth discontinuities
  - **Normal weight**: `pow(max(dot(centerNormal, sampleNormal), 0), normalSigma)` — suppresses blending across different surface orientations (normalSigma = 128 makes this very sharp)
  - **Color weight** (reflections only): `exp(-colorDiff / colorSigma)` — prevents blurring across color discontinuities

The denoise pass reconstructs normals from depth using the same min-abs-delta method.

### Ping-pong pattern

```
Iteration 0: image[0] → image[1]  (stepSize=1)
Iteration 1: image[1] → image[0]  (stepSize=2)
Iteration 2: image[0] → image[1]  (stepSize=4)
```

After 3 iterations (odd count), the final result is always in `image[1]`, hence `mOutputIdx = 1`.

---

## 5. Compositing (`rt_composite.comp`)

A final compute pass that reads the forward pass HDR image (as a storage image for read-modify-write), reads the denoised shadow and reflection textures, and blends them:

**Shadow compositing**:
```glsl
float shadowMul = 0.3 + 0.7 * rtShadow;     // 0.3 ambient floor, never fully black
hdr.rgb *= mix(1.0, shadowMul, shadowStrength);  // shadowStrength controls effect intensity
```
The `0.3` ambient floor ensures shadowed areas retain some visibility instead of going fully black.

**Reflection compositing**:
```glsl
float blendFactor = rtRefl.a * reflectionStrength;
float lum = dot(hdr.rgb, vec3(0.2126, 0.7152, 0.0722));
vec3 modRefl = rtRefl.rgb * max(lum, 0.05);
hdr.rgb = mix(hdr.rgb, modRefl, blendFactor);
```
The reflection color is modulated by the scene's local luminance to prevent bright reflections from overwhelming dark areas.

**Debug mode**: When `debugShadowVis` is enabled, the output shows the raw shadow map as grayscale (white = lit, black = shadowed), or blue if shadows are disabled.

---

## 6. RenderGraph Integration (`RayTracingPass`)

`RayTracingPass` integrates into the engine's render graph:

- **Setup**: Declares a read dependency on the depth resource and a read-write dependency on the HDR color resource, both depending on the forward pass completing first
- **Execute**: Calls `Dispatch` + `Denoise` for shadows (if enabled), then reflections (if enabled), then runs the composite

All per-frame parameters (invViewProj, lightDir, cameraPos, roughness, strength values, enable flags) are passed through `RayTracingPass::Desc`, populated by `Application::DrawFrame`.

---

## 7. Known Limitations

1. **Double shadow stacking**: CSM and RT Shadows operate independently. CSM applies during the forward pass; RT Shadows apply as a post-process multiplier. Both being on produces unnaturally dark shadows.

2. **Reflection color is a placeholder**: The reflection shader doesn't evaluate materials/lighting at the hit point — it uses a constant `vec3(0.4, 0.45, 0.5)` decayed by distance. True reflections would require binding material textures and evaluating PBR at the hit point, or using a ray tracing pipeline with closest-hit shaders.

3. **1 spp with spatial-only denoising**: Each pixel traces exactly one ray per frame. Without temporal accumulation, the denoiser has limited information to work with, so some noise remains — especially for soft shadows with larger `lightRadius`.

4. **Normal reconstruction from depth**: Since the G-Buffer doesn't export normals, they're reconstructed from depth via finite differences. This is approximate and can produce artifacts at silhouette edges, though the min-abs-delta method mitigates this significantly.

---

## 8. Q&A

### Q: Will `GL_EXT_ray_query` become part of core Vulkan?

No. `VK_KHR_ray_query` (the Vulkan API extension behind `GL_EXT_ray_query` in GLSL) is a Khronos-ratified device extension, finalized in November 2020, but it has not been promoted to core Vulkan (1.0–1.4) and there is no announced plan for it.

The main reason is limited hardware coverage — only ~31% on Windows and ~23% on Linux as of 2025. Core features require near-universal support. Ray tracing hardware acceleration is only available on relatively recent GPUs (NVIDIA RTX 20+, AMD RDNA2+, Intel Arc) and is largely absent on mobile. Using `VK_KHR_ray_query` as an optional extension with a runtime fallback (as this engine does via `mDevice.IsRayTracingSupported()`) is the standard practice.

### Q: The document says the system uses `GL_EXT_ray_query`. Why not `VK_KHR_ray_query`?

Both are used — they are the same feature at different layers:

- **`VK_KHR_ray_query`** is the Vulkan API extension, enabled at device creation in C++. This makes the feature available on the GPU.
- **`GL_EXT_ray_query`** is the GLSL shading language extension, declared in shaders with `#extension GL_EXT_ray_query : require`. This exposes the `rayQueryEXT` type and built-in functions.
- **`SPV_KHR_ray_query`** is the SPIR-V capability that the GLSL compiler (`glslangValidator`) emits when compiling shaders that use ray queries.

The `VK_KHR_` vs `GL_EXT_` naming difference is a Khronos convention — Vulkan API extensions use the `VK_` prefix, GLSL extensions use the `GL_` prefix. You need both: the API extension to enable the feature, and the GLSL extension to use it in shaders.

### Q: What does "fire rays inline" mean?

"Inline" means the ray tracing happens within the same shader invocation that requested it, rather than launching separate shaders.

With a **ray tracing pipeline** (`VK_KHR_ray_tracing_pipeline`), you dispatch a dedicated pipeline with multiple shader stages (ray generation, closest-hit, miss, any-hit). Each ray potentially launches a new shader invocation via a Shader Binding Table. The GPU schedules these dynamically.

With **ray queries** (`VK_KHR_ray_query`), you write a regular compute or fragment shader and trace rays directly inside it:

```glsl
rayQueryEXT rq;
rayQueryInitializeEXT(rq, tlas, flags, 0xFF, origin, tMin, direction, tMax);
while (rayQueryProceedEXT(rq)) { }
if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
    shadow = 0.0;
}
```

The ray traversal is a blocking call within the same thread — no new shaders are launched. The shader issues the ray, waits for the result, and continues. This is simpler (one shader file, no SBT management), works in any shader stage, and has less overhead for simple 1-ray-per-pixel effects. The tradeoff is less flexibility for recursive ray tracing or complex per-hit shading.

### Q: BLAS vs. TLAS

The acceleration structure is a two-level hierarchy:

**BLAS (Bottom-Level)** — one per unique mesh. Each BLAS is a BVH (Bounding Volume Hierarchy) built from the triangles of a single mesh in its local space. In this engine, `BuildBLAS` creates one BLAS per `MeshDrawCommand` in the `MeshPool`.

**TLAS (Top-Level)** — one for the whole scene. Instead of containing triangles, the TLAS is a BVH over **instances** — references to BLAS entries paired with world transform matrices. Each renderable entity in the ECS becomes one instance in the TLAS.

For example, 3 cubes at different positions produce 1 BLAS (shared mesh) and 3 TLAS instances:

```
TLAS (scene-level BVH)
 ├── Instance 0: cube BLAS @ position (0, 0, 0)
 ├── Instance 1: cube BLAS @ position (5, 0, 0)
 ├── Instance 2: cube BLAS @ position (-3, 2, 0)
 ├── Instance 3: sphere BLAS @ position (0, 5, 0)
 └── Instance 4: floor BLAS @ position (0, 0, 0)
```

Why two levels:

- **Memory efficiency**: Identical meshes share one BLAS. 100 instances of the same mesh = 1 BLAS + 100 lightweight instance entries.
- **Update efficiency**: When an object moves, only the TLAS needs updating (just transforms). The BLAS stays the same because the mesh geometry hasn't changed. This is what `UpdateTLAS` does with `MODE_UPDATE_KHR`.
- **Build performance**: BLAS builds are expensive (processing all triangles) and done once at load time. TLAS builds are cheap (processing only instance bounding boxes) and can be done per frame.

When a ray is traced, the GPU first traverses the TLAS to find which instances the ray might hit, then for each candidate, transforms the ray into that BLAS's local space and traverses its triangle BVH. This all happens transparently inside `rayQueryProceedEXT`.

### Q: How does the denoising pipeline work?

There are 3 denoising passes (compute dispatches), each applying the same edge-aware 5×5 blur filter but with increasing step size:

| Pass | Step Size | Effective Radius | Read From | Write To |
|------|-----------|-----------------|-----------|----------|
| 0    | 1         | ~2 pixels       | image[0]  | image[1] |
| 1    | 2         | ~4 pixels       | image[1]  | image[0] |
| 2    | 4         | ~8 pixels       | image[0]  | image[1] |

Each pass samples pixels spaced `stepSize` apart, so the 5×5 kernel covers a progressively larger area. This is the "à trous" (French for "with holes") technique — by doubling the step size each pass, the combination of all three is equivalent to a ~16px radius Gaussian blur, but computed with only 3 × 25 = 75 texture reads per pixel instead of hundreds.

The two images ("ping-pong pair") are a memory management pattern: you can't read and write the same image simultaneously in a compute shader, so each pass reads from one image and writes to the other, with a memory barrier in between.

The full pipeline for each effect (RT Shadows or RT Reflections) is:

1. **Trace pass** (1 dispatch) — fires rays, writes noisy 1-spp result to `image[0]`
2. **Denoise pass 0** (1 dispatch) — fine-scale smoothing, `image[0]` → `image[1]`
3. **Denoise pass 1** (1 dispatch) — medium-scale smoothing, `image[1]` → `image[0]`
4. **Denoise pass 2** (1 dispatch) — coarse-scale smoothing, `image[0]` → `image[1]`

That's 4 compute dispatches total per effect. The final denoised result sits in `image[1]`, which is why `mOutputIdx = 1`.

### Q: What is the purpose of `jitteredL` in the shadow shader?

The jitter creates **soft shadows** from an area light, rather than hard binary shadows from a point light.

Without jitter, every pixel fires its shadow ray in the exact same direction — straight toward the light (`L`). The result is either 0 (blocked) or 1 (lit), producing perfectly sharp shadow edges that look harsh and unrealistic.

With jitter, the light is treated as having a radius (`lightRadius`). Each pixel perturbs the ray direction slightly within a cone around `L`:

```glsl
vec3 T = normalize(cross(L, ...));
vec3 B = cross(L, T);
float angle = noise * 6.2831853;
float radius = sqrt(noise) * lightRadius;
vec3 jitteredL = normalize(L + T * cos(angle) * radius + B * sin(angle) * radius);
```

This constructs a random point on a disc perpendicular to the light direction, then shoots the ray toward that offset instead of the exact light center. Across neighboring pixels, each gets a different random offset (from Interleaved Gradient Noise), so some hit the occluder and some don't. The denoiser then averages these noisy 0/1 samples, producing a smooth gradient at shadow edges — approximating the penumbra of an area light.

If `lightRadius = 0`, the jitter does nothing and you get hard shadows. Larger values produce progressively softer edges. This is why the denoiser is essential — with only 1 sample per pixel, the raw output is very noisy, and the A-Trous filter smooths it into a clean penumbra.

### Q: How does the shadow ray query work?

The core shadow ray tracing is:

```glsl
rayQueryEXT rq;
rayQueryInitializeEXT(rq, tlas,
    gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
    0xFF,
    worldPos + L * 0.05,
    0.01,
    jitteredL,
    200.0);

while (rayQueryProceedEXT(rq)) {}

if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
    shadow = 0.0;
}
```

**`rayQueryInitializeEXT`** parameters:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `tlas` | acceleration structure | The scene geometry to trace against |
| flags | `TerminateOnFirstHit \| Opaque` | Stop as soon as ANY intersection is found (we only care if something blocks the light, not which object or how far). Treat all geometry as opaque, skipping any-hit processing. Both are performance optimizations. |
| `0xFF` | cull mask | Test against all instance mask bits — don't exclude any geometry |
| `worldPos + L * 0.05` | ray origin | Surface point, offset 0.05 units along the light direction to prevent self-intersection (the ray hitting the surface it starts from due to floating-point imprecision) |
| `0.01` | tMin | Ignore intersections closer than 0.01 units — another layer of self-intersection prevention |
| `jitteredL` | ray direction | Direction toward the light, randomly perturbed for soft shadows |
| `200.0` | tMax | Maximum ray distance — shadow casters farther than this are ignored |

**`while (rayQueryProceedEXT(rq)) {}`** drives the BVH traversal. Each call advances one step through the acceleration structure. The loop body is empty because `TerminateOnFirstHit` and `Opaque` flags mean there are no per-intersection decisions to make — the GPU auto-commits the first hit and stops.

**`rayQueryGetIntersectionTypeEXT(rq, true)`** checks the committed (final) result. If it's not `None`, the ray hit something — geometry blocks the light, so `shadow = 0.0` (fully occluded). If no intersection was found, `shadow` stays at `1.0` (fully lit).

### Q: The shader only references TLAS — is BLAS not used?

BLAS is used, but indirectly. The shader only binds the TLAS, and the GPU automatically traverses into BLAS structures during ray query execution.

When the TLAS is built on the CPU, each `VkAccelerationStructureInstanceKHR` stores a BLAS device address:

```cpp
VkDeviceAddress blasAddr = vkGetAccelerationStructureDeviceAddressKHR(mDevice, &addrInfo);
inst.accelerationStructureReference = blasAddr;
```

Each TLAS instance points to a BLAS via its GPU device address. When the shader calls `rayQueryProceedEXT(rq)`, the GPU hardware does a two-level traversal internally:

1. **Traverse the TLAS BVH** — test the ray against instance bounding boxes
2. For each instance the ray might hit, **transform the ray** into that instance's local space
3. **Traverse that instance's BLAS BVH** — test against actual triangles
4. Return intersection results

This two-level indirection is completely transparent to the shader. From the shader's perspective, you bind the TLAS and get back triangle intersection results — the BLAS traversal is handled automatically by the ray query implementation. BLAS is essential (it contains the actual geometry), but the shader never references it explicitly.

### Q: Why does the scratch buffer need manual alignment but the instance buffer doesn't?

The scratch buffer address must be aligned to `minAccelerationStructureScratchOffsetAlignment` (typically 128 bytes), a device-specific property reported in `VkPhysicalDeviceAccelerationStructurePropertiesKHR`. This is a requirement of the acceleration structure **build operation**, not the buffer itself. The driver/VMA guarantees the buffer's own `VkMemoryRequirements::alignment` is met, but that may be only 16 or 64 bytes — not enough for the 128-byte scratch requirement. So we manually over-allocate and align:

```cpp
scratchBuffer.CreateDeviceLocalEmpty(allocator, usage, size + kScratchAlign);
VkDeviceAddress scratchAddr = AlignUp(scratchBuffer.GetDeviceAddress(mDevice), kScratchAlign);
```

The instance buffer, on the other hand, requires 16-byte alignment for its device address. Buffers with `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` always get at least 16-byte aligned addresses from the driver, so the instance buffer is aligned automatically.

| Buffer | Required alignment | Who guarantees it |
|--------|-------------------|-------------------|
| Instance buffer | 16 bytes | Driver/VMA — buffer memory alignment is always >= 16 for device-address buffers |
| Scratch buffer | `minAccelerationStructureScratchOffsetAlignment` (128) | **Application** — this is an AS build requirement, not a buffer memory requirement |
| Vertex/index buffers in BLAS | Per-format alignment (e.g., 4 for R32) | Driver/VMA — standard buffer alignment covers this |

The key distinction: buffer memory alignment is handled automatically by the allocator, but acceleration structure build requirements (like scratch alignment) are additional constraints the application must enforce explicitly.

### Q: Why implement RT Shadows in a compute shader rather than a fragment shader?

RTShadows uses `GL_EXT_ray_query` in a **compute shader**, not in a fragment shader. You *could* use `GL_EXT_ray_query` inside a fragment shader (it's supported there too), but compute is preferred here for several reasons:

**1. No geometry or render pass needed.** A fragment shader requires a rasterization pipeline — you'd have to draw a fullscreen triangle/quad, set up a render pass (or dynamic rendering), create a graphics pipeline with vertex/fragment stages, configure a framebuffer, etc. The RT Shadows pass doesn't rasterize anything. It reads depth, reconstructs world positions, and traces shadow rays. A compute dispatch is simpler: just `vkCmdDispatch((width+7)/8, (height+7)/8, 1)`.

**2. Output to storage images.** The shadow output is `image2D shadowOutput` written via `imageStore`. Fragment shaders write to color attachments, not storage images. They *can* write to storage images, but then you lose the advantages of fragment shaders (rasterization, blending, depth testing) and gain nothing over compute.

**3. Better work distribution for ray tracing.** Fragment shaders are scheduled per-pixel by the rasterizer's fixed-function hardware. Compute shaders have explicit workgroup sizing (`local_size_x = 8, local_size_y = 8`). For ray tracing workloads, where nearby pixels often trace rays in similar directions (spatial coherence in the TLAS), the 8×8 tile grouping maps well to the GPU's warp/wavefront organization. The driver has more freedom to schedule work efficiently without the rasterizer's constraints.

**4. No dependency on the graphics pipeline state.** The shadow trace runs as a standalone pass between the forward pass and the final composite. In compute, it only needs the TLAS, depth texture, and push constants. In a fragment shader, you'd need to set up viewport, scissor, a dummy vertex shader, disable depth testing, configure color writes — boilerplate that adds nothing.

**5. The denoise pass is also compute.** After the trace, `RTShadows::Denoise()` runs 3 iterations of an A-Trous bilateral filter, also as compute dispatches. Keeping both trace and denoise as compute passes means uniform barrier handling (compute → compute) and no render pass transitions in between. If the trace were a fragment shader, you'd need a render pass begin/end, then transition back to compute for the denoise — extra complexity for no benefit.

**6. Decoupled from screen resolution and triangle count.** A compute dispatch doesn't need a framebuffer or know about the rasterization viewport. The resolution comes in via push constants. This makes the class trivially resizable (`Resize()` just recreates the images) without touching any pipeline state.

A full ray tracing pipeline (`VK_KHR_ray_tracing_pipeline` with `vkCmdTraceRaysKHR`) would also be overkill for shadows: shadow rays only need a binary hit/miss result with no material evaluation, no closest-hit processing, and no BSDF sampling. `GL_EXT_ray_query` gives exactly that — a simple inline ray-scene intersection test — without requiring an SBT or shader groups.
