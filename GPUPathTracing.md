# GPU Path Tracing Pipeline — Design & Implementation

This document describes the design and implementation of the GPU path tracing pipeline in VulkanRenderVB, including the ray tracing pipeline, path tracer, NRD denoiser, and composite passes.

---

## 1. Architecture Overview

```
                    AccelStructure (BLAS/TLAS)
                              │
                              ▼
    PathTracer  ◄──  RTPipeline + ShaderBindingTable
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
  PathTracingPass      HybridRTPass         NRDDenoiser
  (Full PT mode)       (Hybrid mode)       (REBLUR_DIFFUSE)
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
              Composite (pt_composite / pt_composite_denoise / pt_composite_compare)
                              │
                              ▼
                         HDR Output
```

**Data flow:**
- **AccelStructure** builds and manages the bottom-level (BLAS) and top-level (TLAS) acceleration structures.
- **PathTracer** launches the ray tracing pipeline, producing per-pixel color, albedo, normal, depth, and motion outputs.
- **NRDDenoiser** (optional) runs NVIDIA REBLUR on the path tracer's 1-SPP output for real-time denoising.
- **Composite** writes the final result (denoised or raw) into the HDR image for post-processing.

---

## 2. Acceleration Structure

### 2.1 BLAS (Bottom-Level Acceleration Structure)

- One BLAS per mesh in the scene.
- Built from vertex and index buffers via `VkAccelerationStructureGeometryKHR` with `VK_GEOMETRY_TYPE_TRIANGLES_KHR`.
- Compaction is applied after build to reduce memory usage.
- BLAS is built once during initialization and reused across frames.

### 2.2 TLAS (Top-Level Acceleration Structure)

- Contains instances referencing BLAS entries with world-space transforms.
- Built from the scene registry: each renderable entity becomes a `VkAccelerationStructureInstanceKHR`.
- `instanceShaderBindingTableRecordOffset` is set per instance for SBT indexing (currently uses a simplified layout with offset 0 for all instances).
- TLAS is rebuilt when the scene changes; for path tracing, `numRayTypes = 2` (primary + shadow).

### 2.3 Instance Info

```cpp
struct RTInstanceInfo {
    int32_t  vertexOffset;
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t materialIndex;
};
```

Each TLAS instance maps to mesh geometry and a material index. The closest-hit shader uses `gl_InstanceCustomIndexEXT` to look up `RTInstanceInfo` and fetch vertex data.

---

## 3. RTPipeline — Ray Tracing Pipeline Builder

`RTPipeline` uses a builder pattern to assemble the Vulkan ray tracing pipeline.

### 3.1 API

- **AddStage(stage, module)** — Registers a shader module (raygen, miss, closest-hit, any-hit) and returns its index.
- **AddRayGenGroup(idx)** — Defines the raygen shader group.
- **AddMissGroup(idx)** — Defines a miss shader group (one per ray type).
- **AddHitGroup(closestHitIdx, anyHitIdx, intersectionIdx)** — Defines a hit group. Unused slots use `VK_SHADER_UNUSED_KHR`.
- **Build(device, layout, maxRecursionDepth)** — Creates `VkRayTracingPipelineKHR`.

### 3.2 Recursion Depth

`maxRecursionDepth = 1` — The path tracer does **not** use recursive `traceRayEXT()` from hit/miss shaders. All bounces are handled by an **iterative loop in the raygen shader**. This:

- Avoids stack overflow and performance cliffs.
- Works on AMD RDNA2 (which has `maxRayRecursionDepth = 1`).
- Keeps ray generation and shading logic in one place.

---

## 4. ShaderBindingTable (SBT)

The SBT is laid out in four contiguous regions:

```
[ Raygen | Miss₀ Miss₁ | Hit₀ Hit₁ | Callable ]
```

### 4.1 Layout

- **Raygen:** 1 group.
- **Miss:** 2 groups — primary miss (environment sampling) and shadow miss (returns "unshadowed").
- **Hit:** 2 groups — primary (closest-hit + any-hit) and shadow (any-hit only, `SkipClosestHitShader`).
- **Callable:** 0 groups.

### 4.2 SBT Indexing

- `sbtRecordOffset` and `sbtRecordStride` in `traceRayEXT()` select the group:
  - Primary ray: offset 0, stride 2.
  - Shadow ray: offset 1, stride 2.
- `missIndex` selects which miss shader runs on a miss (0 = primary, 1 = shadow).

### 4.3 Simplified SBT

The current implementation uses a **single hit group pair** for all materials. Material differentiation happens inside the shader via `instanceInfos[gl_InstanceCustomIndexEXT].materialIndex`, not via per-material SBT entries. This simplifies the SBT and avoids per-material pipeline rebuilds.

---

## 5. PathTracer — CPU-Side

### 5.1 Output Images

| Image         | Format              | Purpose                                      |
|---------------|---------------------|----------------------------------------------|
| colorOutput   | RGBA32F             | Raw 1-SPP path-traced color (per frame)      |
| accumBuffer   | RGBA32F             | Progressive accumulation (when denoiser off) |
| albedoOutput  | RGBA8               | Base color for denoiser remodulation         |
| normalOutput  | RGBA16F             | World-space normal + roughness               |
| depthOutput   | R32F                | Linear view-space depth (hit distance)      |
| motionOutput  | RG32F               | Screen-space motion vectors                  |

### 5.2 Descriptor Bindings (Set 0)

| Binding | Type                    | Content                          |
|---------|-------------------------|----------------------------------|
| 0       | Acceleration structure  | TLAS                             |
| 1–5     | Storage image           | color, accum, normal, albedo, depth |
| 6–9     | Storage buffer          | vertex, index, material, instance |
| 10      | Combined image sampler  | Environment cube map             |
| 11      | Combined image sampler  | BRDF LUT                         |
| 12      | Combined image sampler  | Irradiance map                   |
| 13      | Storage image           | Motion output                    |
| 14      | Uniform buffer          | viewProj, prevViewProj           |

Set 1: Bindless texture array for material textures.

### 5.3 Push Constants (128 bytes)

```cpp
struct PushConstants {
    glm::mat4  invViewProj;       // Clip to world
    glm::vec4  cameraPosAndFrame; // .xyz = camera pos, .w = sample offset
    glm::vec4  sunDirAndRadius;   // .xyz = sun direction, .w = light radius
    glm::vec4  sunColorIntensity; // .rgb = color, .w = intensity
    glm::uvec4 params;             // x=maxBounces, y=sampleOff, z=enableMIS, w=accumFrames
};
```

### 5.4 Progressive Accumulation

- `mAccumFrames` tracks accumulated samples.
- When the camera moves (viewProj change > 1e-4), accumulation resets.
- `WasAccumulationReset()` signals the NRD denoiser to invalidate its history.
- Accumulation formula: `newAccum = mix(prevAccum, color, 1/(accumFrames+1))`.

### 5.5 Trace() Flow

1. Detect camera movement; reset accumulation if needed.
2. Update frame UBO (viewProj, prevViewProj) for motion vectors.
3. Transition output images to GENERAL.
4. Clear accum buffer on reset.
5. Push constants, bind descriptors, dispatch `vkCmdTraceRaysKHR(width, height, 1)`.

---

## 6. Ray Generation Shader (`pt_raygen.rgen`)

The raygen shader implements the full path tracing loop. One invocation per pixel.

### 6.1 Primary Ray Setup

```glsl
vec2 jitter = rand2() - 0.5;  // Sub-pixel jitter for AA
vec2 uv = (vec2(pixel) + 0.5 + jitter) / vec2(size);
vec4 clip = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
vec4 worldTarget = invViewProj * clip;
vec3 origin = cameraPos;
vec3 direction = normalize(worldTarget.xyz / worldTarget.w - origin);
```

### 6.2 Bounce Loop (Iterative)

For each bounce (up to `maxBounces`):

1. **Trace primary/bounce ray** — `traceRayEXT(..., 0, 2, 0, ...)` with payload 0.
2. **Handle miss** — If `payload.hitT < 0`, add environment color and exit.
3. **Handle hit:**
   - Add emissive contribution.
   - On first hit: write G-buffer (normal, albedo, depth, motion) for the denoiser.
   - **Direct lighting:** Sample sun with area light jitter; trace shadow ray; if unshadowed, add Cook-Torrance BRDF contribution.
   - **BSDF importance sampling:** Choose specular or diffuse by Fresnel weight; sample direction; compute throughput.
   - **Russian roulette:** After bounce 2, probabilistically terminate based on throughput.
4. **Next bounce:** Set `origin = hitPos + N*EPSILON`, `direction = newDir`.

### 6.3 Direct Lighting

- Sun modeled as disk light with configurable radius.
- Jittered light direction: `L + T*cos(angle)*r + B*sin(angle)*r` for soft shadows.
- Shadow ray uses `gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT`.
- BRDF: Cook-Torrance (GGX D, Smith G, Schlick F) for specular; Lambertian for diffuse.

### 6.4 BSDF Sampling

- **Specular:** GGX VNDF sampling in tangent space; `reflect(-V, H)` for outgoing direction.
- **Diffuse:** Cosine-weighted hemisphere sampling.
- Throughput updated with MIS weight: `bsdfWeight / pdf`.

### 6.5 Firefly Suppression

```glsl
if (cMax > 50.0) color *= 50.0 / cMax;
```

Clamps extreme samples to avoid corrupting accumulation and denoiser.

### 6.6 Outputs

- **colorOutput:** Raw 1-SPP color (for NRD when denoiser on).
- **accumBuffer:** Progressive blend when denoiser off.
- **G-buffer (first hit only):** normal, albedo, depth, motion. Sky pixels use `viewZ = 600000` so NRD skips denoising.

---

## 7. Closest-Hit Shader (`pt_closesthit.rchit`)

Runs when a ray hits geometry. Writes surface data to the ray payload.

### 7.1 Vertex Fetch

```glsl
InstanceInfo info = instanceInfos[gl_InstanceCustomIndexEXT];
uint i0 = indices[info.firstIndex + gl_PrimitiveID * 3 + 0];
// ... fetch v0, v1, v2
vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
vec3 localPos = v0.position * bary.x + ...;
vec3 localNormal = ...;
vec2 texCoord = ...;
```

### 7.2 Normal Mapping

- Build TBN from interpolated tangent; sample normal map; transform to world space.
- Fallback to geometric normal if tangent length is negligible.

### 7.3 Material Lookup

- `MaterialParams mat = materials[info.materialIndex]`
- Base color, metallic, roughness from textures and factors.
- Emissive from emissive texture × baseColorFactor.

### 7.4 Payload Output

```glsl
payload.hitPos   = worldPos;
payload.hitT     = gl_HitTEXT;
payload.normal   = N;
payload.metallic = metallic;
payload.albedo   = baseColor.rgb;
payload.roughness = roughness;
payload.emissive = emissive;
```

The raygen reads this payload after `traceRayEXT` returns. No recursive rays are traced from the closest-hit shader.

---

## 8. Any-Hit Shader (`pt_anyhit.rahit`)

Handles alpha-tested geometry (foliage, fences, etc.).

### 8.1 Stochastic Alpha Test

```glsl
float alpha = mat.baseColorFactor.a * texture(...).a;
uint seed = gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x;
float threshold = float(pcgHash(seed)) / 4294967296.0;
if (alpha < threshold) ignoreIntersectionEXT;
```

- Turns opacity into a probability: transparent pixels are skipped with probability `1 - alpha`.
- Produces correct semi-transparency on average over many samples.

### 8.2 Usage

- Primary hit group: skip transparent texels so rays pass through.
- Shadow hit group: skip transparent texels so shadows show leaf-shaped holes.

---

## 9. Miss Shaders

### 9.1 Primary Miss (`pt_miss.rmiss`)

- Samples environment cube map with `gl_WorldRayDirectionEXT`.
- Sets `payload.hitT = -1` (signals miss to raygen).
- Sets `payload.albedo = envColor` (environment contribution).

### 9.2 Shadow Miss (`pt_shadow_miss.rmiss`)

- Sets `shadowPayload = 1.0` (light not occluded).
- Shadow ray uses payload location 1; primary uses location 0.

---

## 10. NRD Denoiser

### 10.1 Pipeline

1. **Prepack** (`nrd_prepack.comp`): Converts path tracer G-buffer to NRD format.
   - **Demodulation:** `irradiance = radiance / max(albedo, 0.12)` to avoid 50× blow-up for dark pixels while limiting dark halos.
   - **Hit distance:** Normalized for REBLUR.
   - **Motion:** Zeroed (NRD uses internal motion from view matrices).
2. **NRD REBLUR_DIFFUSE:** Temporal + spatial denoising.
3. **Composite** (`pt_composite_denoise.comp`): Remodulates denoised irradiance with albedo; writes to HDR.

### 10.2 REBLUR Settings

- `minBlurRadius = 1`, `maxBlurRadius = 22`
- `disocclusionThreshold = 0.01`
- `accumulationMode = CLEAR_AND_RESTART` when history invalid
- `isMotionVectorInWorldSpace = true`, `motionVectorScale = 0`

### 10.3 Demodulation / Remodulation

- Prepack: `irradiance = radiance / max(albedo, 0.12)`
- Composite: `ptColor = irradiance * max(albedo, 0.12)`

Effective albedo is clamped to 0.12 to avoid extreme values at sharp albedo edges (e.g. checkerboard).

---

## 11. Composite Passes

### 11.1 pt_composite.comp (No Denoiser)

- Samples accum buffer (progressive result) with a sampler.
- Writes to HDR within `[splitLeft, splitRight)` for split-screen.

### 11.2 pt_composite_denoise.comp (With Denoiser)

- Reads NRD output (YCoCg) and albedo.
- Converts YCoCg to linear; remodulates with effective albedo.
- Clamps to `[0, 100]`.

### 11.3 pt_composite_compare.comp (Debug)

- Split-screen: left = denoised, right = raw accum.
- Enabled via "Denoiser comparison" checkbox when NRD is on.

---

## 12. Render Modes

### 12.1 Full Path Tracing

- `PathTracingPass` runs the path tracer and optional NRD denoiser.
- Composite writes PT output to HDR (full screen or split region).
- No rasterization; all visibility and lighting via rays.

### 12.2 Hybrid

- Forward pass rasterizes G-buffer (depth, HDR).
- `HybridRTPass` runs the same path tracer for multi-bounce GI.
- Composite adds PT result to the rasterized image.
- Uses same NRD and composite pipelines as full PT.

### 12.3 Rasterization Only

- No path tracer; forward + post-process only.
- Phase 9 RT shadows/reflections can run separately.

---

## 13. Key Design Decisions

### 13.1 Iterative Bounces in Raygen

All bounces are handled in a loop in the raygen shader. No recursive `traceRayEXT` from hit/miss. This keeps `maxRecursionDepth = 1` and works on all hardware.

### 13.2 Payload-Based G-Buffer

The closest-hit shader writes surface data to the ray payload. The raygen reads it after each `traceRayEXT`. No separate G-buffer images for bounce data; only first-hit G-buffer is written for the denoiser.

### 13.3 Single Hit Group Pair

All materials share the same closest-hit and any-hit shaders. Material data comes from `instanceInfos[gl_InstanceCustomIndexEXT].materialIndex` and the material buffer. This simplifies the SBT and avoids per-material pipeline variants.

### 13.4 Bindless Textures

Material textures are in a bindless array. The closest-hit shader uses `textures[nonuniformEXT(mat.baseColorTexIdx)]` etc. One descriptor set covers all materials.

### 13.5 1-SPP + Denoiser

Path tracer outputs 1 sample per pixel per frame. With NRD, the result is denoised in real time. Without NRD, progressive accumulation over many frames reduces noise (camera must be still).

### 13.6 Sky ViewZ for NRD

Sky pixels use `viewZ = 600000` (> `denoisingRange` 500000) so NRD does not denoise the sky. Using 0 would incorrectly treat sky as very close geometry.

---

## 14. File Reference

| Component        | Files                                                                 |
|------------------|-----------------------------------------------------------------------|
| Path tracer      | `PathTracer.cpp/h`, `pt_raygen.rgen`, `pt_closesthit.rchit`, `pt_anyhit.rahit`, `pt_miss.rmiss`, `pt_shadow_miss.rmiss`, `pt_common.glsl` |
| NRD denoiser     | `NRDDenoiser.cpp/h`, `nrd_prepack.comp`                               |
| Composite        | `pt_composite.comp`, `pt_composite_denoise.comp`, `pt_composite_compare.comp` |
| Pipeline         | `RTPipeline.cpp/h`, `ShaderBindingTable.cpp/h`                        |
| Acceleration     | `AccelStructure.cpp/h`                                                |
| Passes           | `PathTracingPass.cpp/h`, `HybridRTPass.cpp/h`                         |

---

## 15. Q&A

### Q: Why two miss shaders?

**A:** Primary rays and shadow rays need different behavior on miss. Primary miss returns environment color; shadow miss returns "unshadowed" (1.0). The `missIndex` in `traceRayEXT` selects which runs.

### Q: Why does the primary hit group need an any-hit shader?

**A:** For alpha-tested geometry. Without it, rays would hit transparent texels as solid. The any-hit shader calls `ignoreIntersectionEXT` for transparent pixels so the ray continues.

### Q: How does the raygen get hit data from the closest-hit?

**A:** Via the ray payload. `traceRayEXT` is synchronous: when it returns, the payload has been written by the closest-hit (or miss). The raygen reads `payload.hitPos`, `payload.normal`, etc., and uses them for shading and the next bounce.

### Q: What is the Builder pattern in RTPipeline?

**A:** A creational pattern that configures a complex object step by step, then builds it. `RTPipeline` uses `AddStage`, `AddRayGenGroup`, etc., to assemble data, then `Build()` creates the Vulkan pipeline in one call.

### Q: Can the path tracer reuse the Phase 9 BLAS/TLAS?

**A:** Yes. BLAS is identical. TLAS is rebuilt with `numRayTypes = 2` so SBT offsets are correct. The spatial structure is the same; only instance metadata changes.

### Q: Is mRTShadows used in full path tracing?

**A:** No. In full PT mode, shadows are traced by the path tracer's own shadow rays. `mRTShadows` is used only in rasterization and hybrid modes (Phase 9 ray-query passes).

### Q: Only the raygen shader outputs buffers for the NRD denoiser, right?

**A:** Yes. All buffers consumed by the NRD denoiser are written only by the raygen shader (`pt_raygen.rgen`). The closest-hit shader writes only to the ray payload; the raygen reads that payload and then writes to the output images. The NRD inputs (color, normal, albedo, depth, motion) are all populated by the raygen: on first hit it writes the G-buffer from payload data; on miss it writes sky G-buffer values; at the end it writes the color output.

### Q: Why are only some buffers written when bounce == 0?

**A:** The G-buffer (normal, albedo, depth, motion) is written only on the **primary hit** (first surface the camera ray hits) because NRD is a screen-space denoiser. It denoises the radiance at the primary visible surface for each pixel. The G-buffer describes that surface: normal for edge-aware filtering, depth for depth-based filtering and temporal reprojection, motion for temporal reprojection, albedo for demodulation/remodulation. Secondary bounces (bounce 1, 2, 3...) hit other surfaces that have no screen-space position—they only contribute to the primary pixel's color. NRD operates per pixel in screen space, so it only needs the primary surface's geometry and material. The color output, by contrast, is the full path contribution (direct + all bounces) for that pixel, so it is written once at the end of the path.

### Q: A new ray is created at the end of the bounce loop, right?

**A:** Yes. At the end of each iteration, the next ray's origin and direction are computed (`origin = hitPos + N * EPSILON; direction = newDir`). The ray is traced at the start of the next iteration when `traceRayEXT` is called with those updated values.

### Q: One gl_LaunchID execution is the lifetime of one ray path, right?

**A:** Yes. Each `gl_LaunchIDEXT.xy` corresponds to one pixel and one raygen invocation. That invocation handles the full light path for that pixel: it starts with a camera ray, traces bounces (reflections/refractions via BSDF sampling), and terminates on miss, Russian roulette, or max bounces. Each `traceRayEXT` call traces one ray segment; together they form one path from the camera through multiple bounces until termination.

### Q: Thus REBLUR tends to create blurred result, right?

**A:** Yes. REBLUR is a blur-based denoiser: it uses multiple blur passes to average out noise. It uses normals, depth, and motion for edge-aware filtering to reduce blur across edges, but the result is still softer than raw path-traced output. The trade-off is more blur → less noise but softer image; less blur → sharper but noisier. NRD exposes parameters like `minBlurRadius` and `maxBlurRadius` to tune this balance.
