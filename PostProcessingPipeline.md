# Post-Processing Pipeline

## Overview

The post-processing pipeline transforms the raw HDR output of the forward pass into a polished final image on the swapchain. It's orchestrated by `PostProcessStack` and executed as a single render graph pass (`PostProcessPass`). The pipeline runs **5 effects in a fixed order**, each individually togglable from the UI.

## Data Flow Summary

```
ForwardPass -> [HDR Image (R16G16B16A16_SFLOAT)]
                   |
                   +-- 1. AutoExposure (compute) -> exposure SSBO
                   |
                   +-- 2. SSAO (compute)          -> AO texture (R8)
                   |
                   +-- 3. Bloom (compute)          -> bloom mip chain (R16G16B16A16)
                   |
                   +-- 4. ToneMapping (graphics)   -> LDR intermediate or swapchain
                   |         reads: HDR + bloom + AO + exposure
                   |
                   +-- 5. ColorGrading (graphics)  -> swapchain
                             reads: LDR intermediate
```

## Intermediate Images

`PostProcessStack` owns three key images:

- **mHDRImage** (`R16G16B16A16_SFLOAT`):
  The forward pass renders here. All compute effects read from it.

- **mLDRImage** (swapchain format, e.g. `B8G8R8A8_SRGB`):
  Intermediate between tone mapping and color grading.
  Only used when both are enabled.

- **Placeholders** (1x1 white `R8_UNORM`=1.0 and 1x1 black `R16G16B16A16_SFLOAT`=0.0):
  Fallbacks when effects are disabled.
  White AO = identity for `hdr *= ao`.
  Black bloom = identity for `hdr += bloom`.

The placeholders are lazily initialized on first use via `TransitionPlaceholders()`, which clears them to their target values with `vkCmdClearColorImage` and transitions to `SHADER_READ_ONLY_OPTIMAL`.

---

## Effect 1: Auto Exposure

**Files**: `AutoExposure.h/cpp`, `exposure_histogram.comp`, `exposure_average.comp`

**Purpose**: Adapts scene brightness over time, simulating the human eye's adjustment to bright/dark scenes.

**Algorithm** (two-pass compute):

### Pass A - Histogram (`exposure_histogram.comp`, workgroup 16x16)

- Reads the HDR image via `sampler2D`
- Computes per-pixel luminance:
  `lum = dot(color, vec3(0.2126, 0.7152, 0.0722))`
- Maps luminance to a log-space bin:
  `logLum = (log2(lum) - minLogLum) * invLogLumRange`
  then quantizes to 256 bins (bin 0 = near-black pixels)
- Uses `shared uint sharedBins[256]` for per-workgroup reduction,
  then `atomicAdd` to a global `HistogramSSBO`

### Pass B - Average (`exposure_average.comp`, single workgroup of 256 threads)

- Reads the global histogram into shared memory and resets it to zero for the next frame
- Thread 0 computes the weighted average log-luminance by iterating bins 1-255
  (skipping bin 0 = near-black), computing `weightedSum / totalWeight`
- Converts to exposure:
  `targetExposure = 0.18 / max(avgLum, 1e-5)` (standard 18% middle-gray key)
- Smooths temporally:
  `exposure = mix(prevExposure, targetExposure, 1 - exp(-dt * adaptSpeed))`
- Writes to `ExposureSSBO` (a single float read by the tone mapping shader)

**UI parameters**: Min/Max EV range (controls `minLogLum`/`logLumRange`), adapt speed.

---

## Effect 2: SSAO (Screen-Space Ambient Occlusion)

**Files**: `SSAO.h/cpp`, `gtao.comp`, `gtao_blur.comp`

**Purpose**: Darkens corners, crevices, and contact areas to simulate global illumination occlusion.

**Algorithm** (GTAO-based, three compute dispatches):

### Pass A - GTAO (`gtao.comp`, workgroup 8x8)

- Reads the **single-sample depth buffer** (not HDR) via `texelFetch`
- **Depth linearization**: Multiplies `invProjection` by clip-space
  `(0, 0, depthValue, 1)` (Vulkan [0,1] depth convention)
- **View-space reconstruction**: Uses `projInfo`
  (pixel-to-view-space coefficients) to compute `(x, y, z)` in view space
- **Normal estimation**: Forward-difference cross product of neighboring pixels
- For each pixel, casts 4 directions x 4 steps,
  computing horizon angles via `sinH = dot(diff, N) / dist`
- Distance falloff converts pixel radius to view-space radius:
  `viewRadius = radius * |z| * projInfo.x`
- Output: single-channel R8 AO image.
  `ao = 1.0` = no occlusion, `ao < 1.0` = occluded.

### Pass B - Horizontal blur (`gtao_blur.comp`)

- Depth-aware bilateral blur with direction `(1, 0)`.
  Reads AO image + depth, writes to temp image.
- Uses Gaussian weights `[0.324, 0.232, 0.0855, 0.0205]`
  with a depth threshold rejection (`step(depthDiff, threshold)`)
  to preserve edges.

### Pass C - Vertical blur (`gtao_blur.comp`)

- Same shader, direction `(0, 1)`.
  Reads temp image + depth, writes back to AO image.

**UI parameters**: Radius (pixels), bias, intensity.

---

## Effect 3: Bloom

**Files**: `Bloom.h/cpp`, `bloom_downsample.comp`, `bloom_upsample.comp`

**Purpose**: Creates a soft glow around bright areas, simulating lens scattering.

**Algorithm** (6-level mip chain, compute-based):

The bloom image is a single `R16G16B16A16_SFLOAT` image with 6 mip levels.

### Downsample chain (6 dispatches, mip 0 through 5)

```
HDR (full res) -> mip0 (half) -> mip1 (quarter) -> ... -> mip5 (1/64)
```

`bloom_downsample.comp` uses a **13-tap filter** (from the Call of Duty: Advanced Warfare talk). For the first mip (mip 0), it applies **Karis averaging** to prevent fireflies:

```glsl
float KarisAverage(vec3 c) {
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return 1.0 / (1.0 + lum);  // weight inversely proportional to brightness
}
```

This down-weights extremely bright pixels during the first downsample, preventing bright point lights from blooming disproportionately. Subsequent mips use a standard 13-tap weighted downsample.

### Upsample chain (5 dispatches, mip 5 back to mip 0)

```
mip5 -> +mip4 -> +mip3 -> +mip2 -> +mip1 -> +mip0
```

`bloom_upsample.comp` uses a **9-tap tent filter** and **additively blends** into the destination mip:

```glsl
vec3 existing = imageLoad(dstTexture, coord).rgb;
imageStore(dstTexture, coord, vec4(existing + upsample, 1.0));
```

This progressive accumulation produces the characteristic multi-scale bloom: near objects get a tight glow, far objects get a wide soft glow.

The final bloom result is `mMipViews[0]` (mip 0 of the bloom image).

**UI parameters**: Bloom strength (applied in the tone mapping shader as `hdr += bloom * bloomStrength`).

---

## Effect 4: Tone Mapping

**Files**: `ToneMapping.h/cpp`, `tonemap.frag`, `fullscreen.vert`

**Purpose**: Maps HDR linear radiance values to LDR [0,1] for display, applying the characteristic "filmic look".

**Rendering**: Fullscreen triangle pass (graphics pipeline, `vkCmdDraw(cmd, 3, 1, 0, 0)` with no vertex buffer). `fullscreen.vert` generates a triangle that covers the entire screen from `gl_VertexIndex`.

**Inputs** (all combined in the fragment shader):

- `hdrColor` (binding 0) - the HDR image
- `bloomTex` (binding 1) - the bloom result (or black placeholder)
- `aoTex` (binding 2) - the AO texture (or white placeholder)
- `ExposureSSBO` (binding 3) - the auto-exposure value

### Shader logic (`tonemap.frag`)

```glsl
hdr *= ao;                                      // apply ambient occlusion
hdr += bloom * bloomStrength;                   // add bloom
hdr *= exposureValue * exp2(exposureBias);      // apply exposure

if (curveType == 0)  mapped = ACESFilmic(hdr);
else                 mapped = AgXToneMap(hdr);
```

### ACES Filmic

A configurable Uncharted 2-style filmic curve:

```glsl
vec3 ACESFilmic(vec3 x) {
    // f(x) = (x*(A*x + C*B) + D*0.02) / (x*(A*x + B) + D*0.3) - 0.02/0.3
    // Normalized by the white point W
}
```

Parameters: shoulder strength (A), linear strength (B), linear angle (C), toe strength (D), white point (W). The S-curve compresses highlights (shoulder), lifts shadows (toe), and has a linear middle section.

### AgX

A more modern curve that preserves hue in highlights:

```glsl
vec3 AgXToneMap(vec3 color) {
    color = agxTransform * color;           // rotate to AgX log-space
    color = clamp(log2(color), minEV, maxEV);
    color = (color - minEV) / (maxEV - minEV);
    color = AgXDefaultContrastApprox(color); // 6th-order polynomial S-curve
    color = agxTransformInv * color;         // rotate back
    // Apply saturation and punch
}
```

Parameters: saturation, punch (contrast boost).

**Output**: Written to either the LDR intermediate image (if color grading follows) or directly to the swapchain.

**UI**: Curve selector (ACES/AgX), exposure bias, per-curve sliders, reset buttons.

---

## Effect 5: Color Grading

**Files**: `ColorGrading.h/cpp`, `color_grading.frag`, `fullscreen.vert`

**Purpose**: Final artistic adjustments applied after tone mapping.

**Rendering**: Another fullscreen triangle pass, reading the LDR intermediate and writing to the swapchain.

### Shader effects (applied in order in `color_grading.frag`)

1. **Chromatic Aberration**: Offsets R and B channels radially from the screen center.
   Offset magnitude increases with distance from center:
   `offset = dir * dist * chromaticAberration * 0.01`.

2. **3D LUT Color Grading**: A 32x32x32 3D texture lookup.
   The current LDR color is used as a coordinate into the LUT.
   Strength is blendable: `mix(color, graded, lutStrength)`.
   (Default LUT is identity.)

3. **Vignette**: Radial darkening from screen edges using `smoothstep`
   based on distance from center.

4. **Film Grain**: Random noise scaled by a time-varying seed,
   applied multiplicatively: `color += color * grain * grainStrength`.

**UI parameters**: LUT strength, vignette intensity/radius, film grain strength, chromatic aberration.

---

## Execution Flow in `PostProcessPass::Execute`

The orchestrator in `PostProcessPass.cpp` ties everything together:

1. `TransitionPlaceholders(cmd)` - one-time init of fallback textures
2. **AutoExposure** compute dispatch (if enabled)
3. **SSAO** compute dispatch (if enabled)
4. **Bloom** compute dispatch (if enabled)
5. Resolve fallback views:
   - `aoView` = SSAO result or white placeholder
   - `bloomView` = bloom result or black placeholder
6. Build `ToneMappingPushConstants` from settings
7. **If color grading is enabled**:
   - Transition LDR image to `COLOR_ATTACHMENT_OPTIMAL`
   - Tone map HDR -> LDR intermediate
   - Transition LDR image to `SHADER_READ_ONLY_OPTIMAL`
   - Color grade LDR -> swapchain
8. **Else**:
   - Tone map HDR -> swapchain directly

---

## Render Graph Integration

In `Application::BuildAndExecuteRenderGraph`, the post-process pass is registered as:

```
graph.Read(hdrResource)        <- depends on ForwardPass writing HDR
graph.Read(depthResource)      <- depends on ForwardPass writing depth (for SSAO)
graph.Write(swapchainResource) <- final output
```

The render graph automatically inserts barriers to transition images between passes (e.g., HDR from `COLOR_ATTACHMENT_OPTIMAL` to `SHADER_READ_ONLY_OPTIMAL` before post-processing reads it).

---

## Descriptor Management

All 5 effects use `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT` on their descriptor sets, with corresponding pool and layout flags. This allows descriptor updates (rebinding the HDR view, depth view, etc.) while previous frames' command buffers may still be in flight (since `FRAMES_IN_FLIGHT = 2`).
