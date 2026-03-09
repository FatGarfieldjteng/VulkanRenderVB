# Color Grading — Design and Implementation

Color grading is the **final** post-processing step. It takes the tone-mapped LDR image and applies cinematic color adjustments before presenting to the screen. It runs as a fullscreen triangle graphics pass (not compute) with a fragment shader.

## Position in the Pipeline

```
HDR Image → [AutoExposure] → [SSAO] → [Bloom] → [Tone Mapping → LDR] → [Color Grading] → Swapchain
```

When color grading is enabled, tone mapping writes to an **intermediate LDR image** instead
of directly to the swapchain. Color grading then reads that LDR image and writes the final
result to the swapchain:

```cpp
// Tone map → LDR intermediate
tonemap->Draw(cmd, ldrView, ...);

// Transition LDR: color attachment → shader read
TransitionImage(cmd, ldrImage, ...COLOR_ATTACHMENT → SHADER_READ_ONLY...);

// Color grade → swapchain
cg->Draw(cmd, swapchainView, extent, ldrView, cgPC);
```

When color grading is disabled, tone mapping writes directly to the swapchain, skipping the
intermediate image entirely.

## Four Effects in One Pass

The fragment shader (`color_grading.frag`) applies four effects sequentially on every pixel:

### 1. Chromatic Aberration

Simulates the color fringing seen in real camera lenses, where different wavelengths of
light refract at slightly different angles. The red, green, and blue channels are sampled at
slightly offset UV coordinates:

```glsl
vec2 dir = uv - 0.5;                                      // direction from screen center
float dist = length(dir);                                   // distance from center
vec2 offset = dir * dist * chromaticAberration * 0.01;     // offset scales with distance

color.r = texture(inputColor, uv + offset).r;   // red shifted outward
color.g = texture(inputColor, uv).g;             // green stays centered
color.b = texture(inputColor, uv - offset).b;    // blue shifted inward
```

Key properties:

- The offset is **radial** — it points away from the screen center, mimicking real lens
  optics.
- It scales with `dist` (distance from center), so the center of the screen has zero
  aberration and the edges have maximum. This matches real lenses where aberration is worst
  at the periphery.
- The `* 0.01` factor keeps the UI slider range reasonable (0–5 maps to 0–0.05 UV offset
  at the corners).
- Red shifts outward, blue shifts inward — this matches the typical dispersion order of
  glass (shorter wavelengths bend more).
- The offset uses `dir * dist` (not just `dir`), which produces a **quadratic falloff**
  (`dist²`). Since `dir` already has magnitude `dist`, multiplying by `dist` again gives
  magnitude `dist²`. This matches real lateral chromatic aberration, which increases
  roughly quadratically with distance from the optical axis.

When `chromaticAberration = 0`, the branch is skipped and a single texture fetch is used.

#### Concrete example

For a pixel at UV (0.9, 0.1) — near the top-right corner, with `chromaticAberration = 3.0`:

```
dir    = (0.9, 0.1) - (0.5, 0.5) = (0.4, -0.4)
dist   = sqrt(0.4² + 0.4²) ≈ 0.566
offset = (0.4, -0.4) × 0.566 × 3.0 × 0.01 = (0.00679, -0.00679)
```

In pixel coordinates (1920×1080):

| Channel | Sample UV          | Sample pixel | Shift             |
|---------|--------------------|--------------|-------------------|
| Red     | (0.9068, 0.0932)   | (1741, 101)  | 13 pixels outward |
| Green   | (0.9000, 0.1000)   | (1728, 108)  | no shift          |
| Blue    | (0.8932, 0.1068)   | (1715, 115)  | 13 pixels inward  |

For a pixel near screen center at UV (0.52, 0.48), the R/G/B samples would be less than
0.03 pixels apart — completely invisible. This is why chromatic aberration has zero visible
effect at the screen center and increases toward the edges.

### 2. 3D LUT Color Grading

A **Look-Up Table** (LUT) remaps every input color to a different output color. In film and
photography, LUTs are used to apply a "look" — warm/cool tint, desaturated shadows,
crushed blacks, etc.

```glsl
if (lutStrength > 0.0) {
    vec3 lutCoord = clamp(color, 0.0, 1.0);
    float lutSize = 32.0;
    lutCoord = lutCoord * ((lutSize - 1.0) / lutSize) + 0.5 / lutSize;
    vec3 graded = texture(colorLUT, lutCoord).rgb;
    color = mix(color, graded, lutStrength);
}
```

How it works:

- The input color `(R, G, B)` is used directly as **3D texture coordinates** into a
  `sampler3D`. The red channel maps to the X axis, green to Y, blue to Z.
- A 32×32×32 3D texture can represent any arbitrary color transformation — each of the
  32,768 entries says "if the input is this color, the output should be that color."
- `lutStrength` controls blend: 0 = no LUT effect, 1 = full LUT effect. Intermediate
  values blend between original and graded.

Currently, the engine creates a **1×1×1 placeholder LUT** (identity — no color change).
This is the infrastructure for loading custom LUT files (e.g., `.cube` files exported from
DaVinci Resolve or Photoshop). With the placeholder, `lutStrength = 0` effectively disables
this feature.

#### Texel center correction

The formula `lutCoord * ((lutSize - 1.0) / lutSize) + 0.5 / lutSize` maps the color range
[0, 1] to **texel centers** inside the 3D LUT texture. Without it, you'd sample at the
very edge of the texture.

In a 32-texel texture, texture coordinates [0, 1] span the full extent, not texel centers:

```
Texel index:     0      1      2     ...    30     31
Texel center:  0.5/32  1.5/32  2.5/32      30.5/32  31.5/32
               = 0.016  0.047  0.078       0.953    0.984

Texture coord:  0.0                                      1.0
                |---[0]---[1]---[2]--- ... ---[30]---[31]---|
                ^                                          ^
                edge                                      edge
```

The formula remaps [0, 1] to [0.5/32, 31.5/32]:

- **`(N-1) / N = 31/32`**: The distance from first texel center to last spans `(N-1)`
  texels out of `N`, so the UV range is `31/32 = 0.96875`.
- **`0.5 / N = 0.5/32`**: Offsets to the first texel center at `0.5/32`.

| Input color | After correction | Maps to            |
|-------------|------------------|--------------------|
| 0.0         | 0.015625         | Center of texel 0  |
| 0.5         | 0.5              | Between texels 15-16 |
| 1.0         | 0.984375         | Center of texel 31 |

### 3. Vignette

Darkens the edges and corners of the screen, drawing the viewer's eye toward the center.
Common in film and photography (both as an artistic choice and as a natural lens artifact).

```glsl
vec2 center = uv - 0.5;
float vignette = 1.0 - smoothstep(vignetteRadius, vignetteRadius + 0.3,
                                   length(center) * 1.414);
color *= mix(1.0, vignette, vignetteIntensity);
```

How it works step by step:

1. **`uv - 0.5`**: Moves the origin to the screen center, so `center` is a vector
   pointing from center to the current pixel.

2. **`length(center) * 1.414`**: Computes distance from center, normalized so corners
   reach exactly 1.0. The maximum `length(center)` is at corners:
   `sqrt(0.5² + 0.5²) ≈ 0.707`. Multiplying by `1.414 (= √2 ≈ 1/0.707)` normalizes
   so corners = 1.0.

   | Location            | `length(center)` | `× 1.414` |
   |---------------------|-------------------|-----------|
   | Dead center         | 0.0               | 0.0       |
   | Middle of edge      | 0.5               | 0.707     |
   | Corner              | 0.707             | 1.0       |

3. **`smoothstep(vignetteRadius, vignetteRadius + 0.3, ...)`**: Creates a smooth transition
   band. With `vignetteRadius = 0.5`:
   - Distance ≤ 0.5: `smoothstep` returns 0 → `vignette = 1.0` (full bright)
   - Distance 0.5–0.8: smooth cubic ramp → gradual darkening
   - Distance ≥ 0.8: `smoothstep` returns 1 → `vignette = 0.0` (fully dark)

   The `0.3` is the hardcoded width of the transition band.

   ```
   vignette value across screen:

   1.0 |_______
       |       \
       |        \
       |         \___________
   0.0 |
       +----+--------+------→ normalized distance
          0.5      0.8
       (radius)  (radius+0.3)
   ```

4. **`mix(1.0, vignette, vignetteIntensity)`**: Controls strength. At intensity 0, the
   multiplier is always 1.0 (no effect). At intensity 1, the full vignette is applied.

### 4. Film Grain

Adds random noise that simulates the grain pattern of analog film stock. Gives the image a
more organic, cinematic texture.

```glsl
float grain = Random(uv + vec2(grainTime)) * 2.0 - 1.0;   // random in [-1, 1]
color += color * grain * grainStrength;
```

How it works:

- `Random()` is a classic pseudo-random hash function based on `sin(dot(...))`. It produces
  a deterministic but visually random value for each UV coordinate.
- `grainTime` (derived from `deltaTime * 1000`) shifts the random pattern every frame, so
  the grain animates like real film grain rather than appearing as a static overlay.
- The grain is **multiplicative** (`color * grain * strength`), meaning bright areas get
  more visible grain and dark areas get less — matching the behavior of real film stock
  where grain is more visible in mid-tones and highlights.
- `grainStrength` controls intensity: 0 = no grain, 0.1 = subtle texture, 0.2 = heavy
  grain.

## Rendering Technique: Fullscreen Triangle

Color grading uses a **fullscreen triangle** rather than a fullscreen quad. The vertex
shader generates a single oversized triangle from `gl_VertexIndex` with no vertex buffer:

```glsl
void main() {
    fragUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(fragUV * 2.0 - 1.0, 0.0, 1.0);
}
```

For vertices 0, 1, 2 this produces:

- Vertex 0: UV(0,0), Position(-1,-1) — bottom-left
- Vertex 1: UV(2,0), Position(3,-1) — far right
- Vertex 2: UV(0,2), Position(-1,3) — far top

This single triangle covers the entire screen (and beyond — the GPU clips the excess). It's
preferred over a quad (two triangles) because it avoids the diagonal seam where two
triangles meet, which can cause inefficiency in the GPU's rasterizer due to partial quad
shading along the diagonal.

## Push Constants

All parameters are passed via push constants (no uniform buffer needed):

```cpp
struct ColorGradingPushConstants {
    float    lutStrength;           // 0..1, blend with LUT
    float    vignetteIntensity;     // 0..1, edge darkening strength
    float    vignetteRadius;        // 0.1..1.0, where darkening begins
    float    grainStrength;         // 0..0.2, film grain amount
    float    grainTime;             // animated seed for grain randomness
    float    chromaticAberration;   // 0..5, color fringing amount
    uint32_t resolutionX;           // screen width
    uint32_t resolutionY;           // screen height
};
```

Push constants are ideal here because the data is small (32 bytes) and changes every frame
(due to `grainTime`).

## UI Parameters

| Parameter            | Range    | Default | Effect                                          |
|----------------------|----------|---------|--------------------------------------------------|
| LUT Strength         | 0.0–1.0  | 0.0     | Blend with 3D color LUT (placeholder = no effect)|
| Vignette             | 0.0–1.0  | 0.0     | Edge/corner darkening intensity                  |
| Vignette Radius      | 0.1–1.0  | 0.5     | How far from center darkening begins             |
| Film Grain           | 0.0–0.2  | 0.0     | Noise overlay strength                           |
| Chromatic Aberration | 0.0–5.0  | 0.0     | Color fringing at screen edges                   |

All effects default to 0 (disabled). They are purely artistic tools — none of them are
physically necessary for correct rendering, but they add cinematic quality to the final
image.

---

## Q&A

### Q: In chromatic aberration, can the offset use `dir * chromaticAberration * 0.01` instead of `dir * dist * chromaticAberration * 0.01`?

### A:

Yes. The difference is in how the effect scales with distance from center:

**`dir * dist * ...` (current code) — quadratic falloff:**

Since `length(dir) = dist`, the offset magnitude is `dist²`. The effect grows slowly near
center and accelerates toward the edges. This more closely matches real lateral chromatic
aberration in camera lenses, where the aberration increases roughly quadratically with
distance from the optical axis.

| Distance from center | Offset magnitude (relative) |
|----------------------|-----------------------------|
| 0.0 (center)         | 0.0                         |
| 0.25                 | 0.0625 (6.25%)              |
| 0.5                  | 0.25 (25%)                  |
| 0.707 (corner)       | 0.5 (50%)                   |

**`dir * ...` (without `dist`) — linear falloff:**

The offset magnitude is simply `dist`. The effect grows uniformly from center to edge. The
fringing would be more noticeable at mid-screen distances.

| Distance from center | Offset magnitude (relative) |
|----------------------|-----------------------------|
| 0.0 (center)         | 0.0                         |
| 0.25                 | 0.25 (25%)                  |
| 0.5                  | 0.5 (50%)                   |
| 0.707 (corner)       | 0.707 (70.7%)               |

Neither is physically wrong — both produce zero aberration at center and increasing
aberration at edges. The quadratic version is more physically accurate. The linear version
is a perfectly valid artistic choice that many game engines use. You'd just need to adjust
the slider value since the linear version produces stronger fringing at the same setting.

### Q: Explain `lutCoord = lutCoord * ((lutSize - 1.0) / lutSize) + 0.5 / lutSize` — what is this math?

### A:

This formula maps the color range [0, 1] to **texel centers** inside the 3D LUT texture.
Without it, you'd sample at the very edge of the texture, producing incorrect colors.

In a 32-texel texture, texture coordinates [0, 1] span the full extent, not texel centers:

```
Texel index:     0      1      2     ...    30     31
Texel center:  0.5/32  1.5/32  2.5/32      30.5/32  31.5/32
               = 0.016  0.047  0.078       0.953    0.984

Texture coord:  0.0                                      1.0
                |---[0]---[1]---[2]--- ... ---[30]---[31]---|
                ^                                          ^
                edge                                      edge
```

Sampling at UV 0.0 hits the left edge of texel 0, not its center. The formula remaps
[0, 1] to [0.5/32, 31.5/32]:

- **`(N-1) / N = 31/32`**: Scales the range. First-to-last texel center spans `(N-1)`
  texels out of `N`, giving UV range `31/32 = 0.96875`.
- **`0.5 / N = 0.5/32`**: Offsets to the first texel center.

| Input color | After correction | Maps to              |
|-------------|------------------|----------------------|
| 0.0         | 0.015625         | Center of texel 0    |
| 0.5         | 0.5              | Between texels 15-16 |
| 1.0         | 0.984375         | Center of texel 31   |

Without correction, intermediate color values would be slightly off, producing subtle color
shifts especially for very dark and very bright colors at the texture boundaries.

### Q: Explain the vignette computation in detail.

### A:

```glsl
vec2 center = uv - 0.5;
float vignette = 1.0 - smoothstep(vignetteRadius, vignetteRadius + 0.3,
                                   length(center) * 1.414);
color *= mix(1.0, vignette, vignetteIntensity);
```

**Step 1 — `uv - 0.5`**: Moves the origin to the screen center, so `center` is a vector
pointing from center to the current pixel.

**Step 2 — `length(center) * 1.414`**: Computes distance from center, normalized so corners
reach exactly 1.0. The maximum `length(center)` at corners is `sqrt(0.5² + 0.5²) ≈ 0.707`.
Multiplying by `1.414 (= √2 ≈ 1/0.707)` normalizes so corners = 1.0.

| Location         | `length(center)` | `× 1.414` |
|------------------|-------------------|-----------|
| Dead center      | 0.0               | 0.0       |
| Middle of edge   | 0.5               | 0.707     |
| Corner           | 0.707             | 1.0       |

**Step 3 — `smoothstep`**: Creates a smooth transition band. `smoothstep(edge0, edge1, x)`
returns 0 when `x ≤ edge0`, 1 when `x ≥ edge1`, and smooth cubic interpolation in between.
With `vignetteRadius = 0.5`:

- Distance ≤ 0.5: returns 0 → `vignette = 1.0` (full bright)
- Distance 0.5–0.8: smooth cubic ramp → gradual darkening
- Distance ≥ 0.8: returns 1 → `vignette = 0.0` (fully dark)

```
vignette value across screen:

1.0 |_______
    |       \
    |        \
    |         \___________
0.0 |
    +----+--------+------→ normalized distance
       0.5      0.8
    (radius)  (radius+0.3)
```

The `0.3` is the hardcoded width of the transition band. `vignetteRadius` controls where
darkening begins (UI slider). Smaller radius = darkening starts closer to center.

**Step 4 — `mix(1.0, vignette, vignetteIntensity)`**: Controls strength. At intensity 0,
multiplier is always 1.0 (no effect). At intensity 1, full vignette is applied. Intermediate
values blend between no effect and full vignette.
