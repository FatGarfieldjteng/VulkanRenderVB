# SSAO (Screen-Space Ambient Occlusion) — GTAO Implementation

## What is SSAO?

SSAO simulates **contact shadows** — the soft darkening that occurs in crevices, corners, and where surfaces meet. In real life, these areas receive less ambient light because surrounding geometry blocks part of the incoming light hemisphere. Without SSAO, a scene looks flat and objects appear to "float" rather than sit on surfaces.

SSAO is "screen-space" because it only uses information available in the 2D depth buffer — no knowledge of the actual 3D scene geometry.

## The Algorithm: GTAO (Ground Truth Ambient Occlusion)

This engine uses **GTAO**, a horizon-based technique. The core idea: for each pixel, look outward in several directions across the depth buffer. If nearby pixels represent geometry that is **above** the current pixel's surface (closer to the camera), they occlude ambient light. The more the surrounding geometry rises above the surface plane, the darker the occlusion.

## High-Level Flow

```
Depth Buffer
    |
    v
[GTAO Compute]         8x8 workgroups, full resolution
    |  For each pixel: reconstruct view-space position/normal,
    |  cast rays in 4 directions, measure horizon occlusion
    v
Raw AO Image (R8_UNORM, noisy)
    |
    v
[Blur Horizontal]      depth-aware bilateral blur
    |
    v
Temp AO Image
    |
    v
[Blur Vertical]        depth-aware bilateral blur
    |
    v
Final AO Image (R8_UNORM, smooth)
    |
    read by tonemap.frag: hdr *= ao;
```

## GPU Resources

Two `R8_UNORM` images at full resolution:

- **`mAOImage`**: The main AO output (and final blurred result)
- **`mAOTempImage`**: Intermediate for the separable blur

`R8_UNORM` means a single 8-bit channel storing values in [0, 1]:
- 1.0 = fully lit (no occlusion)
- 0.0 = fully occluded (maximum shadow)

## The GTAO Shader (`gtao.comp`)

**Workgroup**: 8x8 threads. Each thread processes one pixel.

### Step 1: Reconstruct view-space position

```glsl
float LinearizeDepth(float d) {
    vec4 clip = vec4(0.0, 0.0, d, 1.0);
    vec4 view = invProjection * clip;
    return -view.z / view.w;
}
```

The depth buffer stores non-linear depth in [0, 1] (Vulkan NDC). `LinearizeDepth` converts this to a linear view-space distance by multiplying by the inverse projection matrix. The `d` value is used directly (not `d * 2 - 1`) because Vulkan uses [0, 1] depth range, not [-1, 1] like OpenGL.

```glsl
vec3 GetViewPos(ivec2 coord) {
    float depth = texelFetch(depthTex, coord, 0).r;
    float z = LinearizeDepth(depth);
    vec2 pixelCenter = vec2(coord) + 0.5;
    vec3 viewPos;
    viewPos.x = (pixelCenter.x * projInfo.x + projInfo.z) * z;
    viewPos.y = (pixelCenter.y * projInfo.y + projInfo.w) * z;
    viewPos.z = -z;
    return viewPos;
}
```

This reconstructs the 3D view-space position from a 2D pixel coordinate and depth. The `projInfo` values are precomputed on the CPU:

```cpp
projInfo[0] = 2.0 / (width * P00)     // x: pixel-to-NDC scale for x
projInfo[1] = 2.0 / (height * P11)    // y: pixel-to-NDC scale for y
projInfo[2] = -(1 - P02) / P00        // z: NDC offset for x
projInfo[3] = -(1 + P12) / P11        // w: NDC offset for y
```

Where `P00`, `P11`, `P02`, `P12` are elements of the projection matrix. This avoids a full matrix multiply per pixel — instead it's just 2 multiply-adds per axis.

### Step 2: Reconstruct surface normal

```glsl
ivec2 dxCoord = clamp(coord + ivec2(1, 0), ivec2(0), ivec2(resolution) - 1);
ivec2 dyCoord = clamp(coord + ivec2(0, 1), ivec2(0), ivec2(resolution) - 1);
vec3 dPdx = GetViewPos(dxCoord) - P;
vec3 dPdy = GetViewPos(dyCoord) - P;
vec3 N = normalize(cross(dPdy, dPdx));
```

Since SSAO only has the depth buffer (no G-buffer normal), it reconstructs the surface normal using **finite differences**: the cross product of the horizontal and vertical position gradients gives the surface normal. This works because the position gradient along the screen is tangent to the surface.

### Step 3: Early-out for far plane

```glsl
if (-P.z >= farPlane * 0.99) {
    imageStore(aoOutput, coord, vec4(1.0));
    return;
}
```

Pixels at the far plane (sky, background) shouldn't receive AO — they output 1.0 (fully lit).

### Step 4: Generate noise for randomization

```glsl
float noise = InterleavedGradientNoise(vec2(coord));
float rotAngle = noise * PI;
```

`InterleavedGradientNoise` produces a deterministic but visually random value per pixel. This rotates the sample directions differently for each pixel, which:
- Breaks up the regular pattern that would appear with uniform directions
- Converts structured aliasing into noise, which the subsequent blur can smooth out
- Uses no memory (unlike a noise texture) — it's purely computational

### Step 5: Convert radius to view-space

```glsl
float viewRadius = radius * abs(P.z) * projInfo.x;
```

The `radius` parameter (from the UI, in pixels) must be converted to view-space units for the distance falloff calculation. A pixel at a larger depth represents a larger physical area, so `radius` is scaled by depth (`P.z`) and the projection factor (`projInfo.x`).

### Step 6: Horizon search — the core algorithm

```glsl
float ao = 0.0;
for (int d = 0; d < DIRECTION_COUNT; d++) {          // 4 directions
    float angle = (float(d) + noise) / float(DIRECTION_COUNT) * PI;
    vec2 dir = vec2(cos(angle + rotAngle), sin(angle + rotAngle));

    float maxHorizon = -1.0;
    for (int s = 1; s <= STEP_COUNT; s++) {            // 4 steps per direction
        vec2 offset = dir * (float(s) / float(STEP_COUNT)) * radius;
        ivec2 sampleCoord = coord + ivec2(offset);
        sampleCoord = clamp(sampleCoord, ivec2(0), ivec2(resolution) - 1);

        vec3 S = GetViewPos(sampleCoord);
        vec3 diff = S - P;
        float dist = length(diff);
        if (dist < 1e-4) continue;

        float sinH = dot(diff, N) / dist;
        float falloff = clamp(1.0 - dist*dist / (viewRadius*viewRadius), 0.0, 1.0);
        maxHorizon = max(maxHorizon, (sinH - bias) * falloff);
    }
    ao += max(0.0, maxHorizon);
}
```

For each of 4 directions:
1. **Choose a direction**: Evenly spaced around a semicircle (π radians), jittered by noise. The semicircle (not full circle) is sufficient because occlusion is symmetric.
2. **March along the direction** in 4 steps, each farther from the center pixel.
3. At each step, **reconstruct the 3D position** of the sample.
4. **Compute the sine of the horizon angle**: `sinH = dot(diff, N) / dist`. This measures how far above the surface plane the sample is. If `sinH > 0`, the sample is above the surface → it occludes light. If `sinH ≤ 0`, the sample is below the surface → no occlusion.
5. **Apply distance falloff**: `1 - dist²/viewRadius²`. Samples far from the center pixel contribute less. This prevents distant geometry from incorrectly occluding.
6. **Subtract bias**: `sinH - bias` prevents self-occlusion on flat surfaces. Without bias, numerical imprecision would make flat surfaces appear slightly occluded.
7. **Track the maximum horizon** across all steps. The maximum (not average) is used because occlusion is determined by the highest occluder — if one sample blocks light, it doesn't matter that others don't.

The total per-pixel cost: 4 directions × 4 steps = **16 depth buffer fetches + 16 view-space reconstructions**.

### Step 7: Compute final AO

```glsl
ao = 1.0 - (ao / float(DIRECTION_COUNT)) * intensity;
ao = clamp(ao, 0.0, 1.0);
```

- Divide by direction count to average across directions
- Multiply by `intensity` (UI parameter, controls strength)
- Invert: 1.0 = lit, 0.0 = occluded
- Clamp to [0, 1]

## The Blur Shader (`gtao_blur.comp`)

The raw GTAO output is noisy because each pixel uses only 16 samples with random jitter. A **depth-aware bilateral blur** smooths the noise while preserving edges.

The blur is **separable** — split into horizontal and vertical passes, reusing the same shader with a different `direction` push constant:

- Pass 1: direction = (1, 0) → horizontal, writes to `mAOTempImage`
- Pass 2: direction = (0, 1) → vertical, writes back to `mAOImage`

```glsl
float centerDepth = texture(depthTex, uv).r;
float centerAO = texture(aoInput, uv).r;

float totalAO = centerAO;
float totalWeight = 1.0;

const float weights[4] = float[](0.324, 0.232, 0.0855, 0.0205);

for (int i = 1; i <= 3; i++) {
    for (int sign = -1; sign <= 1; sign += 2) {
        vec2 sampleUV = uv + direction * texelSize * float(i * sign);
        float sampleDepth = texture(depthTex, sampleUV).r;
        float sampleAO = texture(aoInput, sampleUV).r;

        float depthDiff = abs(centerDepth - sampleDepth);
        float w = weights[i] * step(depthDiff, depthThreshold);

        totalAO += sampleAO * w;
        totalWeight += w;
    }
}

float result = totalAO / totalWeight;
```

Key aspects:

- **Gaussian weights** `[0.324, 0.232, 0.0855, 0.0205]` approximate a Gaussian kernel. The center pixel (implicit weight index 0 = 0.324 equivalent as initial 1.0 / totalWeight) gets the most weight, falling off for further samples.
- **Depth-aware**: `step(depthDiff, depthThreshold)` returns 0 if the depth difference exceeds the threshold (0.02). This means samples across a depth discontinuity (e.g., an object edge in front of a distant wall) are **rejected** — their AO is not blended in. This preserves sharp AO boundaries at object silhouettes.
- **Normalized**: `totalAO / totalWeight` accounts for rejected samples. If 2 of 6 neighbors are rejected, the remaining 4 are re-normalized so the result doesn't darken.

## How AO Is Consumed

In `tonemap.frag`:

```glsl
float ao = texture(aoTex, fragUV).r;
hdr *= ao;
```

The AO value (0–1) multiplicatively darkens the HDR color before tone mapping. Fully occluded regions (ao = 0) become black. Fully lit regions (ao = 1) are unaffected. This is applied in linear HDR space before the tone curve, so the darkening interacts correctly with the non-linear tone mapping.

When SSAO is disabled, a white placeholder texture (1.0 everywhere) is passed instead, making the multiplication a no-op.

## UI Parameters

| Parameter    | Range     | Default | Effect                                           |
|--------------|-----------|---------|--------------------------------------------------|
| AO Radius    | 1–100     | 16      | Search radius in pixels. Larger = wider shadows  |
| AO Bias      | 0.0–0.5   | 0.01    | Prevents self-occlusion on flat surfaces         |
| AO Intensity | 0.0–5.0   | 1.5     | Strength of darkening. 0 = no effect             |

## Summary of the 3-Pass Pipeline

| Pass | Shader | Input | Output | Cost |
|---|---|---|---|---|
| GTAO | `gtao.comp` | Depth buffer | Raw AO (noisy) | 16 depth fetches/pixel |
| Blur H | `gtao_blur.comp` dir=(1,0) | Raw AO + Depth | Temp AO | 7 taps/pixel |
| Blur V | `gtao_blur.comp` dir=(0,1) | Temp AO + Depth | Final AO (smooth) | 7 taps/pixel |

Total per frame: 3 compute dispatches at full resolution, producing a single-channel 8-bit AO texture consumed by tone mapping.

---

## Q&A

### Q: How do these three lines of code recover camera-space depth from the depth buffer?

```glsl
vec4 clip = vec4(0.0, 0.0, d, 1.0);
vec4 view = invProjection * clip;
return -view.z / view.w;
```

**A:**

**Line 1**: `vec4 clip = vec4(0.0, 0.0, d, 1.0);`

During rendering, the GPU transformed every vertex from camera space to clip space using the projection matrix:

```
clip = projection * cameraSpacePoint
```

Then it performed a **perspective divide** to get what's stored in the depth buffer:

```
d = clip.z / clip.w
```

This divide destroyed the original `clip.z` and `clip.w` — we only have their ratio `d`. But in homogeneous coordinates, the point `(0, 0, d, 1)` represents the **same depth** as the original clip-space point. We set x and y to zero because we only care about depth.

**Line 2**: `vec4 view = invProjection * clip;`

This reverses the projection. It transforms from clip space back to camera space. But because we lost information in the perspective divide, the result is a **scaled** version of the original camera-space point — not the original point itself. The true camera-space z is hidden inside the ratio `view.z / view.w`.

**Line 3**: `-view.z / view.w`

This homogeneous divide **undoes the scaling** and extracts the true camera-space depth. The minus sign is because in a right-handed coordinate system, the camera looks along `-Z`, so `z_v` is negative for visible objects. The negation makes the result a positive distance.

**In summary:**

```
Rendering (forward):    cameraZ  →  projection  →  ÷ clip.w  →  d
Recovery (reverse):     d  →  (0,0,d,1)  →  invProjection  →  ÷ view.w  →  cameraZ
```

The projection matrix bakes depth into a nonlinear value `d` through a divide-by-w. The inverse projection matrix combined with a second divide-by-w **undoes both the matrix and the original divide**, recovering the linear camera-space depth.

### Q: Why reconstruct normals from depth instead of passing a normal buffer?

**A:**

**1. The engine uses forward rendering — there is no G-buffer.**

This engine uses a forward rendering pipeline: the PBR shader directly outputs to an HDR color image. There is no G-buffer that stores per-pixel normals. To provide normals, you would need to either add a geometry pass that renders all objects a second time to output view-space normals, or add a Multi-Render-Target (MRT) output to the forward pass — both of which add pipeline complexity and memory cost.

Using normal map textures directly also doesn't work because normal maps are in tangent space (relative to the triangle surface), not view space. Converting them requires the TBN matrix, which is only available during rasterization, not in a post-process compute shader.

**2. Depth-reconstructed normals are good enough for AO.**

SSAO operates at a coarse scale — it darkens crevices and contact areas with a search radius of ~16 pixels. At that scale, the geometric normal (flat surface orientation) is what matters, not fine detail from normal maps. Normal maps add small-scale perturbation (bumps, scratches) that would only affect AO at sub-pixel scales — well below the typical AO radius.

**3. The trade-off in practice.**

The depth reconstruction artifacts (slightly incorrect normals at silhouette edges) are largely hidden by the bilateral blur, which rejects samples across depth discontinuities anyway. If the engine moved to a deferred pipeline, normals would already be in the G-buffer for free and should be used instead.

### Q: Why can the following code recover view-space X and Y?

```glsl
vec2 pixelCenter = vec2(coord) + 0.5;
vec3 viewPos;
viewPos.x = (pixelCenter.x * projInfo.x + projInfo.z) * z;
viewPos.y = (pixelCenter.y * projInfo.y + projInfo.w) * z;
viewPos.z = -z;
```

**A:**

**The forward direction (what the GPU did during rendering):**

For the X axis, the rendering pipeline performed three steps:

```
Step 1 (projection):        clip.x = P00 * x_view
Step 2 (perspective divide): ndc.x = clip.x / clip.w = P00 * x_view / (-z_view)
Step 3 (viewport transform): pixel.x = (ndc.x + 1) / 2 * screenWidth
```

The framebuffer stores `pixel.x`. We want to recover `x_view`.

**The reverse (what the shader does):**

Undo step 3 — pixel back to NDC:

```
ndc.x = 2 * pixel.x / screenWidth - 1
```

Undo steps 2 and 1 — NDC back to camera space (using `z_linear = -z_view`):

```
x_view = ndc.x * z_linear / P00
```

Substituting the NDC formula:

```
x_view = (2 * pixel.x / screenWidth - 1) * z / P00
       = (pixel.x * 2/(screenWidth * P00) + (-1)/P00) * z
       = (pixel.x * projInfo.x            + projInfo.z) * z
```

That's exactly the code. And Y works identically with `P11`, `screenHeight`, `projInfo.y`, `projInfo.w`.

**What `projInfo` contains:**

| Element | Value | What it does |
|---|---|---|
| `projInfo.x` | `2 / (screenWidth * P00)` | Converts pixel.x to NDC, scaled by 1/P00 |
| `projInfo.y` | `2 / (screenHeight * P11)` | Same for Y axis |
| `projInfo.z` | `-1 / P00` | The NDC offset (-1) scaled by 1/P00 |
| `projInfo.w` | `-1 / P11` | Same for Y axis |

**Why multiply by `z` at the end:**

During rendering, the perspective divide divided by `clip.w = -z_view`. To undo that division, we multiply by `z_linear` (which equals `-z_view`). This is why objects further from the camera map to larger view-space coordinates for the same pixel offset — perspective.

**Why `+ 0.5`:**

`coord` is an integer pixel index (e.g., 0, 1, 2...). Adding 0.5 gives the **center** of the pixel (0.5, 1.5, 2.5...), which is where the depth sample actually represents.

**Why `viewPos.z = -z`:**

`z` is the positive linear distance from the camera. In a right-handed coordinate system, the camera looks along `-Z`, so the actual camera-space z coordinate is negative: `z_view = -z_linear`.

### Q: How does the normal reconstruction work, and can the estimated normal point in the wrong direction?

```glsl
ivec2 dxCoord = clamp(coord + ivec2(1, 0), ivec2(0), ivec2(resolution) - 1);
ivec2 dyCoord = clamp(coord + ivec2(0, 1), ivec2(0), ivec2(resolution) - 1);
vec3 dPdx = GetViewPos(dxCoord) - P;
vec3 dPdy = GetViewPos(dyCoord) - P;
vec3 N = normalize(cross(dPdy, dPdx));
```

**A:**

- `dPdx = GetViewPos(coord + (1,0)) - P` is the **tangent vector** along the screen-X direction in view space — how the 3D surface position changes when you move one pixel to the right.
- `dPdy = GetViewPos(coord + (0,1)) - P` is the **tangent vector** along the screen-Y direction in view space — how the surface position changes when you move one pixel down.
- `cross(dPdy, dPdx)` gives the vector perpendicular to both tangents, which is the **surface normal**.

This is the same as computing a partial derivative via finite differences: `dP/dx ≈ P(x+1) - P(x)`.

**The cross product order is deliberately chosen to produce camera-facing normals.**

The cross product is anti-commutative: `cross(A, B) = -cross(B, A)`. The code uses `cross(dPdy, dPdx)`, not `cross(dPdx, dPdy)`. This specific order was chosen to produce a normal with **positive Z** in view space (pointing toward the camera) for front-facing surfaces under this engine's coordinate convention (right-handed, camera looks along -Z, Vulkan Y-flip).

If someone swapped the order to `cross(dPdx, dPdy)`, every normal would be flipped, and the entire AO result would be wrong.

**Can it still go wrong at specific pixels?**

Yes, at **depth discontinuities** (silhouette edges). For example:

```
Pixel A: wall at depth 5
Pixel B (neighbor): background at depth 100
```

`dPdx = GetViewPos(B) - GetViewPos(A)` produces a huge vector pointing far into the background — not a real surface tangent at all. The resulting cross product gives a meaningless normal.

However, this doesn't cause visible artifacts because:

1. The **bilateral blur** rejects samples across depth discontinuities (the `step(depthDiff, depthThreshold)` in `gtao_blur.comp`), so the garbage AO at silhouette pixels is smoothed away.
2. The `max(0.0, maxHorizon)` clamp prevents negative occlusion values, so even with a wrong normal, the worst case is incorrect AO magnitude at a few edge pixels — not inverted darkening.

For the vast majority of pixels (smooth surface interiors), both neighbors are on the same surface, and the finite-difference normal is a good approximation.

### Q: What is InterleavedGradientNoise and what kind of noise pattern does it create?

```glsl
float InterleavedGradientNoise(vec2 coord) {
    return fract(52.9829189 * fract(0.06711056 * coord.x + 0.00583715 * coord.y));
}
```

**A:**

This is a noise function introduced by **Jorge Jimenez** (Call of Duty: Advanced Warfare). It produces a single float in [0, 1) from a 2D integer pixel coordinate.

**How it works:**

1. **Inner `fract`**: `fract(0.06711056 * x + 0.00583715 * y)` — a linear function of pixel coordinates, wrapped to [0, 1). The two constants are chosen so that neighboring pixels produce very different fractional parts.

2. **Outer `fract`**: `fract(52.9829189 * ...)` — amplifies and wraps again. The multiplication by ~53 spreads the small differences from neighboring pixels across the full [0, 1) range.

**Why the multiplication spreads small differences — with concrete numbers:**

The inner `fract` for two neighboring pixels (x=500, x=501 at y=100):

```
x = 500:  fract(34.139) = 0.139
x = 501:  fract(34.206) = 0.206
                           difference: 0.067  (small — only 6.7% of the range)
```

The outer multiply + `fract`:

```
x = 500:  fract(52.98 × 0.139) = fract(7.364)  = 0.364
x = 501:  fract(52.98 × 0.206) = fract(10.920) = 0.920
                                                   difference: 0.556  (large!)
```

The multiplication by 52.98 amplifies the tiny 0.067 step into a 3.556 step. Then `fract(3.556) = 0.556` — more than half the range. The magic constants are specifically chosen so that these fractional jumps are far from 0 and far from 1, ensuring neighboring pixels land in very different parts of the [0, 1) range.

**What pattern it creates:**

It creates a **gradient noise** — between white noise and blue noise. Visually it looks like a fine diagonal hatching pattern where adjacent pixels have very different values.

| Property | Why it matters for GTAO |
|---|---|
| **Deterministic** | Same pixel always gets the same value — no flickering between frames |
| **Neighboring pixels get very different values** | Each pixel rotates its sample directions by `noise × π`, breaking up banding |
| **No memory cost** | Purely arithmetic — no texture fetch, no VRAM |
| **Fast** | Just 2 multiplies, 2 `fract`s, and 1 add — cheaper than a texture sample |
| **Blur-friendly** | The pattern is well-suited to being cleaned up by a small spatial blur |

### Q: GTAO computes occlusion along 2D screen directions — what does "screen-space slice" mean and how does it approximate 3D occlusion?

**A:**

Imagine the screen as a 2D pixel grid. Pick a pixel P and a direction (say, "rightward"). Now look at the row of pixels going right from P:

```
Screen (2D):

  . . . . . . . . .
  . . . P → → → → .    ← a line of pixels going right from P
  . . . . . . . . .
```

Each of these pixels has a **depth value** in the depth buffer. If you plot those depth values along the line, you get a 2D side-view profile of the scene:

```
Depth profile along the line (the "slice"):

  depth
    ↑
    |      ██
    |     ███
    |    ████         ██
    |   █████        ███
    |  ██████  ░░░░ ████
    | ████████░░░░██████
    +────P──────────────→ pixel distance
         ↑         ↑
       a wall    another object
```

That 2D profile is the **slice**. It's like cutting the 3D scene with a vertical knife along the screen direction and looking at the cross-section.

A "screen direction" is simply **which way you march across the pixel grid** — at 0° you go rightward, at 90° upward, at 45° diagonally. Any angle defines a line of pixels to sample along.

**In the GTAO algorithm**, the shader marches along the line, sampling a few depth points (4 steps), and for each one asks: "how high above my surface is this sample?" The highest one determines the **horizon** — how much of the sky is blocked in this direction.

```
The slice with horizon angle:

    ↑
    |      S ← highest sample (horizon)
    |     /
    |    /  ← horizon angle
    |   / )
    +──P─────────→
```

GTAO does this for 4 different directions on the screen, each producing its own slice and its own horizon angle. Averaging them approximates the full 3D hemisphere occlusion — because the hemisphere can be decomposed into vertical planes through its center, each contributing independently to the total occlusion.

### Q: Why does GTAO only sample a semicircle (π) instead of a full circle (2π)?

**A:**

GTAO marches outward from P in **one direction only** per loop iteration (positive steps). So each direction is a half-line, not a full line:

```
Direction 0° (rightward):     P ──→ S1 → S2 → S3 → S4
Direction 180° (leftward):    S4 ← S3 ← S2 ← S1 ──← P
```

These two are opposite half-lines along the same line through P.

**Option A: 4 directions over 2π (full circle)**

```
        90°
         ↑
         |
180° ←── P ──→ 0°
         |
         ↓
        270°
```

Spacing: 90° apart. But 0° and 180° are the same line (opposite sides). Same for 90° and 270°. So you're really only covering **2 unique lines**, sampled from both sides.

**Option B: 4 directions over π (semicircle)**

```
   135°  90°  45°
     \    |    /
      \   |   /
       \  |  /
        \ | /
         P ──→ 0°
```

Spacing: 45° apart. These are **4 unique lines**, each sampled from one side only.

**The trade-off:**

| | Full circle (2π) | Semicircle (π) |
|---|---|---|
| Unique lines | 2 | 4 |
| Angular spacing | 90° | 45° |
| Covers both sides of each line | Yes | No |

The semicircle gives **twice the angular resolution** (45° vs 90°), but misses the opposite side of each line.

**What about the missing opposite side?**

Consider P at the base of a wall:

```
        Wall
         |
    ─────P─────  floor
```

- Direction 0° (toward wall): finds high horizon → strong occlusion
- Direction 180° (away from wall): would find flat floor → no occlusion

If a pixel only samples the semicircle containing 180° and misses 0°, it underestimates occlusion.

**But the noise rotation saves it.** Each pixel gets a different random rotation:

```
Pixel A's semicircle:  [10°, 55°, 100°, 145°]    → includes toward-wall direction ✓
Pixel B's semicircle:  [170°, 215°, 260°, 305°]   → misses the wall ✗
Pixel C's semicircle:  [80°, 125°, 170°, 215°]    → partially catches wall ✓
```

Pixel B gets wrong AO, but the **bilateral blur** averages B with its neighbors A and C, who measured correctly. The result after blur is close to the true value.

**In short:** the semicircle is not correct per-pixel — it misses half the directions. But it buys better angular density (45° instead of 90°), and the per-pixel noise rotation + spatial blur recovers the missing half **statistically** across neighboring pixels.

### Q: Explain the math in the depth-aware bilateral blur in detail.

**A:**

The raw GTAO output is noisy — each pixel uses only 16 samples with random jitter, so neighboring pixels produce slightly different AO values. A blur smooths this noise. But a naive Gaussian blur would smear AO across object edges. A **bilateral** blur prevents this by rejecting samples at a different depth.

**The kernel shape:**

The blur uses a 7-tap kernel: the center pixel plus 3 neighbors on each side along one direction.

```
direction = (1, 0) for horizontal blur:

  [-3]  [-2]  [-1]  [center]  [+1]  [+2]  [+3]
```

The weight array `weights[4] = {0.324, 0.232, 0.0855, 0.0205}` approximates a Gaussian distribution:

| Tap | Distance | Weight |
|---|---|---|
| center | 0 | 1.0 (implicit, used as initial value) |
| ±1 | 1 pixel | 0.232 |
| ±2 | 2 pixels | 0.0855 |
| ±3 | 3 pixels | 0.0205 |

**The bilateral weight computation:**

```glsl
float depthDiff = abs(centerDepth - sampleDepth);
float w = weights[i] * step(depthDiff, depthThreshold);
```

The weight has two factors:

1. **Spatial weight**: `weights[i]` — the Gaussian weight based on distance (always applied).
2. **Range weight**: `step(depthDiff, depthThreshold)` — binary acceptance/rejection.

`step(a, b)` in GLSL returns 1.0 if `b >= a`, 0.0 otherwise. So:

- `depthThreshold >= depthDiff` → **1.0** (depths are similar → ACCEPT, same surface)
- `depthThreshold < depthDiff` → **0.0** (depths differ → REJECT, across an edge)

With `depthThreshold = 0.02`, a depth difference less than 0.02 passes; greater or equal is rejected.

**Normalization:**

```glsl
float result = totalAO / totalWeight;
```

Dividing by `totalWeight` ensures the result is properly normalized even when some samples are rejected. Example at an object edge:

```
              depth:  0.5   0.5   0.5  |  0.9   0.9   0.9
              AO:     0.3   0.3   0.3  |  1.0   1.0   1.0
                      [-2]  [-1]  [C]  |  [+1]  [+2]  [+3]
```

Right-side pixels (depth 0.9) are rejected (w = 0):

```
totalAO     = 0.3 + 0.3×0.232 + 0.3×0.0855 + 0 + 0 + 0 = 0.39525
totalWeight = 1.0 + 0.232     + 0.0855      + 0 + 0 + 0 = 1.3175
result      = 0.39525 / 1.3175 = 0.3   ← correct, no leaking
```

Without normalization, the result would be 0.39525 — incorrectly brightened because half the kernel was zeroed out.

**Why separable (two passes)?**

A true 2D bilateral blur with a 7×7 kernel needs 49 texture reads. Splitting into horizontal then vertical needs only 7 + 7 = **14 reads**. This works because the Gaussian kernel is separable, and the bilateral depth rejection makes it only an approximation — but visually indistinguishable.

**Summary formula:**

```
result = Σ(ao_i × gaussian_weight_i × depth_accept_i) / Σ(gaussian_weight_i × depth_accept_i)
```

Where `depth_accept_i` is 1 if the neighbor is on the same surface, 0 if across an edge.

### Q: For samples with abrupt depth changes, the feature is preserved, right?

**A:**

Yes. When there's an abrupt depth change (a depth discontinuity = an object edge), the blur preserves that sharp boundary by rejecting samples on the other side:

```
Object A          |  Object B
depth 0.5         |  depth 0.9
AO = 0.3          |  AO = 1.0
                   ↑
             depth discontinuity
             (depthDiff = 0.4 > threshold 0.02)
             → samples across this edge are REJECTED
             → each side blurs only with its own neighbors
             → the sharp AO difference is preserved
```

The filter smooths noise **within** a surface but refuses to blend **across** surfaces. The AO edge stays crisp at object silhouettes, which is what you'd expect physically — the shadow contact between two objects should have a sharp boundary, not a blurry smear.
