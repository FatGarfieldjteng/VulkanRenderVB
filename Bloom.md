# Bloom Algorithm

Bloom simulates the light scattering that happens in real camera lenses and the human eye, where very bright areas "bleed" light into their surroundings, creating a soft glow. It's implemented as a **dual-pass mip chain**: a multi-level downsample followed by a multi-level upsample, both running as compute shaders.

## High-Level Flow

```
HDR Image (full resolution)
    |
    v
[Downsample Pass 0]  HDR → Mip 0 (half res)     with Karis average
[Downsample Pass 1]  Mip 0 → Mip 1 (quarter res)
[Downsample Pass 2]  Mip 1 → Mip 2 (1/8 res)
[Downsample Pass 3]  Mip 2 → Mip 3 (1/16 res)
[Downsample Pass 4]  Mip 3 → Mip 4 (1/32 res)
[Downsample Pass 5]  Mip 4 → Mip 5 (1/64 res)
    |
    v
[Upsample Pass 4]   Mip 5 → additive blend into Mip 4
[Upsample Pass 3]   Mip 4 → additive blend into Mip 3
[Upsample Pass 2]   Mip 3 → additive blend into Mip 2
[Upsample Pass 1]   Mip 2 → additive blend into Mip 1
[Upsample Pass 0]   Mip 1 → additive blend into Mip 0
    |
    v
Mip 0 (half res bloom texture, read by tonemap.frag)
```

## GPU Resource: Single Mip-Chain Image

Instead of allocating separate images, bloom uses **one `VkImage` with 6 mip levels**:

```cpp
static constexpr uint32_t MIP_COUNT = 6;

imgInfo.format    = VK_FORMAT_R16G16B16A16_SFLOAT;  // HDR 16-bit float
imgInfo.extent    = { width/2, height/2, 1 };       // mip 0 = half resolution
imgInfo.mipLevels = MIP_COUNT;
imgInfo.usage     = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
```

Each mip level gets its own `VkImageView` for individual read/write access:

| Mip Level | Resolution (for 1920x1080) | Texels  |
|-----------|----------------------------|---------|
| 0         | 960 x 540                  | 518,400 |
| 1         | 480 x 270                  | 129,600 |
| 2         | 240 x 135                  | 32,400  |
| 3         | 120 x 67                   | 8,040   |
| 4         | 60 x 33                    | 1,980   |
| 5         | 30 x 16                    | 480     |

The deeper mips capture progressively **wider** bloom because each texel represents a larger area of the original image. Mip 5 at 30x16 covers the entire screen, so any bright pixel there spreads its glow across the whole image.

## Downsample: 13-Tap Filter with Karis Average

**Shader**: `bloom_downsample.comp`
**Workgroup**: 8x8 threads. Each thread writes one destination texel.

### The 13-tap sample pattern

Each destination texel samples 13 points from the source in a cross-shaped pattern:

```
A . B . C
. D . E .
F . G . H
. I . J .
K . L . M
```

Where G is the center (at `uv`), and offsets are in multiples of `srcTexelSize`:

```glsl
vec3 A = texture(srcTexture, uv + srcTexelSize * vec2(-2, -2)).rgb;
vec3 B = texture(srcTexture, uv + srcTexelSize * vec2( 0, -2)).rgb;
// ... 11 more taps ...
vec3 M = texture(srcTexture, uv + srcTexelSize * vec2( 2,  2)).rgb;
```

Because the sampler uses bilinear filtering (`VK_FILTER_LINEAR`), each `texture()` call actually blends 4 texels. So the 13 taps effectively cover a large neighborhood with a smooth falloff, much wider than a simple box filter or even a 5x5 Gaussian.

This pattern is from the **Call of Duty: Advanced Warfare** presentation (Jorge Jimenez, SIGGRAPH 2014). It was chosen because it provides high-quality downsampling with minimal aliasing artifacts while being efficient on GPU hardware.

### Mip 0: Karis Average (firefly suppression)

The first downsample (HDR → Mip 0) is special. The 13 samples are grouped into 5 overlapping 2x2 quads:

```glsl
vec3 g0 = (D + E + I + J) * 0.25;    // center quad
vec3 g1 = (A + B + D + G) * 0.25;    // top-left quad
vec3 g2 = (B + C + E + H) * 0.25;    // top-right quad
vec3 g3 = (F + G + I + K) * 0.25;    // bottom-left quad
vec3 g4 = (G + H + J + M) * 0.25;    // bottom-right quad
```

Each quad is then **weighted by the Karis average**:

```glsl
float KarisAverage(vec3 c) {
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return 1.0 / (1.0 + lum);
}
```

This weight is `1 / (1 + luminance)` — brighter quads get **lower weight**. The final result is:

```glsl
result = (g0*w0 + g1*w1 + g2*w2 + g3*w3 + g4*w4) / (w0+w1+w2+w3+w4);
```

**Why?** In HDR scenes, a single extremely bright pixel (e.g., a specular highlight with luminance 10000) can dominate a simple average, producing a pulsating "firefly" artifact. The Karis average suppresses this by reducing the influence of very bright regions in the first downsample step. Once the image is at half resolution, the extreme peaks are already smoothed out, so subsequent mips use a standard weighted filter.

### Mips 1-5: Standard weighted filter

For subsequent mip levels, a fixed-weight combination is used:

```glsl
result  = G * 0.125;                          // center: 12.5%
result += (D + E + I + J) * 0.03125 * 4.0;   // inner diamond: 0.03125 × 4.0 = 0.125 per sample
result += (B + F + H + L) * 0.0625;          // cross: 6.25% per sample
result += (A + C + K + M) * 0.03125;         // corners: 3.125% per sample
```

Note: `0.03125 * 4.0 = 0.125` — the `* 4.0` is a scaling factor on the weight, not the
sample count. Each of D, E, I, J gets weight 0.125 (same as center G).

Per-sample weights and group totals:

| Samples              | Per-sample weight | Count | Group total |
|----------------------|-------------------|-------|-------------|
| G (center)           | 0.125             | 1     | 0.125       |
| D, E, I, J (inner)   | 0.125             | 4     | 0.500       |
| B, F, H, L (cross)   | 0.0625            | 4     | 0.250       |
| A, C, K, M (corners) | 0.03125           | 4     | 0.125       |
| **Total**            |                   | **13** | **1.000**  |

The weights sum to exactly 1.0, making the filter energy-preserving: average brightness
is neither gained nor lost during downsampling. The weight distribution falls off from
center to corner, approximating a Gaussian bell curve:

```
0.03125  .  0.0625  .  0.03125
   .    0.125  .  0.125    .
0.0625   .  0.125   .  0.0625
   .    0.125  .  0.125    .
0.03125  .  0.0625  .  0.03125
```

## Upsample: 9-Tap Tent Filter with Additive Blend

**Shader**: `bloom_upsample.comp`
**Workgroup**: 8x8 threads. Each thread writes one destination texel.

The upsample runs **backwards** through the mip chain: from mip 5 up to mip 0.

### The 9-tap tent filter

Each destination texel samples 9 points from the smaller (source) mip:

```
a  b  c
d  e  f
g  h  i
```

With offsets scaled by `filterRadius` (default 1.0) and `srcTexelSize`:

```glsl
vec3 a = texture(srcTexture, uv + vec2(-r, -r) * srcTexelSize).rgb;
vec3 b = texture(srcTexture, uv + vec2( 0, -r) * srcTexelSize).rgb;
// ... 7 more taps ...
```

The weights form a **tent** (pyramid) distribution:

```glsl
vec3 upsample = e * 0.25                          // center: 25%
              + (b + d + f + h) * 0.125            // edges:  4 × 12.5% = 50%
              + (a + c + g + i) * 0.0625;          // corners: 4 × 6.25% = 25%
```

Total = 0.25 + 0.50 + 0.25 = 1.0. This is energy-preserving.

The tent filter produces a smoother result than bilinear upsampling alone, reducing the blocky artifacts you'd see from just scaling up a low-resolution image.

### Additive blend (the key to bloom)

The upsample result is **added** to the existing content at the destination mip:

```glsl
vec3 existing = imageLoad(dstTexture, coord).rgb;
imageStore(dstTexture, coord, vec4(existing + upsample, 1.0));
```

This is crucial. At each upsample step, the sharp detail from the downsample is preserved (`existing`), and the blurry glow from deeper mips is layered on top (`upsample`). The result is a natural multi-scale bloom where:

- **Mip 5 → Mip 4**: Adds very wide, soft glow (covers whole screen)
- **Mip 4 → Mip 3**: Adds medium-wide glow on top
- **Mip 3 → Mip 2**: Adds medium glow
- **Mip 2 → Mip 1**: Adds fine glow
- **Mip 1 → Mip 0**: Adds the finest glow detail

By the time mip 0 is finished, it contains the **sum of all scales** — a natural-looking bloom with both tight halos around small highlights and broad glows around large bright areas.

## How the Bloom Result Is Consumed

In `tonemap.frag`, the bloom texture (mip 0) is sampled and added to the HDR color **before** tone mapping:

```glsl
vec3 bloom = texture(bloomTex, fragUV).rgb;
hdr += bloom * bloomStrength;
```

`bloomStrength` is a UI-controlled multiplier (default ~0.04). It's applied **before** exposure and tone mapping, so the bloom is naturally compressed by the tone curve along with everything else — bright bloom gets compressed more than dim bloom, preserving the natural feel.

## Synchronization (Barriers)

Between each downsample step, a barrier ensures the writes to mip `i` are visible before mip `i` is read as the source for mip `i+1`:

```cpp
TransitionImage(cmd, mBloomImage,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
    VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1);   // only mip i
```

The barrier is per-mip (not whole image) for maximum overlap — the GPU can start reading mip `i` for the next pass while other mips are still being processed.

Similarly, upsample passes have barriers between each step. After the final upsample, mip 0 is transitioned from `GENERAL` to `SHADER_READ_ONLY_OPTIMAL` for efficient reading by the tone mapping fragment shader.

## UI Parameters

- **Bloom Strength** (default ~0.04): Multiplier for the bloom contribution in the tone mapping shader. 0 = no bloom, higher = more glow.
- **Filter Radius** (default 1.0): Controls the spread of the upsample tent filter. Larger values produce wider bloom per mip level.

## Summary: Why This Approach?

This dual-filter mip-chain method (often called "dual Kawase" or "progressive downscale/upscale bloom") is the industry standard because:

1. **Multi-scale**: Different mip levels naturally capture different bloom radii without needing multiple separate blur passes at full resolution.
2. **Efficient**: Only 11 compute dispatches total (6 down + 5 up), each at progressively smaller resolutions. The total work is roughly equivalent to 2x the full-resolution cost.
3. **Energy-preserving**: The additive upsample ensures no light is lost — the total energy of the bloom matches the bright areas of the input.
4. **Firefly-free**: The Karis average on the first downsample prevents single extreme-brightness pixels from dominating.
5. **Smooth**: The 13-tap downsample and 9-tap tent upsample produce artifact-free results with no visible banding or aliasing.

---

## Q&A
**Downsample then Upsample pipeline**
### Q: How is sharp detail preserved during the upsample? What does "the blurry glow from deeper mips is layered on top" mean?

### A:

The key is to trace what happens to each mip level's content through both phases.

#### After the downsample phase

Each mip stores a filtered version of the level above it:

```
Mip 0: filtered half-res of HDR    (relatively sharp — fine detail of bright areas)
Mip 1: filtered quarter-res        (blurrier — medium features)
Mip 2: filtered 1/8 res            (blurrier still)
Mip 3: filtered 1/16 res           (coarse features only)
Mip 4: filtered 1/32 res           (very coarse)
Mip 5: filtered 1/64 res           (essentially a tiny smear of the whole screen)
```

Each mip captures detail at a **different spatial scale**. Mip 0 still has relatively fine
glow patterns (tight halos around small highlights). Mip 5 has lost all fine detail — it's
just a broad color average.

#### The upsample phase — additive, not replacing

Here's the critical shader code:

```glsl
vec3 existing = imageLoad(dstTexture, coord).rgb;   // what downsample wrote here
imageStore(dstTexture, coord, vec4(existing + upsample, 1.0));  // ADD, don't replace
```

The `existing` value is what was written to this mip during the downsample phase. The
`upsample` value is the interpolated content from the deeper (blurrier) mip. They are
**summed**, not replaced.

#### Tracing through the upsample chain

Let's call the downsample content at each mip `D0, D1, D2, D3, D4, D5`:

| Upsample step | Source | Destination before | Destination after |
|---|---|---|---|
| Mip 5 → Mip 4 | D5 | D4 | D4 + upsample(D5) |
| Mip 4 → Mip 3 | D4 + upsample(D5) | D3 | D3 + upsample(D4 + upsample(D5)) |
| Mip 3 → Mip 2 | D3 + ... | D2 | D2 + upsample(D3 + ...) |
| Mip 2 → Mip 1 | D2 + ... | D1 | D1 + upsample(D2 + ...) |
| Mip 1 → Mip 0 | D1 + ... | D0 | D0 + upsample(D1 + ...) |

The final mip 0 contains:

```
D0 + upsample(D1 + upsample(D2 + upsample(D3 + upsample(D4 + upsample(D5)))))
```

Every `D_i` contributes. **No level is thrown away.** Each adds its own spatial scale of bloom:

- `D0` → fine glow (tight halos, ~2-pixel radius)
- `D1` → slightly wider glow (~4-pixel radius)
- `D2` → medium glow (~8-pixel radius)
- `D3` → broad glow (~16-pixel radius)
- `D4` → very broad glow (~32-pixel radius)
- `D5` → screen-wide glow (~64-pixel radius)

If instead the shader had done `imageStore(dstTexture, coord, vec4(upsample, 1.0))`
(replace, no add), then `D0` through `D4` would all be lost, and you'd only see the
blurriest level — a shapeless fog with no fine halo structure.

#### Visual analogy

Think of stacking transparent sheets, each with a glow at a different blur radius:

```
Sheet 5: ░░░░░░░░░░░░░░░░░░░░  (very wide, very faint)
Sheet 4:   ░░░░░░░░░░░░░░░░    (wide, faint)
Sheet 3:     ░░░░░░░░░░░░      (medium)
Sheet 2:       ░░░░░░░░        (narrower)
Sheet 1:         ░░░░░░        (narrow)
Sheet 0:          ░░░░         (tight halo)
─────────────────────────────
Combined:  ░░░▒▒▓▓██▓▓▒▒░░░   (natural falloff from bright center)
```

The additive blend stacks them all together. The result is a smooth, natural bloom that is
bright near the source and gradually fades — just like real lens flare.


**Tent filter**

### Q: What is a tent (pyramid) distribution and what is its purpose in the bloom upsample?

### A:

#### What it is

A "tent" is a filter kernel shaped like a tent or pyramid. In 1D, the weight profile looks
like this:

```
Weight
  ^
  |      *
  |     / \
  |    /   \
  |   /     \
  |  /       \
  | /         \
  +--+--+--+--+--> Position
    -2 -1  0 +1 +2
```

The weight is highest at the center and drops **linearly** to zero at the edges. Compare
this to:

- **Box filter** (flat): all positions get equal weight → produces blocky artifacts
- **Tent filter** (linear falloff): smooth triangle shape → much smoother
- **Gaussian filter** (exponential falloff): bell curve → smoothest, but more expensive

#### In 2D: the pyramid

A 2D tent is the **product of two 1D tents** (one horizontal, one vertical). For the 3x3
case:

1D tent weights: `[1, 2, 1]` (corner, edge, center)

2D = outer product:

```
        1   2   1
    ┌──────────────┐
1   │  1×1  1×2  1×1  │     1  2  1
2   │  2×1  2×2  2×1  │  =  2  4  2
1   │  1×1  1×2  1×1  │     1  2  1
    └──────────────┘
```

Normalize by dividing by the sum (16):

```
1/16   2/16   1/16       0.0625   0.125   0.0625
2/16   4/16   2/16   =   0.125    0.25    0.125
1/16   2/16   1/16       0.0625   0.125   0.0625
```

This is exactly the upsample shader's weights:

```glsl
vec3 upsample = e * 0.25                          // center: 4/16
              + (b + d + f + h) * 0.125            // edges:  2/16 each
              + (a + c + g + i) * 0.0625;          // corners: 1/16 each
```

If you view this from the side, it looks like a pyramid — hence the name.

#### Why the upsample needs a blur filter

When upsampling from a small mip to a larger one, each source pixel must fill multiple
destination pixels. You must interpolate somehow:

| Method | Kernel size | Smoothness | Bloom quality |
|---|---|---|---|
| Nearest neighbor | 1x1 | Blocky | Unusable — visible pixel grid |
| Bilinear only | 2x2 | OK | Minor diamond artifacts |
| Tent filter | 3x3 (× bilinear = 6x6 effective) | Smooth | Artifact-free, natural glow |
| Gaussian | 5x5+ | Smoothest | Overkill — tent is sufficient |

Smoothness matters because bloom simulates real lens light scattering, which is
**perfectly smooth** — no hard edges or grid patterns. Any interpolation artifacts from
upsampling would be directly visible as unnatural structure in the glow halo.

#### Secondary benefit: slight spread

Because the tent samples a 3x3 neighborhood (wider than the minimum 2x2 needed for
bilinear), it slightly **spreads** the glow at each upsample step. This contributes
additional blur on top of what the mip chain already provides, making the final bloom halo
even smoother and wider. The `filterRadius` parameter (default 1.0) controls this spread —
increasing it widens the tent, producing a softer, broader bloom.

**Downsample process**
### Q: The downsample is just blurring the image step by step. How does it keep brighter values and suppress non-bright pixels?

### A:

The downsample does **NOT** explicitly separate bright from non-bright pixels. It blurs
**everything** — there is no brightness threshold in the shader. The downsample shader
samples all pixels unconditionally and writes the filtered result. The tone mapping shader
also adds the bloom unconditionally:

```glsl
vec3 bloom = texture(bloomTex, fragUV).rgb;
hdr += bloom * bloomStrength;
```

No threshold anywhere. So how does bloom appear to only affect bright areas?

#### HDR math does the work automatically

In an HDR scene, pixel values span a huge range:

| Scene element      | Typical luminance |
|--------------------|-------------------|
| Deep shadow        | 0.001             |
| Dark wall          | 0.05              |
| Lit surface        | 0.5               |
| Bright white       | 1.0               |
| Specular highlight | 50                |
| Light bulb         | 500               |
| Sun                | 100,000           |

When the downsample blurs a bright pixel with its neighbors, the bright pixel **dominates**
because it is orders of magnitude larger. For example, a lamp pixel (500) surrounded by
dark wall pixels (0.05):

**Before blur** (simplified 2x2):
```
 0.05   0.05
 0.05   500
```

**After averaging**: `(0.05 + 0.05 + 0.05 + 500) / 4 ≈ 125`

The lamp is 10,000x brighter than the wall, so it dominates the result. Through multiple
downsample levels, this bright value spreads outward, creating a glow pattern naturally
centered on the lamp.

Now consider 4 ordinary wall pixels:
```
 0.05   0.06
 0.04   0.05
```

**After averaging**: `≈ 0.05` — no interesting spread. It stays dim.

#### The `bloomStrength` multiplier makes dim bloom invisible

The bloom is added with a small multiplier (default ~0.04):

- **Bloom from the lamp**: bloom texture ≈ 50 → `50 × 0.04 = +2.0` — very visible addition
- **Bloom from the wall**: bloom texture ≈ 0.05 → `0.05 × 0.04 = +0.002` — invisible (0.2% of a dim pixel)

The bloom from dim areas is technically present but so tiny relative to existing pixel
values that it is completely imperceptible.

#### Why no explicit threshold?

Some engines use an explicit brightness threshold (e.g., "only bloom pixels with
luminance > 1.0"). This engine does not. The threshold-free approach has advantages:

1. **No hard cutoff artifacts**: A brightness threshold creates a sharp boundary — pixels
   at luminance 0.99 get zero bloom, pixels at 1.01 get full bloom. This can produce
   visible "popping" when objects cross the threshold.
2. **Simpler pipeline**: No extra pass needed to extract bright pixels.
3. **HDR does the work**: The dynamic range is so large that bright areas naturally
   dominate the blur, and `bloomStrength` controls visibility.

#### Summary

```
Downsample blurs everything equally
    ↓
But HDR values mean bright pixels are 1000-100000x stronger
    ↓
So after blurring, mip chain content is dominated by bright areas
    ↓
bloomStrength × 0.04 makes the dim contribution negligible
    ↓
Result: only bright areas produce visible glow
```

The selectivity is emergent, not explicit.

**Upsample process**
### Q: The upsampling uses a blur filter — why?

### A:

When upsampling from a small mip to a larger one, each source pixel must fill multiple
destination pixels. You cannot just leave gaps — you must interpolate. The question is
which interpolation method to use.

#### Nearest neighbor (no filtering)

Just copy the nearest source pixel:

```
Source (3 pixels):    Upsampled (6 pixels):

 10   50   10    →    10  10  50  50  10  10
```

Result: visible hard edges where source pixel boundaries are. The bloom halo would look
like a staircase of colored blocks.

#### Bilinear (hardware texture sampler)

The GPU's `texture()` with `VK_FILTER_LINEAR` blends the 4 nearest source texels weighted
by distance. This uses a 2x2 kernel:

```
Source (3 pixels):    Bilinear upsampled (6 pixels):

 10   50   10    →    10  22  42  42  22  10
```

Better, but the 2x2 kernel is small and can still show subtle "diamond" artifacts in 2D.

#### Tent filter (what the shader does)

Sample a 3x3 neighborhood with pyramid-shaped weights, on top of bilinear hardware
sampling. Each `texture()` call already blends 4 texels internally, so 9 taps effectively
cover a 6x6 source region:

```
Source (3 pixels):    Tent-filtered upsampled (6 pixels):

 10   50   10    →    10  18  36  36  18  10
```

The widest kernel produces the **smoothest** transition with no visible block edges or
diamond patterns.

#### Why smoothness matters here

Bloom simulates real lens light scattering, which is **perfectly smooth** — no hard edges
or grid patterns. Any interpolation artifacts from upsampling would be directly visible as
unnatural structure in the glow halo. The tent filter is the cheapest way to get
artifact-free upsampling:

| Method | Kernel size | Smoothness | Bloom quality |
|---|---|---|---|
| Nearest neighbor | 1x1 | Blocky | Unusable — visible pixel grid |
| Bilinear only | 2x2 | OK | Minor diamond artifacts |
| Tent filter | 3x3 (× bilinear = 6x6 effective) | Smooth | Artifact-free, natural glow |
| Gaussian | 5x5+ | Smoothest | Overkill — tent is sufficient |

The tent filter hits the sweet spot: 9 texture fetches per pixel for perfectly smooth
upsampling. Any less and you see artifacts; any more is wasted work.


