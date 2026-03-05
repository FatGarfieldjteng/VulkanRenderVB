# Hi-Z Occlusion Culling

## Core Idea

Skip drawing objects that are completely hidden behind other objects. Instead of doing expensive per-pixel visibility tests, build a **hierarchical depth buffer (Hi-Z)** — a mip chain where each texel stores the **maximum** depth in its footprint. This lets you cheaply test an object's bounding box against the depth buffer at the appropriate resolution level with a single texture sample.

## The Full Pipeline (5 Stages)

### Stage 1: Frustum Cull (`cull.comp`)

**Input:** All draw commands + object data + frustum planes

For each mesh, test its world-space AABB against the 6 frustum planes. Discard anything outside the frustum. The survivors are split into two buckets based on index:

- **Occluders** (index `< occluderCount`): the first 20% of meshes. Written to `occluderCmds[]`.
- **Candidates** (index `>= occluderCount`): the remaining 80%. Written to `candidateCmds[]`.

**Output:** Two indirect draw buffers + their counts.

### Stage 2: Occluder Depth Pass (`OccluderDepthPass`)

**Input:** `occluderCmds[]` from Stage 1

Render only the occluders into a depth-only framebuffer using `vkCmdDrawIndexedIndirectCount`. No color output — just depth. This produces a conservative depth buffer: it won't contain every object, but the objects it does contain are guaranteed to be visible (they passed frustum culling, and they're drawn with standard depth testing).

**Output:** A depth buffer containing the occluders' depth.

### Stage 3: Hi-Z Build (`HiZBuffer::BuildMipChain`)

**Input:** The occluder depth buffer from Stage 2

Build a mip chain from the depth buffer. Each mip texel stores the **maximum** depth of the 4 texels it covers from the previous mip level (`hiz_reduce.comp`):

```glsl
vec4 depths = textureGather(srcDepth, uv, 0);
float maxDepth = max(max(depths.x, depths.y), max(depths.z, depths.w));
```

Why maximum? Larger depth = farther away. The max ensures that if *any* part of a region has a far depth, the conservative test will use that far value. This prevents false positives (incorrectly culling a visible object) — an object is only culled if it's behind the *farthest* depth in its screen-space footprint.

The build is sequential — mip 0 reads the source depth, mip 1 reads mip 0, mip 2 reads mip 1, etc. A barrier between each mip ensures the previous mip is fully written before the next reads it. The sampler also uses `VK_SAMPLER_REDUCTION_MODE_MAX` so even hardware filtering takes the max rather than average.

**Output:** A full mip chain Hi-Z image (R32F, `GENERAL` layout).

### Stage 4: Occlusion Test (`cull_occlusion.comp`)

**Input:** `candidateCmds[]` from Stage 1 + Hi-Z image from Stage 3

For each candidate, the shader:

1. **Computes the world-space AABB** from the object's model matrix.

2. **Projects all 8 AABB corners to NDC**. If any corner has `clip.w <= 0` (behind the camera), it conservatively marks the object as visible (skip culling).

3. **Finds the screen-space bounding rectangle** (`ndcMin`, `ndcMax`) and the **closest depth** (`closestZ`) among all 8 corners.

4. **Picks the right mip level** (see detailed explanation below).

5. **Samples the Hi-Z** at the center of the bounding rectangle at that mip level. This returns the **maximum depth** (farthest occluder) in that region.

6. **Compares**: if the candidate's closest depth > sampled depth, the candidate is entirely behind the occluders in that screen region — **cull it**. Otherwise, it might be visible — keep it.

Survivors are written to the `visibleCmds[]` buffer.

**Output:** A visible indirect draw buffer + count.

### Stage 5: Forward Pass

Draws both sets:
- **Occluders** (from Stage 1) — always drawn, they're known visible.
- **Visible candidates** (from Stage 4) — survived both frustum and occlusion culling.

## Why It's Conservative (No False Culling)

Every step is designed to never wrongly hide a visible object:

- Hi-Z stores **max depth** (farthest) per region, so an object is only culled if it's behind the farthest point in its footprint.
- The AABB is a bounding volume, always larger than the actual mesh — so the screen-space footprint overestimates.
- `closestZ` uses the nearest corner of the bounding box — if even the nearest point is behind the Hi-Z, the whole object is hidden.
- Behind-camera corners (`clip.w <= 0`) skip culling entirely.

What clip.w means
When you multiply a world-space position by the view-projection matrix, you get a clip-space coordinate (x, y, z, w). The w component represents the depth from the camera — positive means in front of the camera, negative or zero means behind (or exactly at) the camera.
The perspective divide ndc = clip.xyz / clip.w converts clip space to NDC (Normalized Device Coordinates, the -1 to +1 range). This division is only valid when w > 0.
What goes wrong if clip.w <= 0
If a corner is behind the camera (w < 0), the perspective divide produces nonsense:
The x,y coordinates get flipped (dividing by a negative mirrors them)
The z value becomes meaningless
The resulting NDC bounding rectangle would be completely wrong
For example, a corner at clip (10, 5, 3, -2) would produce NDC (-5, -2.5, -1.5) — it appears to be on the opposite side of the screen from where it actually is.
Why "skip culling" is the safe choice
If even one AABB corner is behind the camera, the object straddles the near plane — part of it is in front of the camera, part behind. In this case:
You can't compute a valid screen-space bounding rectangle (the math breaks)
The object is almost certainly very close to the camera and likely visible
Culling it incorrectly would cause a visible pop-in artifact
So return false ("not occluded") means: don't cull this object, just draw it. This is conservative — you might draw something that's actually hidden, but you'll never accidentally hide something that should be visible. A bit of wasted shading is always preferable to a visual glitch.

The trade-off is occasional **false visibility** (drawing something that's actually hidden), which is fine — it just means a bit of wasted shading, not visual artifacts.

---

## Mip Level Selection — Detailed Explanation

The relevant code in `cull_occlusion.comp`:

```glsl
vec2 sizePixels = (uvMax - uvMin) * params.hiZSize;
float mipLevel = ceil(log2(max(sizePixels.x, sizePixels.y)));
mipLevel = clamp(mipLevel, 0.0, float(textureQueryLevels(hiZMap) - 1));
```

### What the Hi-Z Mip Chain Looks Like

Say the Hi-Z image is 512×512 (mip 0). Each mip level halves the resolution:

| Mip Level | Resolution | Each texel covers (in mip 0 pixels) |
|-----------|-----------|--------------------------------------|
| 0 | 512×512 | 1×1 |
| 1 | 256×256 | 2×2 |
| 2 | 128×128 | 4×4 |
| 3 | 64×64 | 8×8 |
| 4 | 32×32 | 16×16 |
| 5 | 16×16 | 32×32 |
| 6 | 8×8 | 64×64 |
| 7 | 4×4 | 128×128 |
| 8 | 2×2 | 256×256 |
| 9 | 1×1 | 512×512 |

A single texel at mip N covers a 2^N × 2^N region of the original image.

### The Goal

Test the candidate's bounding rectangle against the Hi-Z with **one texture sample**. That sample must cover the entire bounding rectangle. So find a mip level where one texel is large enough to contain the whole rectangle.

### Line by Line

**`vec2 sizePixels = (uvMax - uvMin) * params.hiZSize;`**

`uvMax - uvMin` is the bounding rectangle size in UV space (0 to 1). Multiplying by `params.hiZSize` (e.g., `vec2(512, 512)`) converts it to pixels at mip 0.

Example: if the bounding rectangle spans UV 0.1 to 0.2 horizontally, that's `0.1 × 512 = 51.2` pixels wide.

**`float mipLevel = ceil(log2(max(sizePixels.x, sizePixels.y)));`**

Take the larger dimension (say 51.2 pixels). Find a mip level where one texel covers at least 51.2 pixels. Since one texel at mip N covers 2^N pixels:

```
2^N >= 51.2
N >= log2(51.2)
N >= 5.68
N = ceil(5.68) = 6
```

At mip 6, one texel covers 64×64 pixels, which is enough to contain the 51.2-pixel footprint. `ceil` rounds up to guarantee the texel is large enough — mip 5 (32×32) would be too small.

**`mipLevel = clamp(mipLevel, 0.0, float(textureQueryLevels(hiZMap) - 1));`**

Safety clamping — don't go below mip 0 or above the last mip level.

### Why This Works

After picking mip 6, the shader samples at the center of the bounding rectangle:

```glsl
float sampledDepth = textureLod(hiZMap, (uvMin + uvMax) * 0.5, mipLevel).r;
```

At mip 6 (8×8 resolution), that center UV maps to one texel that represents a 64×64 pixel area. Since Hi-Z stores the **max depth** per texel (built with reduction mode MAX), this single sample returns the farthest depth across that entire 64×64 region — which fully covers the candidate's 51.2-pixel footprint.

If the candidate's closest depth is still behind this value, it's guaranteed to be hidden.

### Visual Intuition

Think of it like zooming out on a map. At full resolution you'd need to check many pixels. By going up mip levels, you "zoom out" until the candidate's bounding rectangle fits inside a single pixel. That one pixel already holds the worst-case (farthest) depth for the whole area thanks to the max reduction, so one comparison is enough.


How to set hiZSize? doe this 
cull_occlusion.comp
 get called mip-level of times? each with its onw hiZSize?

No, cull_occlusion.comp is dispatched once, not per mip level. And hiZSize is a single fixed value — the resolution of mip 0 of the Hi-Z image.
It's set on the CPU side:
Application.cpp
Lines 942-942
        cullParams.hiZSize        = glm::vec2(float(mHiZBuffer.GetWidth()), float(mHiZBuffer.GetHeight()));
This is the base resolution of the Hi-Z texture (e.g., 512x512). It's passed to the shader as part of CullParams and stays the same for every invocation within the dispatch.
The occlusion test shader runs once per candidate mesh (one thread per candidate). Each thread independently:
Computes its own object's screen-space bounding rectangle in UV space
Uses the same params.hiZSize to convert that UV rectangle to pixel size at mip 0
Calculates which mip level to sample based on that pixel size
Samples the Hi-Z at that mip level using textureLod(hiZMap, uv, mipLevel)
So different threads may sample different mip levels of the same Hi-Z texture — a small object on screen might sample mip 2, while a large object might sample mip 6. But they all use the same hiZSize to figure out which mip to pick. The hiZSize is just the conversion factor from UV space to pixel space.
This is different from BuildMipChain, which is dispatched once per mip level to construct the Hi-Z. The occlusion test just reads the already-built Hi-Z.

---

## What If the Sample Point Lands on a Texel Boundary?

When sampling the Hi-Z with:

```glsl
float sampledDepth = textureLod(hiZMap, (uvMin + uvMax) * 0.5, mipLevel).r;
```

The center UV might land on the edge between texels at the chosen mip level. If this were a simple point sample, the bounding rectangle could straddle two adjacent texels and you'd only get the max depth from one of them — potentially causing incorrect culling.

This is handled by the Hi-Z sampler configuration:

```cpp
VkSamplerReductionModeCreateInfo reductionInfo{};
reductionInfo.reductionMode = VK_SAMPLER_REDUCTION_MODE_MAX;

VkSamplerCreateInfo samplerInfo{};
samplerInfo.magFilter = VK_FILTER_LINEAR;
samplerInfo.minFilter = VK_FILTER_LINEAR;
samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
```

Two key settings work together:

- **`VK_FILTER_LINEAR`** — When the sample point falls between texels, the hardware does bilinear filtering, which reads a **2×2 neighborhood** of texels.
- **`VK_SAMPLER_REDUCTION_MODE_MAX`** — Instead of the normal behavior (weighted average of those 4 texels), the hardware returns the **maximum** of those 4 texels.

So if the center UV lands on a texel boundary at the chosen mip level, the hardware automatically returns the max depth from the 2×2 texel neighborhood. This effectively doubles the coverage area for free.

Combined with `ceil` rounding up the mip level (one texel already covers the bounding rect), the 2×2 max gives a comfortable safety margin. Even if the bounding rect straddles a texel boundary, the max of the neighboring texels covers it.

Note: `VK_SAMPLER_MIPMAP_MODE_NEAREST` snaps to the exact mip level computed by the shader, so no inter-mip blending occurs.

This is still not 100% foolproof in extreme edge cases. Some implementations go further by sampling 4 points (corners of the UV rect), adding +1 to the mip level, or using `textureGather` for explicit 2×2 sampling. But for most practical scenarios, `LINEAR` + `REDUCTION_MODE_MAX` is an elegant hardware-accelerated solution that handles the boundary problem with zero extra shader cost.