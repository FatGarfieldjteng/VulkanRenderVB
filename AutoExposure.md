# Auto Exposure Algorithm

Auto exposure simulates the human eye (or camera) adapting to overall scene brightness. It works entirely on the GPU as two sequential compute dispatches per frame, producing a single `float` exposure multiplier that the tone mapping shader reads.

## High-Level Flow

```
HDR Image
    |
    v
[Pass 1: Histogram]  16x16 workgroups, many groups
    |  Builds a 256-bin log-luminance histogram in a GPU SSBO
    v
[Pass 2: Average]    256 threads, 1 group
    |  Reads histogram, computes weighted avg luminance,
    |  derives target exposure, smooths over time
    v
Exposure SSBO (single float, read by tonemap.frag)
```

## GPU Buffers

Two SSBOs persist across frames:

- **HistogramBuffer** (256 x `uint32_t` = 1024 bytes):
  Stores the per-bin pixel count. Cleared to zero every frame
  before the histogram pass.
- **ExposureBuffer** (1 x `float` = 4 bytes):
  Stores the current smoothed exposure value.
  Initialized to 1.0 on the first frame.
  Persists between frames for temporal smoothing.

## Pass 1: Histogram (`exposure_histogram.comp`)

**Workgroup size**: 16x16 = 256 threads.
**Dispatch**: `ceil(width/16) x ceil(height/16)` groups.

Each thread processes one pixel of the HDR image.

### Step 1 - Shared memory init

```glsl
shared uint sharedBins[256];

if (localIdx < 256)
    sharedBins[localIdx] = 0;
barrier();
```

Each workgroup has a local 256-bin histogram in shared memory,
zeroed by the first 256 threads.

`sharedBins` is declared with the `shared` qualifier, meaning it lives in the GPU's
**on-chip shared memory** (also called local data share / LDS on AMD, or shared memory
on NVIDIA). It is only visible to threads within the **same workgroup**, exists only
for the duration of that workgroup's execution, and is roughly 10-100x faster than
global memory.

In contrast, `bins` is a **storage buffer** backed by device-visible VRAM (the
`mHistogramBuffer` allocated with VMA on the CPU side). It is visible to **all
workgroups** across the entire dispatch and persists until the CPU destroys the buffer.

The shader uses a classic **local-then-global reduction** pattern: per-workgroup
shared memory atomics (fast, no cross-workgroup contention), then one global atomic
per bin per workgroup. This massively reduces global memory contention compared to
having every thread do a global atomic.

The guard `if (localIdx < 256)` is defensive — with 16x16 = 256 threads it is always
true, but protects against future workgroup size changes.

### Step 2 - Compute luminance and bin index

```glsl
vec3 color = texture(hdrColor, uv).rgb;
float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
```

Luminance uses the standard BT.709 coefficients (same as sRGB).

The luminance is then mapped to a log-space bin:

```glsl
if (lum < 1e-5)
    bin = 0;           // near-black pixels go to bin 0
else {
    float logLum = clamp(
        (log2(lum) - minLogLum) * invLogLumRange,
        0.0, 1.0
    );
    bin = uint(logLum * 254.0 + 1.0);   // bins 1..255
}
```

The mapping works as follows:

- `minLogLum` and `invLogLumRange` are derived from the UI
  parameters `exposureMinEV` (default -10) and
  `exposureMaxEV` (default 20).
  On the CPU:
  `logLumRange = maxLogLum - minLogLum` (= 30),
  `invLogLumRange = 1.0 / logLumRange`.
- `log2(lum)` maps the luminance to EV (exposure value) space.
  Human perception of brightness is roughly logarithmic —
  doubling the light (1 EV stop) feels like the same perceptual
  step whether you go from 1→2 or from 1000→2000. So log2 is the
  natural space for binning luminance.
- Subtracting `minLogLum` and multiplying by `invLogLumRange`
  normalizes to [0, 1].
- Multiplying by 254 and adding 1 maps to bins 1-255.
- **Bin 0 is reserved for near-black pixels** (< 1e-5).
  These are excluded from the average calculation to avoid
  pitch-black areas dragging the exposure up.

| Luminance  | log2(luminance) |
|------------|-----------------|
| 0.001      | -10             |
| 0.0625     | -4              |
| 1.0        | 0               |
| 16.0       | 4               |
| 1048576.0  | 20              |

The entire HDR range is compressed from [0.001 … 1048576] to [-10 … 20] — a linear
30-unit range, then discretized into 255 bins.

The bounds check `if (coord.x < inputSize.x && coord.y < inputSize.y)` prevents
out-of-bounds threads from sampling outside the image. Because the dispatch rounds up
to multiples of 16, the last row/column of workgroups may extend beyond the actual
image dimensions (e.g., 1080 is not a multiple of 16, so 68 groups cover 1088 rows).

### Step 3 - Accumulate to shared, then global

```glsl
atomicAdd(sharedBins[bin], 1);
barrier();

if (localIdx < 256)
    atomicAdd(bins[localIdx], sharedBins[localIdx]);
```

Two-level reduction: per-workgroup shared memory atomics
(fast, on-chip), then one global atomic per bin per workgroup.
This reduces global memory contention significantly compared
to having every thread do a global atomic.

## Pass 2: Average (`exposure_average.comp`)

**Workgroup size**: 256 threads.
**Dispatch**: 1 group.

### Step 1 - Load histogram into shared memory and reset

```glsl
sharedBins[idx] = bins[idx];
bins[idx] = 0;
barrier();
```

Each of the 256 threads loads one bin into shared memory
and simultaneously clears the global buffer for the next frame.

### Step 2 - Weighted average (thread 0 only)

```glsl
if (idx == 0) {
    float weightedSum = 0.0;
    float totalWeight = 0.0;
    for (uint i = 1; i < 256; i++) {      // skip bin 0
        float logLum = (float(i) - 1.0) / 254.0
                       * logLumRange + minLogLum;
        float w = float(sharedBins[i]);
        weightedSum += logLum * w;
        totalWeight += w;
    }
```

This iterates bins 1 through 255. For each bin:

- **`logLum`**: Reconstructs the log-luminance at that bin's center.
  This is the **inverse** of the histogram mapping:

  Forward: `bin = (log2(lum) - minLogLum) / logLumRange * 254 + 1`
  Reverse: `logLum = (bin - 1) / 254 * logLumRange + minLogLum`

  Substituting the forward into the reverse recovers `log2(lum)` exactly
  (ignoring the integer truncation from `float → uint → float`). With
  255 bins spanning 30 EV stops, each bin is about 0.12 EV wide, which
  is well below perceptual thresholds.

- **`w`**: The number of pixels in that bin.
- **`weightedSum`**: Accumulates `logLum × pixelCount` — the total log-brightness
  weighted by pixel count.
- **`totalWeight`**: The total number of pixels **excluding bin 0** (near-black pixels).

**Bin 0 is skipped** intentionally. Near-black pixels (`lum < 1e-5`) carry almost no
useful visual information. Including them would pull the average luminance down and
cause the algorithm to compute a higher exposure, blowing out the bright areas just to
make already-invisible shadows slightly less black. For example, in a space scene with
90% black sky, including bin 0 would yield an extreme exposure (~1800x) that washes out
every star and planet. Excluding bin 0 lets the algorithm focus on the meaningful
brightness range.

This mirrors real-world camera metering, which excludes the darkest and brightest
extremes, and how human eyes naturally adapt to the meaningful brightness range rather
than pitch-black regions.

### Step 3 - Compute target exposure

```glsl
    float avgLogLum;
    if (totalWeight < 1.0)
        avgLogLum = minLogLum;
    else
        avgLogLum = weightedSum / totalWeight;

    float avgLum = exp2(avgLogLum);
    float targetExposure = 0.18 / max(avgLum, 1e-5);
```

- `avgLogLum` is the weighted average log2-luminance.
- `avgLum = exp2(avgLogLum)` converts back to linear luminance.
- `targetExposure = 0.18 / avgLum` is the **middle-gray key value**
  formula from photography. 0.18 (18% gray) is the standard
  reference: if the average scene luminance equals 0.18, the
  exposure should be 1.0. Brighter scenes get lower exposure,
  darker scenes get higher exposure.

After multiplication: `avgLum * (0.18 / avgLum) = 0.18`. The average pixel ends up at
0.18 in linear space, corresponding to roughly 50% perceived brightness (middle gray)
after gamma correction. This is why 18% gray cards are used as the reference point for
correct exposure in photography.

### Step 4 - Temporal smoothing

```glsl
    float speed = 1.0 - exp(-deltaTime * adaptSpeed);
    exposure = mix(exposure, targetExposure, speed);
}
```

- `exposure` is the **persisted** value from the previous frame
  (read from and written back to the same SSBO location).
- `speed` is an exponential smoothing factor based on frame time
  and `adaptSpeed` (default 1.5). This produces a smooth
  adaptation curve rather than an instant snap.
- When `adaptSpeed` is high, the eye adapts quickly.
  When low, it takes several seconds to fully adapt.
- The formula `1 - exp(-dt * speed)` is frame-rate independent:
  it produces the same visual adaptation speed at 30 FPS and
  144 FPS.

## CPU-Side Orchestration (`AutoExposure::Dispatch`)

The CPU side, in `AutoExposure.cpp`, performs these steps each
frame:

1. **First frame only**: `vkCmdFillBuffer` the exposure buffer
   with `0x3F800000` (IEEE 754 for 1.0f), then barrier.
2. **Clear histogram**: `vkCmdFillBuffer` with 0.
   (Subsequent frames are cleared by the shader itself in
   the average pass, but the first frame needs this.)
3. **Barrier**: transfer write → compute read/write.
4. **Bind histogram pipeline**, push constants
   (`minLogLum`, `invLogLumRange`, `inputSize`),
   dispatch `ceil(w/16) × ceil(h/16)`.
5. **Barrier**: compute write → compute read/write
   (histogram buffer must be visible to the average pass).
6. **Bind average pipeline**, push constants
   (`minLogLum`, `logLumRange`, `deltaTime`, `adaptSpeed`,
   `pixelCount`), dispatch 1 group.
7. **Barrier**: compute write → bottom of pipe
   (exposure buffer must be visible for later reads by the
   tone mapping fragment shader).

## How the Exposure Value Is Consumed

In `tonemap.frag`:

```glsl
layout(std430, set = 0, binding = 3) readonly buffer ExposureSSBO {
    float autoExposure;
};

// ...
float ev = useAutoExposure != 0 ? autoExposure : 1.0;
hdr *= ev * exp2(exposureBias);
```

The exposure multiplier is applied to the HDR color before
tone mapping.

**`exposureBias`** is a manual offset in **EV stops** (log2 units), exactly like the
exposure compensation dial on a camera (the +/- button). It acts as a relative
adjustment on top of the auto-computed value:

- `exposureBias = 0` → no change, the auto value is used as-is
- `exposureBias = +1` → `exp2(1) = 2×` brighter (one stop up)
- `exposureBias = -1` → `exp2(-1) = 0.5×` darker (one stop down)
- `exposureBias = +2` → `exp2(2) = 4×` brighter (two stops up)

This matches how exposure compensation works on real cameras — the metering system
picks a base exposure, and the photographer dials +/- stops from there. For example,
shooting a snow scene: auto exposure sees "very bright average" and darkens — making
snow look gray. Dialing in `exposureBias = +1` or `+2` tells the system "I know it's
bright, keep it bright."

## UI Parameters

- **Min EV** (default -10): The darkest luminance the histogram
  can represent. Pixels darker than `2^(-10)` go to bin 0.
- **Max EV** (default 20): The brightest luminance the histogram
  can represent.
- **Adapt Speed** (default 1.5): How fast the exposure converges.
  Higher = faster adaptation.
- **Exposure Bias** (in Tone Mapping panel, default 0): Manual
  EV offset applied on top of auto exposure.


Q&As
**HDR->LDR for histogram computation**
Q:why 
float logLum = clamp((log2(lum) - minLogLum) * invLogLumRange, 0.0, 1.0);
bin = uint(logLum * 254.0 + 1.0);

can map hdr value to ldr value?

A:The goal
We have a luminance value lum that can span an enormous range in HDR — from something like 0.00001 (deep shadow) to 1,000,000+ (the sun). We need to squeeze that into one of 256 integer bins (0–255). The key insight is: we don't map the luminance linearly, we map it in log2 space.
Why log2?
Human perception of brightness is roughly logarithmic. Doubling the light (1 EV stop) feels like the same perceptual step whether you go from 1→2 or from 1000→2000. So log2 is the natural space for binning luminance.
log2(lum) converts the luminance to EV (exposure value):
Luminance	log2(luminance)
0.001	-10
0.0625	-4
1.0	0
16.0	4
1048576.0	20
Now the entire HDR range is compressed from [0.001 … 1048576] to [-10 … 20] — a linear 30-unit range.
Step-by-step breakdown
Given the defaults minLogLum = -10 and maxLogLum = 20:
logLumRange    = maxLogLum - minLogLum = 30invLogLumRange = 1.0 / 30
Step 1: log2(lum) → converts to EV space
For example, lum = 1.0 → log2(1.0) = 0.0
Step 2: log2(lum) - minLogLum → shifts the range so the minimum becomes 0
0.0 - (-10) = 10.0
Now the range is [0 … 30] instead of [-10 … 20].
Step 3: * invLogLumRange → normalizes to [0 … 1]
10.0 * (1/30) = 0.333
Step 4: clamp(..., 0.0, 1.0) → safety clamp for values outside the expected range
Step 5: logLum * 254.0 + 1.0 → maps [0, 1] to [1, 255]
0.333 * 254.0 + 1.0 = 85.7 → bin 85
Why 254 + 1, not 255?
Bin 0 is reserved for near-black pixels (lum < 1e-5). These are excluded from the average calculation later because they represent pixels that shouldn't influence exposure (like pure black backgrounds). So the usable bins are 1 through 255, which is 255 bins, but the formula uses 254 because:
When logLum = 0.0: 0.0 * 254 + 1 = 1 → bin 1 (darkest non-black)
When logLum = 1.0: 1.0 * 254 + 1 = 255 → bin 255 (brightest)
That gives bins 1–255, exactly 255 values mapping to the normalized [0, 1] range.
This is NOT HDR→LDR tone mapping
It's important to clarify: this formula does not produce an LDR color. It produces a histogram bin index. The purpose is purely analytical — counting how many pixels fall at each brightness level. The actual HDR→LDR conversion happens later in the tone mapping pass.
What this formula does is:
HDR luminance (huge range)    → log2 (compress to EV stops)    → normalize to [0, 1] within the configured EV window    → quantize to an integer bin [1, 255]
It's essentially discretizing the continuous log-luminance distribution into 255 buckets so the average pass can compute a weighted mean. The inverse mapping in exposure_average.comp reverses this:
float logLum = (float(i) - 1.0) / 254.0 * logLumRange + minLogLum;
This reconstructs the log2-luminance at the center of each bin, which is then used to compute the weighted average scene brightness.

**Exclude near-black pixels**
Q:why always exclude near-black-pixels?
A:This is a practical design choice rooted in how real cameras and human eyes work. Here's why:
The problem without exclusion
Imagine a scene: a bright room with a window showing daylight. The image might be:
30% very dark pixels (deep shadows under furniture, behind objects)
50% mid-range pixels (walls, floor, objects)
20% bright pixels (window, light sources)
If you include the near-black pixels in the average, they pull the average luminance down, which causes the algorithm to compute a higher exposure. That higher exposure blows out the bright areas (the window becomes pure white, walls look overexposed) just to make already-invisible shadows slightly less black.
Near-black pixels carry almost no useful visual information — lum < 1e-5 means the pixel is essentially indistinguishable from pure black to human eyes. Letting them influence exposure is counterproductive.
Real-world analogy
Camera metering: Professional cameras exclude the darkest and brightest extremes when computing exposure. A common technique is center-weighted or matrix metering that ignores outliers on both ends. The histogram approach here is a simplified version of the same idea — skip the bottom bin.
Human eye: When you walk into a dim room with a bright TV screen, your eyes adapt to the meaningful brightness (the TV, the furniture you can see), not the pitch-black corners behind the couch. Your visual system naturally ignores regions below the perception threshold.
What would happen if bin 0 were included?
Consider a space scene: 90% black sky (bin 0), 10% stars and planets.
Scenario	avgLum	targetExposure (0.18/avgLum)
With bin 0	Very low (~0.0001)	Very high (~1800)
Without bin 0	Moderate (~0.5)	Moderate (~0.36)
With bin 0 included, the exposure skyrockets to ~1800x, completely blowing out every star and planet into pure white. Without bin 0, the algorithm focuses on the actual visible content and produces a sensible exposure.
Why only exclude the bottom, not the top?
Extremely bright pixels (bin 255) are kept because they usually represent real scene content you want correctly exposed — the sun, a lamp, a specular highlight. Overexposing those is acceptable (they're already at the top of the range), but underexposing the mid-range to compensate for near-black pixels is not.
That said, more advanced implementations do trim both ends (e.g., ignore the bottom 5% and top 5% of the histogram). This engine uses the simpler "skip bin 0 only" approach, which works well in practice for most scenes.

**18% Gray**

Q: Because float targetExposure = 0.18 / max(avgLum, 1e-5);  hdr *= ev * exp2(exposureBias);  make the average brightness to 0.18, right? what is exposureBias？ is it exposure compensation?

A:Yes, that's correct on both points.
targetExposure = 0.18 / avgLum
This formula ensures that the average scene luminance gets mapped to 0.18 (middle gray) after multiplication. In the tone mapping shader:
hdr *= exposure;    // hdr_avg * (0.18 / hdr_avg) = 0.18
The average pixel ends up at 0.18 in linear space, which corresponds to roughly 50% perceived brightness (middle gray) after gamma correction. This is the standard key value from photography — 18% gray cards are used as the reference point for correct exposure.
exposureBias — yes, it is exposure compensation
hdr *= ev * exp2(exposureBias);
exposureBias is a manual offset in EV stops (log2 units), exactly like the exposure compensation dial on a camera (the +/- button marked with EV).
exposureBias = 0 → no change, the auto value is used as-is
exposureBias = +1 → exp2(1) = 2x brighter (one stop up)
exposureBias = -1 → exp2(-1) = 0.5x darker (one stop down)
exposureBias = +2 → exp2(2) = 4x brighter (two stops up)
It lets the user override the auto exposure system. For example, if you're shooting a snow scene, auto exposure sees "very bright average" and compensates by darkening — making the snow look gray. Dialing in exposureBias = +1 or +2 tells the system "I know it's bright, keep it bright."
Combined formula breakdown
hdr *= ev * exp2(exposureBias);
Component	Meaning
ev	Auto exposure value (0.18 / avgLum), or 1.0 if auto exposure is off
exp2(exposureBias)	Manual compensation in EV stops
Product	Final multiplier applied to HDR color before tone mapping
The two are multiplied together, so exposureBias acts as a relative adjustment on top of the auto-computed value, not a replacement. This matches exactly how exposure compensation works on real cameras — the metering system picks a base exposure, and the photographer dials +/- stops from there.