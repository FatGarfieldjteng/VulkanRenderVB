# Tone Mapping

## Why Tone Mapping Exists

An HDR rendering pipeline produces pixel values that can range from 0 to thousands (or
millions for the sun). A monitor can only display values in [0, 1]. **Tone mapping**
compresses the infinite HDR range into the displayable [0, 1] range while preserving the
perceptual appearance of the scene.

A naive approach like `clamp(hdr, 0, 1)` would clip everything above 1.0 to pure white —
losing all highlight detail. A better approach is a curve that:

- Passes dark values through roughly unchanged (the **toe**)
- Has a near-linear middle section that preserves contrast
- Gradually compresses bright values, never quite reaching 1.0 (the **shoulder**)

This is what a filmic tone mapping curve does.

## Pre-Tone-Mapping Steps

Before either tone curve is applied, the shader combines multiple inputs:

```glsl
vec3 hdr = texture(hdrColor, fragUV).rgb;    // 1. Read HDR pixel
float ao = texture(aoTex, fragUV).r;          // 2. Read ambient occlusion
vec3 bloom = texture(bloomTex, fragUV).rgb;   // 3. Read bloom

hdr *= ao;                                    // 4. Darken by AO
hdr += bloom * bloomStrength;                 // 5. Add bloom glow

float ev = useAutoExposure != 0 ? autoExposure : 1.0;
hdr *= ev * exp2(exposureBias);               // 6. Apply exposure
```

The order matters:

1. **AO** is applied multiplicatively (darkens contact shadows)
2. **Bloom** is added (bright glow from nearby highlights)
3. **Exposure** scales the entire result — this sets the overall brightness before the tone
   curve compresses it

All of this happens in **linear HDR space** before the non-linear tone curve, which is
important because the tone curve is non-linear — adding bloom after tone mapping would
produce different (worse) results.

## Rendering Technique

Tone mapping runs as a **fullscreen triangle** graphics pass. The fragment shader reads four
inputs via descriptors:

| Binding | Type          | Content                                       |
|---------|---------------|-----------------------------------------------|
| 0       | `sampler2D`   | HDR color from forward pass                   |
| 1       | `sampler2D`   | Bloom texture (mip 0)                         |
| 2       | `sampler2D`   | SSAO texture (or white placeholder if disabled)|
| 3       | `SSBO`        | Auto exposure value (single float)            |

The shader branches on `curveType` (push constant) to select ACES or AgX:

```glsl
vec3 mapped;
if (curveType == 0)
    mapped = ACESFilmic(hdr);
else
    mapped = AgXToneMap(hdr);

mapped = clamp(mapped, 0.0, 1.0);
```

---

## Method 1: ACES Filmic

### Origin

The formula comes from real photographic film. In the 1890s, Hurter and Driffield studied
how chemical silver halide grains in film respond non-linearly to light exposure, publishing
the "H&D curve" (characteristic curve) — an S-shaped plot of film density vs. log exposure.

Kodak later published parametric models of this curve for their film stocks, with parameters
controlling the toe, linear section, and shoulder. **John Hable** (Naughty Dog) adapted
Kodak's parametric film curve into a GPU-friendly rational function for **Uncharted 2**
(GDC 2010). The game industry adopted this as "ACES Filmic" tone mapping.

The name "ACES" is a simplification — the real ACES RRT (Academy Reference Rendering
Transform) is more complex. What game engines call "ACES" is the Hable/Uncharted 2 curve
that was inspired by the ACES look.

### The S-Curve Shape

```
Output (LDR)
1.0 |                    ______________ shoulder (compression)
    |                  /
    |                /
    |              /    ← linear section (contrast preserved)
    |            /
    |          /
    |        /
    |      /
0.0 |____/                              toe (shadow lift)
    +--------+--------+--------+------→ Input (HDR)
    0        1        5        11.2
                              (white point)
```

### The Formula

```glsl
vec3 ACESFilmic(vec3 x) {
    float A = shoulderStrength;   // default 2.51
    float B = linearStrength;     // default 0.03
    float C = linearAngle;        // default 2.43
    float D = toeStrength;        // default 0.59
    float W = whitePoint;         // default 11.2

    vec3 num = (x * (A * x + C * B) + D * 0.02);
    vec3 den = (x * (A * x + B) + D * 0.3);
    vec3 mapped = num / den - vec3(0.02 / 0.3);

    float wNum = (W * (A * W + C * B) + D * 0.02);
    float wDen = (W * (A * W + B) + D * 0.3);
    float wMapped = wNum / wDen - 0.02 / 0.3;

    return mapped / wMapped;
}
```

### The Core Rational Function

The heart is a per-channel function `f(x)`:

```
f(x) = (x(Ax + CB) + D×0.02) / (x(Ax + B) + D×0.3) - 0.02/0.3
```

Expanding:

```
Numerator:   A×x² + C×B×x + 0.02×D
Denominator: A×x² + B×x   + 0.3×D
```

Both are quadratic in `x`. The ratio of two quadratics naturally produces an S-curve:

- **Small x** (near zero): Constant terms dominate. Ratio → `D×0.02 / D×0.3 = 0.02/0.3`,
  and the `- 0.02/0.3` offset cancels it → output ≈ 0. The curve starts flat (the toe).
- **Medium x**: Linear terms (`CB×x` and `B×x`) dominate. Ratio ≈ `CB/B = C`, giving a
  nearly constant slope. The curve rises linearly (the linear section). C controls the
  slope — hence "Linear Angle".
- **Large x**: Quadratic terms (`A×x²`) dominate. Ratio → `A×x²/A×x² = 1`. The curve
  flattens toward an asymptote (the shoulder).

The `- 0.02/0.3` offset ensures `f(0) = 0` (black stays black):

```
f(0) = (0 + 0 + 0.02D) / (0 + 0 + 0.3D) - 0.02/0.3
     = 0.02/0.3 - 0.02/0.3
     = 0
```

### White Point Normalization

Without normalization, `f(x)` approaches a value less than 1.0 asymptotically. To ensure
white maps to 1.0, the curve is divided by its value at the white point W:

```glsl
return mapped / wMapped;    // = f(x) / f(W)
```

This ensures `f(W) / f(W) = 1.0`. With default `W = 11.2`, any HDR value ≥ 11.2 maps to
pure white.

### The Constants 0.02 and 0.3

These come from Kodak's film model:

- `0.02` in the numerator: relates to the minimum density (base fog) of unexposed film
- `0.3` in the denominator: relates to the development contrast

These values were empirically determined from real film stock measurements.

### What Each Parameter Controls

**A — Shoulder Strength (default 2.51):** How aggressively bright values are compressed.
Higher A = tighter shoulder, more compression. Lower A = gentler rolloff, more highlight
detail.

**B — Linear Strength (default 0.03):** Interacts with C to define mid-tone contrast.
Increasing B raises overall brightness in mid-tones.

**C — Linear Angle (default 2.43):** Multiplies with B to form the `C×B` term. Controls the
slope of the linear section — essentially mid-tone contrast. Higher C = steeper = more
contrast.

**D — Toe Strength (default 0.59):** Controls the shadow region. Higher D lifts the toe
(brighter shadows). Lower D deepens the toe (darker shadows).

**W — White Point (default 11.2):** The HDR value that maps to output 1.0. Lower W clips
earlier (brighter overall but loses highlights). Higher W preserves highlights but the image
appears darker.

### Example Values with Defaults

| HDR input | f(x)/f(W) | Perceived      |
|-----------|-----------|----------------|
| 0.0       | 0.0       | Black          |
| 0.18      | 0.058     | Dark           |
| 1.0       | 0.372     | Mid-tone       |
| 2.0       | 0.590     | Bright         |
| 5.0       | 0.847     | Very bright    |
| 11.2      | 1.0       | White          |
| 50.0      | clamped 1 | Clipped white  |

---

## Method 2: AgX

### What is AgX?

AgX (from the chemical symbol Ag for silver — a nod to silver-based film) was created by
**Troy Sobotka** as an alternative to ACES. It was designed to solve specific problems:

1. **ACES over-saturates bright colors**: A bright red light stays intensely red even at
   extreme brightness. In reality, very bright colored lights should desaturate toward
   white — a red-hot metal eventually glows white, not intensely red.
2. **ACES produces hue shifts**: Saturated colors can shift hue (e.g., bright blue → purple).
3. **ACES has harsh highlight clipping**: The transition to white can feel abrupt for
   saturated colors.

AgX addresses all of these by working in a **perceptually uniform log-encoded color space**
where highlight desaturation and smooth rolloff happen naturally.

### The Pipeline

```
Linear HDR input (sRGB primaries)
    ↓
[1] Transform to AgX color space      (matrix multiply)
    ↓
[2] Log2 encode + clamp to EV range   (compress dynamic range)
    ↓
[3] Normalize to [0, 1]               (prepare for contrast curve)
    ↓
[4] Apply contrast curve              (6th-degree polynomial)
    ↓
[5] Transform back to sRGB            (inverse matrix)
    ↓
[6] Post-processing: saturation + punch
    ↓
LDR output
```

### Stage 1: Transform to AgX Color Space

```glsl
const mat3 agxTransform = mat3(
    0.842479062253094,  0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772,  0.0784336,
    0.0792237451477643, 0.0791661274605434,  0.879142973793104
);

color = agxTransform * color;
```

This 3x3 matrix converts from sRGB linear primaries into the AgX log-encoding color space.
The off-diagonal terms (~0.04–0.08) are significant — each output channel is a **blend of
all three input channels**.

This is the key difference from ACES (which applies tone curves per-channel in sRGB). By
mixing channels before the tone curve, AgX ensures that a pure bright red `(100, 0, 0)`
gets some green and blue mixed in → it naturally desaturates toward white as brightness
increases. Per-channel mapping (ACES) preserves saturation at extreme brightness, producing
neon-looking highlights. AgX's cross-channel mixing avoids this.

### Stage 2: Log2 Encode and Clamp

```glsl
const float minEV = -12.47393;
const float maxEV = 4.026069;

color = clamp(log2(color), minEV, maxEV);
```

Each channel is converted to log2 space (exposure values), then clamped:

- `minEV = -12.47393` → linear value `2^(-12.47) ≈ 0.000175` (near-black)
- `maxEV = 4.026069` → linear value `2^4.03 ≈ 16.3` (bright highlight)
- Total dynamic range: `16.5 EV stops`

Log2 is used because human perception is logarithmic — equal intervals in log space
correspond to equal perceptual brightness steps. The subsequent contrast curve operates in a
perceptually uniform domain.

### Stage 3: Normalize to [0, 1]

```glsl
color = (color - minEV) / (maxEV - minEV);
```

Linearly maps the clamped log2 range [minEV, maxEV] to [0, 1]:

| Log2 value      | Normalized | Meaning                        |
|-----------------|------------|--------------------------------|
| -12.47 (minEV)  | 0.0        | Darkest representable          |
| -4.22           | 0.5        | Mid-point (~0.054 linear)      |
| 4.03 (maxEV)    | 1.0        | Brightest representable        |

### Stage 4: Contrast Curve (6th-Degree Polynomial)

```glsl
vec3 AgXDefaultContrastApprox(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return + 15.5     * x4 * x2    // x^6
           - 40.14    * x4 * x     // x^5
           + 31.96    * x4         // x^4
           - 6.868    * x2 * x     // x^3
           + 0.4298   * x2         // x^2
           + 0.1191   * x          // x^1
           - 0.00232;              // constant
}
```

This 6th-degree polynomial approximates the AgX "default contrast" look. It maps [0, 1] →
[0, 1] with an S-curve shape:

```
f(x) = 15.5x⁶ - 40.14x⁵ + 31.96x⁴ - 6.868x³ + 0.4298x² + 0.1191x - 0.00232
```

| Input (x) | Output f(x) | Region          |
|-----------|-------------|-----------------|
| 0.0       | ≈ 0         | Black           |
| 0.1       | 0.009       | Deep shadow     |
| 0.3       | 0.044       | Shadow          |
| 0.5       | 0.183       | Mid-tone        |
| 0.7       | 0.537       | Bright mid-tone |
| 0.9       | 0.919       | Highlight       |
| 1.0       | ≈ 1         | White           |

**Why a polynomial instead of ACES's rational function?**

1. **No division needed** — just multiply-accumulate, which is faster on GPUs.
2. **Easy to fit**: Coefficients were obtained by least-squares fitting against the
   reference AgX contrast curve (defined as a lookup table). 6th-degree provides sufficient
   accuracy within perceptual thresholds.
3. **Smooth**: Polynomials are infinitely differentiable — no kinks or discontinuities.

The computation is structured to minimize operations: precomputing `x2 = x*x` and
`x4 = x2*x2` means powers up to x⁶ require only 4 multiplies, then 7 multiply-adds for
the weighted sum.

### Stage 5: Transform Back to sRGB

```glsl
const mat3 agxTransformInv = mat3(
    1.19687900512017,   -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417,   -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433,  1.15107367264116
);

color = agxTransformInv * color;
```

The inverse of the forward AgX matrix, converting back to sRGB linear primaries. The
diagonal values (~1.15–1.20) compensate for the channel mixing in the forward transform.

### Stage 6: Post-Processing (Saturation and Punch)

**Saturation control:**

```glsl
vec3 lum = vec3(dot(color, vec3(0.2126, 0.7152, 0.0722)));
color = lum + (color - lum) * saturation;
```

`(color - lum)` is the chrominance. Multiplying by `saturation`:

- `saturation = 1.0` (default): no change
- `saturation > 1.0`: more vivid colors
- `saturation < 1.0`: washed out
- `saturation = 0.0`: grayscale

This is needed because AgX intentionally desaturates highlights. The saturation control lets
artists add color back if the result looks too muted.

**Punch (contrast boost):**

```glsl
color = mix(color, color * color * (3.0 - 2.0 * color), agxPunch);
```

`color * color * (3.0 - 2.0 * color)` is the **smoothstep** function: `3x² - 2x³`. It maps
[0,1] → [0,1] with an S-curve that darkens darks and brightens brights:

| Input | smoothstep | Effect             |
|-------|------------|--------------------|
| 0.0   | 0.0        | Black stays black  |
| 0.25  | 0.156      | Darks get darker   |
| 0.5   | 0.5        | Mid-point unchanged|
| 0.75  | 0.844      | Brights get brighter|
| 1.0   | 1.0        | White stays white  |

`agxPunch` blends between original and contrast-boosted: 0 = no boost, 1 = full smoothstep
contrast.

---

## ACES vs AgX Comparison

| Aspect                  | ACES                                     | AgX                                       |
|-------------------------|------------------------------------------|-------------------------------------------|
| Highlight desaturation  | Minimal — bright red stays red           | Natural — bright red → orange → white     |
| Hue stability           | Can shift (blue → purple)                | More stable                               |
| Shadow detail           | Good (adjustable toe)                    | Good (log encoding preserves range)       |
| Mid-tone contrast       | Adjustable via 5 parameters              | Fixed by polynomial, adjustable via punch |
| Parameterization        | 5 curve parameters (A, B, C, D, W)       | 2 post-process parameters (saturation, punch) |
| Computation             | Rational function (1 divide)             | Polynomial + 2 matrix multiplies (no divide) |
| Look                    | Classic Hollywood film                   | More neutral/photographic                 |

---

## UI Parameters

### ACES

| Parameter            | Range      | Default | Effect                                |
|----------------------|------------|---------|---------------------------------------|
| Shoulder Strength (A)| 0.0–10.0   | 2.51    | Highlight compression aggressiveness  |
| Linear Strength (B)  | 0.0–1.0    | 0.03    | Mid-tone brightness                   |
| Linear Angle (C)     | 0.0–5.0    | 2.43    | Mid-tone contrast (slope)             |
| Toe Strength (D)     | 0.0–2.0    | 0.59    | Shadow lift / darkness                |
| White Point (W)      | 1.0–30.0   | 11.2    | HDR value that becomes pure white     |

### AgX

| Parameter  | Range    | Default | Effect                       |
|------------|----------|---------|------------------------------|
| Saturation | 0.0–3.0  | 1.0     | Color intensity post-mapping |
| Punch      | 0.0–1.0  | 0.0     | Contrast boost via smoothstep|

### Shared

| Parameter          | Range       | Default | Effect                              |
|--------------------|-------------|---------|-------------------------------------|
| Curve              | ACES / AgX  | ACES    | Which tone mapping algorithm        |
| Exposure Bias (EV) | -5.0 to 5.0 | 0.0    | Manual brightness offset in EV stops|

---

## Q&A

### Q: The ACES formula looks like magic — how did the author come up with it?

### A:

The formula wasn't invented from scratch — it was **reverse-engineered from real
photographic film**.

**Historical lineage:**

```
Real photographic film chemistry (1890s, Hurter & Driffield)
    ↓
Kodak parametric film curve model (published in technical specs)
    ↓
John Hable adapts to GPU-friendly rational function (2010, GDC)
    ↓
Game industry adopts as "ACES Filmic" / "Uncharted 2 tone mapping"
    ↓
Your shader's ACESFilmic() function
```

When light hits photographic film, the chemical silver halide grains respond non-linearly.
Kodak studied this and published characteristic curve models with parameters named "shoulder
strength", "linear strength", "linear angle", and "toe strength" — the exact names in the
shader.

John Hable (Naughty Dog) took Kodak's parametric model and expressed it as a rational
function (ratio of two quadratics) that runs in microseconds on a GPU. He presented this at
GDC 2010 for Uncharted 2.

**Why a ratio of two quadratics produces an S-curve:**

- **Small x**: Constant terms dominate → ratio ≈ `D×0.02 / D×0.3`, offset cancels → 0.
  The curve starts flat (the toe).
- **Medium x**: Linear terms dominate → ratio ≈ `CB×x / B×x = C`, giving constant slope.
  The curve rises linearly (the linear section).
- **Large x**: Quadratic terms dominate → ratio ≈ `A×x² / A×x² = 1`. The curve flattens
  (the shoulder).

This flat → linear → flat progression is the S-curve, arising naturally from the math.

**The specific constants 0.02 and 0.3** come from Kodak's film model — they represent the
film's base fog level and development factor, empirically determined from real film stock
measurements.

**The default parameter values** (A=2.51, B=0.03, C=2.43, D=0.59, W=11.2) were published
by Hable as producing a curve that closely matches a standard film print. They were tuned by
curve-fitting against empirical film response data.

### Q: Why does AgX mix color channels before applying the tone curve, while ACES doesn't?

### A:

This is the fundamental design difference between the two approaches.

**ACES** applies the tone curve **per-channel** in sRGB space. Each of R, G, B is compressed
independently. This means a pure bright red `(100, 0, 0)` stays as `(f(100), f(0), f(0))`
— still pure red, just compressed. The result: highlights retain their original hue and
saturation, producing vivid, neon-like bright colors.

**AgX** multiplies by a 3x3 matrix **before** the tone curve, blending channels:

```
out.r = 0.842×R + 0.043×G + 0.042×B
out.g = 0.078×R + 0.878×G + 0.078×B
out.b = 0.079×R + 0.079×G + 0.879×B
```

A pure bright red `(100, 0, 0)` becomes `(84.2, 7.8, 7.9)` — already partially desaturated
before the tone curve even runs. The brighter the input, the more the cross-channel terms
matter, naturally pushing highlights toward white.

This matches reality: in the physical world, a red-hot metal glows red at moderate
temperature, then orange, then yellow-white at extreme temperature. The color desaturates
toward white as energy increases. ACES doesn't model this; AgX does.

### Q: What does the smoothstep function `color * color * (3.0 - 2.0 * color)` do in AgX punch?

### A:

Smoothstep is a standard mathematical function that maps [0,1] → [0,1] with an S-curve.
The formula `3x² - 2x³` has these properties:

- `f(0) = 0` and `f(1) = 1` (endpoints preserved)
- `f'(0) = 0` and `f'(1) = 0` (flat at both ends — smooth transition)
- `f(0.5) = 0.5` (mid-point preserved)
- Below 0.5: output < input (darks get darker)
- Above 0.5: output > input (brights get brighter)

This is a **contrast boost** — it stretches the difference between darks and lights while
keeping black, white, and middle gray unchanged. The `agxPunch` parameter controls how much
of this contrast boost is blended in:

```glsl
color = mix(color, smoothstep(color), agxPunch);
```

At `agxPunch = 0`: no change. At `agxPunch = 1`: full contrast boost. This gives artists a
simple dial to add "pop" to the AgX result, which can otherwise look quite flat and neutral.

### Q: Does W (white point) in ACES Filmic have a physical meaning?

### A:

Yes — W represents the **scene luminance value that should be rendered as pure white** on
screen. Anything at or above W maps to output 1.0 (clipped to pure white).

**What 11.2 corresponds to physically:**

After auto exposure scales the scene so average luminance sits around 0.18 (middle gray),
the value 11.2 is roughly **6 stops above middle gray**:

```
log2(11.2 / 0.18) ≈ 6 stops
```

This is a reasonable dynamic range for a well-exposed scene:

- Middle gray (0.18) → moderate indoor surfaces
- 1 stop up (0.36) → well-lit wall
- 3 stops up (1.44) → bright surface in sunlight
- 6 stops up (11.2) → specular highlight, edge of a light source → pure white

**What changing W does:**

| W value | Stops above mid-gray | Effect                                              |
|---------|----------------------|------------------------------------------------------|
| 5.0     | ~4.8                 | Brighter image, highlights clip to flat white early  |
| 11.2    | ~6.0 (default)       | Balanced — most highlights preserve detail           |
| 20.0    | ~6.8                 | Darker, but bright lights show gradient before white |
| 30.0    | ~7.4                 | Very dark — more range compressed into [0, 1]        |

**Real-world analogy:** Think of W as the camera's ISO + aperture + shutter speed
combination that determines which brightness clips to white. A low W is like overexposing —
brighter overall but highlight detail lost. A high W is like underexposing — more highlight
detail preserved but the image looks darker and flatter.

The default 11.2 was chosen by John Hable to match the look of standard cinema film
projection — it preserves enough highlight detail for most scenes without making the overall
image feel flat.
