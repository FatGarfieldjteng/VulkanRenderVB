# Compute Shader Execution Model

Compute shaders have a three-level hierarchy for organizing parallel work:

## Thread (Invocation)

A **thread** is the smallest unit of execution. Each thread runs the same shader code (`main()`) but with a unique ID. In Vulkan/GLSL, a single thread is officially called an **invocation**.

Each thread has a built-in variable `gl_LocalInvocationIndex` (a flat 1D index within its workgroup) and `gl_GlobalInvocationID` (a 3D index across the entire dispatch).

One thread = one execution of `main()`.

## Workgroup (Local)

A **workgroup** is a group of threads that execute together and can cooperate. You define the workgroup size in the shader:

```glsl
layout(local_size_x = 16, local_size_y = 16) in;  // 16 x 16 = 256 threads per workgroup
```

What makes threads within a workgroup special:

- **Shared memory**: They can read/write the same `shared` variables (on-chip, fast).
- **Barriers**: They can synchronize with `barrier()` — all threads in the workgroup must reach the barrier before any can proceed past it.
- **Co-scheduling**: The GPU guarantees all threads in a workgroup are running "at the same time" on the same compute unit (CU / SM).

Threads in **different** workgroups cannot communicate through shared memory and cannot synchronize with barriers. They can only communicate through global memory (SSBOs) with atomics.

## Dispatch (Global)

The **dispatch** is the CPU command that launches all the workgroups:

```cpp
vkCmdDispatch(cmd, groupsX, groupsY, groupsZ);
```

This creates `groupsX × groupsY × groupsZ` workgroups. Each workgroup then contains `local_size_x × local_size_y × local_size_z` threads.

**Total threads = workgroups × threads per workgroup.**

## Visual Hierarchy

```
vkCmdDispatch(120, 68, 1)          ← 120 × 68 = 8,160 workgroups
│
├─ Workgroup (0,0,0)               ← 16 × 16 = 256 threads
│  ├─ Thread 0   (gl_LocalInvocationIndex = 0)
│  ├─ Thread 1   (gl_LocalInvocationIndex = 1)
│  ├─ ...
│  └─ Thread 255 (gl_LocalInvocationIndex = 255)
│  └─ [shared memory: sharedBins[256]]
│
├─ Workgroup (1,0,0)               ← 256 threads, own shared memory
│  ├─ Thread 0
│  ├─ ...
│  └─ Thread 255
│  └─ [shared memory: sharedBins[256]]
│
├─ ...
└─ Workgroup (119,67,0)
   └─ ...
```

## Built-in ID Variables

| Variable | Type | Meaning |
|---|---|---|
| `gl_LocalInvocationID` | uvec3 | Thread position within its workgroup (e.g., `(5, 3, 0)`) |
| `gl_LocalInvocationIndex` | uint | Flat 1D index within workgroup (e.g., `5 + 3*16 = 53`) |
| `gl_WorkGroupID` | uvec3 | Which workgroup this is (e.g., `(42, 15, 0)`) |
| `gl_GlobalInvocationID` | uvec3 | Unique position across entire dispatch = `WorkGroupID * WorkGroupSize + LocalInvocationID` |
| `gl_NumWorkGroups` | uvec3 | Total number of workgroups dispatched (e.g., `(120, 68, 1)`) |
| `gl_WorkGroupSize` | uvec3 | Size declared in shader (e.g., `(16, 16, 1)`) |

## Concrete Example from the Histogram Shader

```glsl
layout(local_size_x = 16, local_size_y = 16) in;   // 256 threads per workgroup
```

```cpp
vkCmdDispatch(cmd, (1920+15)/16, (1080+15)/16, 1);  // = dispatch(120, 68, 1)
```

- **Total workgroups**: 120 × 68 × 1 = 8,160
- **Threads per workgroup**: 16 × 16 × 1 = 256
- **Total threads**: 8,160 × 256 = 2,088,960
- Each thread processes **one pixel**: `gl_GlobalInvocationID.xy` maps directly to pixel coordinates.
- For a 1920×1080 image (2,073,600 pixels), there are 15,360 excess threads (from rounding up to multiples of 16). Those excess threads are filtered out by the bounds check `if (coord.x < inputSize.x && coord.y < inputSize.y)`.

## Hardware Mapping (for intuition)

On actual GPU hardware, threads don't run individually — they're grouped into **waves** (AMD) or **warps** (NVIDIA) of 32 or 64 threads that execute in lockstep (SIMD). A workgroup may contain multiple waves/warps. The GPU schedules entire workgroups onto a single Compute Unit (CU on AMD / SM on NVIDIA), which is why shared memory and barriers work — they're local to that hardware unit.
