# What is Pipeline Cache and How Does It Accelerate Pipeline Creation?

## The Observation

`mGraphicsPipeline` is created via `vkCreateGraphicsPipelines` every time the application runs. The pipeline object itself is always recreated. So how does the `PipelineManager` help?

## What `VkPipelineCache` Actually Caches

The key insight is: `VkPipelineCache` does **not** cache the `VkPipeline` object. It caches the **intermediate compilation results** that the driver produces internally during pipeline creation.

When you call `vkCreateGraphicsPipelines`, the driver internally:

1. Takes your SPIR-V shaders
2. Compiles them to the GPU's native ISA (instruction set architecture) -- this is GPU-specific machine code
3. Resolves pipeline state combinations (rasterizer settings, blend modes, vertex layout, etc.)
4. Produces the final `VkPipeline`

Steps 1-3 are the expensive part (especially shader compilation). The `VkPipelineCache` stores these intermediate compilation artifacts so the driver can skip them next time.

## The Lifecycle Across Runs

**First run (cold start):**

```
Initialize()          --> creates empty VkPipelineCache
LoadCache("file")     --> file doesn't exist, cache stays empty
vkCreateGraphicsPipelines(device, mPipelines.GetCache(), ...)
                      --> driver compiles shaders from scratch (SLOW)
                      --> driver stores compilation results INTO the cache
SaveCache("file")     --> serializes cache to disk (pipeline_cache.bin)
```

**Second run (warm start):**

```
Initialize()          --> creates empty VkPipelineCache
LoadCache("file")     --> file exists, cache is seeded with previous data
vkCreateGraphicsPipelines(device, mPipelines.GetCache(), ...)
                      --> driver finds matching compilation in cache
                      --> skips shader recompilation (FAST)
SaveCache("file")     --> saves (potentially updated) cache to disk
```

The critical line in `Application::CreatePipeline`:

```cpp
VK_CHECK(vkCreateGraphicsPipelines(device, mPipelines.GetCache(), 1,
                                   &pipelineInfo, nullptr, &mGraphicsPipeline));
```

The second argument is the `VkPipelineCache`. If you passed `VK_NULL_HANDLE`, the driver would compile from scratch every time. By passing a populated cache, the driver can look up previously compiled shader variants.

## How the Cache Data Works Internally

The cache data blob (what gets saved to `pipeline_cache.bin`) is an opaque, driver-specific binary. It contains:

- A header with vendor ID, device ID, driver version, and a UUID -- so it's **invalidated** if you switch GPUs or update your driver
- Compiled shader bytecode in the GPU's native format
- Pipeline state lookup tables keyed by a hash of the pipeline create info

When `LoadCache` provides this data to `vkCreatePipelineCache`, the driver validates the header. If the vendor/device/driver match, the internal data is usable. If not, the driver silently ignores it and starts fresh.

## Why the Pipeline Object Is Still Recreated

The `VkPipeline` object is a runtime handle that ties together GPU state: compiled shaders, fixed-function settings, resource bindings, etc. It can't be serialized to disk because:

- It contains GPU memory pointers that change each run
- The driver may store it in device-local memory
- It references other live objects (pipeline layout, render pass info, etc.)

So you always call `vkCreateGraphicsPipelines` -- but with a warm cache, the driver skips the expensive shader compilation and just assembles the final object from cached intermediates. This can be **10-100x faster** depending on the driver and shader complexity.

## When It Matters

With just one simple pipeline (like Phase 2), the difference is barely noticeable (maybe a few milliseconds). But in a real game engine with hundreds of shader variants (different material types, shadow passes, post-processing effects), cold pipeline creation can take **seconds**. A warm cache brings that down to near-instant.

## Summary

| Aspect | Without cache | With cache |
|--------|--------------|------------|
| `vkCreateGraphicsPipelines` called? | Yes | Yes |
| `VkPipeline` object created? | Yes | Yes |
| Shaders compiled from SPIR-V to GPU ISA? | Yes (slow) | No, reused from cache (fast) |
| Pipeline state resolved? | Yes | Partially cached |

The `PipelineManager` doesn't avoid pipeline creation -- it avoids **recompilation**.
