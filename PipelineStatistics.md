# Pipeline Statistics

## Why PipelineStatistics exists alongside GPUProfiler

They measure completely different things:

**GPUProfiler** answers: **"How long did each pass take?"** (timing)
- Uses `VK_QUERY_TYPE_TIMESTAMP` queries
- Outputs durations in milliseconds (e.g. Shadow: 0.224 ms, Forward: 0.234 ms)

**PipelineStatistics** answers: **"How much work did the GPU actually do?"** (workload counters)
- Uses `VK_QUERY_TYPE_PIPELINE_STATISTICS` queries
- Outputs raw counts:
  - `vertexShaderInvocations` — how many vertices were shaded
  - `fragmentShaderInvocations` — how many fragments were shaded
  - `computeShaderInvocations` — how many compute shader invocations ran
  - `clippingPrimitives` — how many primitives survived clipping

## How they complement each other

They complement each other. For example, if you enable GPU culling and the Forward pass drops from 0.5 ms to 0.2 ms, the profiler tells you *that* it got faster, but the pipeline statistics tell you *why* — say fragment shader invocations dropped from 2M to 800K because culled objects are no longer being rasterized. Timing alone can't tell you that.

## How it works

Pipeline statistics use `VK_QUERY_TYPE_PIPELINE_STATISTICS` with a single query per frame-in-flight. The query is bracketed around pass execution using `vkCmdBeginQuery` / `vkCmdEndQuery`, and the GPU accumulates hardware counters for the duration.

The requested statistics are configured at pool creation time via flags:

```cpp
static constexpr VkQueryPipelineStatisticFlags kStatFlags =
    VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
    VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
    VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT |
    VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT;
```

Results are read back via `vkGetQueryPoolResults` on the next frame (after the fence confirms GPU completion), returning a `uint64_t` array with one value per requested statistic.

The feature is togglable at runtime (`SetEnabled` / `IsEnabled`) and disabled by default to avoid any overhead when not needed.
