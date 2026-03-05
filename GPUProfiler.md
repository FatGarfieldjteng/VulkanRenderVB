# GPU Profiler

## Core Idea

GPU work runs asynchronously from the CPU. You can't just use `std::chrono` to time it. Instead, the GPU has a **timestamp counter** that ticks at a hardware-specific rate. Vulkan provides **timestamp queries** — you insert "write the current GPU timestamp here" commands into the command buffer, and later read back the values from the CPU side.

## Initialization

- Queries `timestampPeriod` from the physical device — this is how many **nanoseconds** each GPU timestamp tick represents (e.g. 1.0 ns/tick on an RTX 3070 Ti).
- Creates one `VkQueryPool` **per frame-in-flight** (2 pools). This is critical — while frame N is being read back on the CPU, frame N+1 can be recording new timestamps on the GPU without conflict.
- Each pool has `maxScopes * 2 + 2` slots: 2 per scope (begin + end) plus 2 for the whole-frame bracket.

## Recording (GPU side, during command buffer building)

Each frame follows this pattern, driven from `Application.cpp` and `RenderGraph.cpp`:

```
BeginFrame(cmd)              → writes timestamp #0 (top of pipe)
  BeginScope(cmd, "Shadow")  → writes timestamp #1 (top of pipe)
  EndScope(cmd)              → writes timestamp #2 (bottom of pipe)
  BeginScope(cmd, "Forward") → writes timestamp #3 (top of pipe)
  EndScope(cmd)              → writes timestamp #4 (bottom of pipe)
  ...
EndFrame(cmd)                → writes timestamp #N (bottom of pipe)
```

The scopes are inserted automatically by the render graph — every registered pass gets wrapped:

```cpp
const char* passName = entry.pass->GetName().c_str();
ObjectLabeling::BeginLabel(cmd, passName);
if (profiler) profiler->BeginScope(cmd, frameIndex, passName);
// ... entry.pass->Execute(cmd) ...
if (profiler) profiler->EndScope(cmd, frameIndex);
ObjectLabeling::EndLabel(cmd);
```

Key details:
- `BeginScope` uses `VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT` — timestamp is written as early as possible when work begins.
- `EndScope` uses `VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT` — timestamp is written only after all prior work finishes.
- `nextQuery` is a simple incrementing counter that assigns query indices sequentially.
- `scopeStack` tracks nesting (though in practice the render graph only uses flat scopes).

## Readback (CPU side, next frame)

At the start of the **next** frame (after the fence confirms the GPU finished), `CollectResults` reads back the timestamps:

```cpp
vkGetQueryPoolResults(device, f.queryPool, 0, f.nextQuery, ...);

// Skip query 0 (BeginFrame), then iterate pairs:
for (uint32_t i = 1; i + 1 < f.nextQuery; i += 2) {
    float ms = (timestamps[i + 1] - timestamps[i]) * mTimestampPeriod / 1e6f;
    mResults.push_back({ f.scopeNames[scopeIdx], ms });
}
```

- `VK_QUERY_RESULT_WAIT_BIT` makes the call block until results are ready (safe since the fence already signaled).
- It skips index 0 (the `BeginFrame` timestamp) and walks pairs: `[1,2]` = scope 1 duration, `[3,4]` = scope 2 duration, etc.
- The delta between two timestamps is multiplied by `timestampPeriod` (ns/tick) and divided by 1,000,000 to get **milliseconds**.

## Timeline Visualization

```
Query index:  0        1        2        3        4        5        6
              │        │        │        │        │        │        │
              ▼        ▼        ▼        ▼        ▼        ▼        ▼
GPU timeline: ├─frame──├─Cull───┤────────├Shadow──┤────────├Forward─┤──frame─┤
              BeginFrame  scope0          scope1           scope2   EndFrame
```

Results stored in `mHistory` (up to 300 frames) can be exported as:
- **CSV** (`ExportCSV`) — one row per frame, columns per pass, for spreadsheet analysis.
- **Chrome Tracing JSON** (`ExportChromeTracing`) — open in `chrome://tracing` for a visual timeline view.

## Summary

| Step | Where | What happens |
|------|-------|-------------|
| Init | `Initialize` | Create per-frame query pools, read GPU tick rate |
| Record | `BeginFrame` / `BeginScope` / `EndScope` / `EndFrame` | Insert `vkCmdWriteTimestamp2` into command buffer |
| Auto-wrap | `RenderGraph::Execute` | Every pass automatically gets `BeginScope`/`EndScope` |
| Readback | `CollectResults` (called next frame) | Read timestamp values, compute deltas → milliseconds |
| Display | `DebugUI::DrawProfilerPanel` | Show per-pass GPU times in the ImGui overlay |
