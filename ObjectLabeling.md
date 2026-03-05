# Object Labeling

## Overview

`ObjectLabeling` is a thin wrapper around Vulkan's **debug utils** extension (`VK_EXT_debug_utils`). It serves two purposes:

## 1. Naming Vulkan Objects

`SetName` and the inline helpers (`NameBuffer`, `NameImage`, `NamePipeline`, etc.) assign human-readable names to Vulkan handles via `vkSetDebugUtilsObjectNameEXT`. These names show up in:
- **Validation layer error messages** — instead of seeing "VkDescriptorSet 0x9b800000009b8", you see the name you assigned.
- **GPU debugging tools** like RenderDoc, Nsight Graphics, etc. — objects appear with their labels in the resource inspector.

## 2. Command Buffer Labels (`BeginLabel` / `EndLabel`)

They insert **colored region markers** into the command buffer using `vkCmdBeginDebugUtilsLabelEXT` / `vkCmdEndDebugUtilsLabelEXT`. These markers are completely invisible to the GPU and don't affect rendering. Their sole purpose is to show up in **external GPU capture tools** like RenderDoc or Nsight Graphics.

For example, when you capture a frame in RenderDoc, instead of seeing a flat list of hundreds of raw Vulkan commands (draw calls, barriers, dispatches), you see them grouped under named, collapsible regions:

```
▼ FrustumCull
    vkCmdBindPipeline
    vkCmdDispatch
▼ Shadow
    vkCmdBeginRendering
    vkCmdDrawIndexedIndirect
    vkCmdEndRendering
▼ Forward
    vkCmdBeginRendering
    vkCmdDraw × 103
    vkCmdEndRendering
```

Each region gets a label name (the pass name) and a color (the RGBA values passed to `BeginLabel`, defaulting to a light blue). This makes it much easier to navigate a frame capture and find the specific pass you want to inspect.

## ScopedLabel

The `ScopedLabel` struct is a RAII convenience — it calls `BeginLabel` in its constructor and `EndLabel` in its destructor, so you can scope a label to a C++ block without worrying about forgetting `EndLabel`.

## Runtime Cost

Purely a **debugging/tooling aid** with zero runtime cost in release builds. The functions early-out if the debug extension isn't loaded (`if (!vkCmdBeginDebugUtilsLabelEXT) return;`). It doesn't affect rendering behavior at all.
