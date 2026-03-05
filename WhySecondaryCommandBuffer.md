# Why Use Secondary Command Buffers?

## The Problem

In Vulkan, a `VkCommandBuffer` is **not thread-safe**. You cannot have two threads recording commands into the same command buffer simultaneously. This is stated explicitly in the Vulkan spec.

So if you want 4 threads to record draw calls in parallel, you need 4 separate command buffers. But then how do you get all those draw calls into a single submission?

## Vulkan's Two Levels of Command Buffers

Vulkan provides two levels:

| | Primary | Secondary |
|---|---|---|
| Allocated from | `VK_COMMAND_BUFFER_LEVEL_PRIMARY` | `VK_COMMAND_BUFFER_LEVEL_SECONDARY` |
| Can be submitted to a queue | Yes, via `vkQueueSubmit` | No -- cannot be submitted directly |
| Can call other command buffers | Yes, via `vkCmdExecuteCommands` | No |
| Can begin/end rendering | Yes | Yes, with inheritance |

The key operation is `vkCmdExecuteCommands` -- a primary command buffer can **embed** one or more secondary command buffers. When the GPU executes the primary, it inlines the secondary buffers' commands at that point.

## How It Works for Multithreaded Recording

```
Main thread                    Worker 1              Worker 2              Worker 3
-----------                    --------              --------              --------
Begin primary cmd
Record barriers
vkCmdBeginRendering
                               Begin secondary       Begin secondary       Begin secondary
                               Draw meshes 0-99      Draw meshes 100-199   Draw meshes 200-299
                               End secondary         End secondary         End secondary
                    <-- WaitAll() -----------------------------------------------------------
vkCmdExecuteCommands(sec1, sec2, sec3)
vkCmdEndRendering
Record present transition
End primary cmd
vkQueueSubmit(primary)
```

The GPU sees it as one continuous stream of commands: barriers -> begin rendering -> draws 0-99 -> draws 100-199 -> draws 200-299 -> end rendering -> present transition. The secondary buffers are transparent to the GPU.

## Why Not Just Use Multiple Primary Buffers?

You might think: "Why not have each thread record a primary command buffer, then submit all of them?" There are two reasons:

### 1. Rendering state doesn't carry across primary buffers

With dynamic rendering (`vkCmdBeginRendering` / `vkCmdEndRendering`), the rendering scope (which attachments are bound, the render area, etc.) exists only within one primary command buffer. You can't begin rendering in primary A and record draws in primary B -- primary B has no rendering context.

Secondary command buffers solve this with **inheritance**. When you begin a secondary buffer, you pass a `VkCommandBufferInheritanceRenderingInfo` that tells it "you'll be executed inside a rendering scope with these color formats, this depth format, etc.":

```cpp
VkCommandBufferInheritanceRenderingInfo inheritRendering{};
inheritRendering.sType                   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO;
inheritRendering.colorAttachmentCount    = 1;
inheritRendering.pColorAttachmentFormats = &colorFormat;
inheritRendering.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

VkCommandBufferInheritanceInfo inheritance{};
inheritance.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
inheritance.pNext = &inheritRendering;

VkCommandBufferBeginInfo beginInfo{};
beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
beginInfo.pInheritanceInfo = &inheritance;

vkBeginCommandBuffer(secondaryCmd, &beginInfo);
// Now you can record vkCmdDraw*, vkCmdBindPipeline, etc. here
// even though vkCmdBeginRendering was called in the primary
```

The `RENDER_PASS_CONTINUE_BIT` flag means "I will be executed inside an active rendering scope." Without this, the secondary buffer would have no rendering context and draw calls would be invalid.

### 2. Multiple primary submits have overhead and synchronization complexity

Each `vkQueueSubmit` is an expensive driver call. Submitting 4 primary buffers means 4 submits, each with potential driver-level serialization. With secondaries, you get one submit containing everything.

Also, if you wanted primary B to start after primary A's rendering, you'd need semaphores between them. Secondaries avoid this entirely -- they're inlined sequentially within the primary.

## Summary

| Approach | Thread-safe recording | Single submit | Shared rendering scope | Complexity |
|---|---|---|---|---|
| One primary, single thread | N/A | Yes | Yes | Simple |
| Multiple primaries, multi thread | Yes | No (N submits) | No (need semaphores) | High |
| One primary + N secondaries | Yes | Yes | Yes (via inheritance) | Moderate |

Secondary command buffers are Vulkan's designed mechanism for "record commands on multiple threads, then assemble them into one submission." They exist specifically for this use case.
