# Why `Application::mImageFences` Is Needed

## Background

The engine uses **2 frames-in-flight** and **3 swapchain images**. Each frame-in-flight slot has its own fence (`fence[0]`, `fence[1]`), and each swapchain image has its own command buffer (`cmdBuf[0]`, `cmdBuf[1]`, `cmdBuf[2]`).

At the top of `DrawFrame()`, we wait on the fence for the current frame-in-flight slot:

```cpp
auto frameFence = mSync.GetFence(mFrameIndex);     // mFrameIndex = 0 or 1
vkWaitForFences(device, 1, &frameFence, VK_TRUE, UINT64_MAX);
```

This guarantees that the **previous use of this slot** (from 2 frames ago) is complete. But it does NOT guarantee that the **previous use of a particular swapchain image** is complete — because that image might have been submitted under the other slot's fence.

## The Problem

`vkAcquireNextImageKHR` returns an `imageIndex` chosen by the driver. The driver can return images in any order. Consider this timeline:

```
Frame 0: slot=0, image=0, submit cmdBuf[0] with fence[0]
Frame 1: slot=1, image=1, submit cmdBuf[1] with fence[1]
Frame 2: slot=0, wait fence[0] → frame 0 done ✓, image=1 ← driver returns image 1!
```

At frame 2:
- We waited on `fence[0]`, confirming frame 0 is done.
- The driver returned `imageIndex=1`, so we want to record into `cmdBuf[1]`.
- `cmdBuf[1]` was last submitted in frame 1 with `fence[1]`.
- We did **NOT** wait on `fence[1]` — frame 1 might still be executing on the GPU.
- Without `mImageFences`, we would overwrite `cmdBuf[1]` while the GPU is still running it.

## The Solution

`mImageFences` is an array of size 3 (one per swapchain image). Each entry records which fence was last submitted with that image's command buffer:

```cpp
// After acquiring imageIndex, before recording:
if (mImageFences[imageIndex] != VK_NULL_HANDLE && mImageFences[imageIndex] != frameFence)
    vkWaitForFences(device, 1, &mImageFences[imageIndex], VK_TRUE, UINT64_MAX);
mImageFences[imageIndex] = frameFence;
```

In the scenario above:
- `mImageFences[1] = fence[1]` (set during frame 1)
- `frameFence = fence[0]` (current slot)
- `fence[1] != fence[0]`, so we wait on `fence[1]`
- This guarantees frame 1's GPU work is done before we reuse `cmdBuf[1]`

## When Does This Trigger?

With FIFO present mode and 3 images, images typically cycle in order (0, 1, 2, 0, 1, 2...), and this check rarely activates. But it can happen:

- During swapchain recreation or window resize
- With MAILBOX present mode (images returned sooner)
- Under driver-specific scheduling decisions
- On certain platforms or GPU vendors

## What Each Sync Mechanism Protects

| Mechanism | Protects | Guarantees |
|-----------|----------|------------|
| `fence[mFrameIndex]` wait | Same frame-in-flight slot | Previous use of this slot (N-2) is done |
| `mImageFences[imageIndex]` wait | Same swapchain image | Previous command buffer for this image is done |
| Pipeline barrier | Same GPU resource across submissions | Previous stage/access is complete before next begins |

## Conclusion

`mImageFences` is a safety net for the case where the driver returns a swapchain image that was most recently used by a **different** frame-in-flight slot whose fence has not yet been waited on. Without it, the command buffer for that image could be overwritten while the GPU is still executing it, leading to undefined behavior or GPU crashes.
