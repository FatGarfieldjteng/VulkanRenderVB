#pragma once

#include <volk.h>
#include <vector>
#include <cstdint>

/// Tracks per-resource layout/stage/access state across a frame and
/// accumulates image and buffer barriers, emitting them in minimal
/// batched vkCmdPipelineBarrier2 calls.
class BarrierBatcher {
public:
    struct ImageState {
        VkImageLayout         layout;
        VkPipelineStageFlags2 stage;
        VkAccessFlags2        access;
    };

    /// Prepare for a new frame with the given number of image resources.
    void Reset(uint32_t imageResourceCount);

    /// Set the initial state for a resource at the start of the frame.
    void SetInitialState(uint32_t resourceIdx, VkImageLayout layout,
                         VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         VkAccessFlags2 access = 0);

    /// Request a transition for an image resource. Determines whether a
    /// barrier is needed (layout change or write hazard) and, if so,
    /// appends a VkImageMemoryBarrier2 to the pending list.
    void TransitionImage(uint32_t resourceIdx,
                         VkImage image, VkImageAspectFlags aspect, uint32_t arrayLayers,
                         VkImageLayout newLayout,
                         VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

    /// Add an explicit buffer memory barrier.
    void AddBufferBarrier(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                          VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                          VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

    /// Emit all pending barriers via a single vkCmdPipelineBarrier2 call,
    /// then clear the pending lists.
    void Flush(VkCommandBuffer cmd);

    bool HasPendingBarriers() const;

    const ImageState& GetState(uint32_t resourceIdx) const { return mImageStates[resourceIdx]; }

private:
    static constexpr VkAccessFlags2 kWriteBits =
        VK_ACCESS_2_SHADER_WRITE_BIT |
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_2_TRANSFER_WRITE_BIT |
        VK_ACCESS_2_HOST_WRITE_BIT |
        VK_ACCESS_2_MEMORY_WRITE_BIT;

    std::vector<ImageState>              mImageStates;
    std::vector<VkImageMemoryBarrier2>   mPendingImageBarriers;
    std::vector<VkBufferMemoryBarrier2>  mPendingBufferBarriers;
};
