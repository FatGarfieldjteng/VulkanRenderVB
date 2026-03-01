#include "RenderGraph/BarrierBatcher.h"

void BarrierBatcher::Reset(uint32_t imageResourceCount) {
    mImageStates.assign(imageResourceCount, {VK_IMAGE_LAYOUT_UNDEFINED,
                                              VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0});
    mPendingImageBarriers.clear();
    mPendingBufferBarriers.clear();
}

void BarrierBatcher::SetInitialState(uint32_t resourceIdx, VkImageLayout layout,
                                     VkPipelineStageFlags2 stage, VkAccessFlags2 access) {
    if (resourceIdx < mImageStates.size()) {
        mImageStates[resourceIdx] = {layout, stage, access};
    }
}

void BarrierBatcher::TransitionImage(uint32_t resourceIdx,
                                     VkImage image, VkImageAspectFlags aspect,
                                     uint32_t arrayLayers,
                                     VkImageLayout newLayout,
                                     VkPipelineStageFlags2 dstStage,
                                     VkAccessFlags2 dstAccess)
{
    if (resourceIdx >= mImageStates.size()) return;

    auto& s = mImageStates[resourceIdx];
    bool layoutChange = (s.layout != newLayout);
    bool hazard       = (s.access & kWriteBits) || (dstAccess & kWriteBits);

    if (!layoutChange && !hazard) {
        s.stage  |= dstStage;
        s.access |= dstAccess;
        return;
    }

    VkImageMemoryBarrier2 b{};
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask        = s.stage;
    b.srcAccessMask       = s.access;
    b.dstStageMask        = dstStage;
    b.dstAccessMask       = dstAccess;
    b.oldLayout           = s.layout;
    b.newLayout           = newLayout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = image;
    b.subresourceRange    = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, arrayLayers};
    mPendingImageBarriers.push_back(b);

    s.layout = newLayout;
    s.stage  = dstStage;
    s.access = dstAccess;
}

void BarrierBatcher::AddBufferBarrier(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                                      VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                                      VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess)
{
    VkBufferMemoryBarrier2 b{};
    b.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    b.srcStageMask        = srcStage;
    b.srcAccessMask       = srcAccess;
    b.dstStageMask        = dstStage;
    b.dstAccessMask       = dstAccess;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer              = buffer;
    b.offset              = offset;
    b.size                = size;
    mPendingBufferBarriers.push_back(b);
}

void BarrierBatcher::Flush(VkCommandBuffer cmd) {
    if (mPendingImageBarriers.empty() && mPendingBufferBarriers.empty())
        return;

    VkDependencyInfo dep{};
    dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount  = static_cast<uint32_t>(mPendingImageBarriers.size());
    dep.pImageMemoryBarriers     = mPendingImageBarriers.data();
    dep.bufferMemoryBarrierCount = static_cast<uint32_t>(mPendingBufferBarriers.size());
    dep.pBufferMemoryBarriers    = mPendingBufferBarriers.data();
    vkCmdPipelineBarrier2(cmd, &dep);

    mPendingImageBarriers.clear();
    mPendingBufferBarriers.clear();
}

bool BarrierBatcher::HasPendingBarriers() const {
    return !mPendingImageBarriers.empty() || !mPendingBufferBarriers.empty();
}
