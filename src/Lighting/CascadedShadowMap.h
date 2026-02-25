#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <array>

class CascadedShadowMap {
public:
    static constexpr uint32_t CASCADE_COUNT = 4;
    static constexpr uint32_t SHADOW_DIM    = 2048;
    static constexpr float    LAMBDA        = 0.5f; // practical split scheme mix

    void Initialize(VmaAllocator allocator, VkDevice device);
    void Shutdown(VmaAllocator allocator, VkDevice device);

    /// Compute cascade splits and light-space matrices for the current frame.
    void Update(const glm::mat4& cameraView, const glm::mat4& cameraProj,
                float cameraNear, float cameraFar,
                const glm::vec3& lightDir);

    VkImageView GetLayerView(uint32_t cascade) const { return mLayerViews[cascade]; }
    VkImageView GetArrayView()                 const { return mArrayView; }
    VkSampler   GetShadowSampler()             const { return mSampler; }
    VkImage     GetImage()                     const { return mImage; }

    const glm::mat4& GetViewProj(uint32_t cascade) const { return mViewProj[cascade]; }
    const glm::vec4& GetSplits()                    const { return mSplitDepths; }

private:
    VkImage       mImage      = VK_NULL_HANDLE;
    VmaAllocation mAllocation = VK_NULL_HANDLE;
    VkImageView   mArrayView  = VK_NULL_HANDLE;
    std::array<VkImageView, CASCADE_COUNT> mLayerViews{};
    VkSampler     mSampler    = VK_NULL_HANDLE;

    std::array<glm::mat4, CASCADE_COUNT> mViewProj{};
    glm::vec4 mSplitDepths{};
};
