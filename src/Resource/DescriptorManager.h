#pragma once

#include <volk.h>
#include <vector>

class DescriptorManager {
public:
    static constexpr uint32_t MAX_TEXTURES = 16384;

    void Initialize(VkDevice device);
    void Shutdown(VkDevice device);

    uint32_t AllocateTextureIndex();
    void     FreeTextureIndex(uint32_t index);

    void UpdateTexture(VkDevice device, uint32_t index,
                       VkImageView view, VkSampler sampler);

    VkDescriptorSetLayout GetLayout()         const { return mLayout; }
    VkDescriptorSet       GetSet()            const { return mSet; }
    VkSampler             GetDefaultSampler() const { return mDefaultSampler; }

private:
    VkDescriptorSetLayout mLayout         = VK_NULL_HANDLE;
    VkDescriptorPool      mPool           = VK_NULL_HANDLE;
    VkDescriptorSet       mSet            = VK_NULL_HANDLE;
    VkSampler             mDefaultSampler = VK_NULL_HANDLE;
    std::vector<bool>     mUsed;
    uint32_t              mNextFree = 0;
};
