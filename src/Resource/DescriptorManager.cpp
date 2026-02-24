#include "Resource/DescriptorManager.h"
#include "Core/Logger.h"

void DescriptorManager::Initialize(VkDevice device) {
    mUsed.resize(MAX_TEXTURES, false);
    mNextFree = 0;

    // --- default sampler (linear, repeat, with anisotropy) ---
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter               = VK_FILTER_LINEAR;
    samplerInfo.minFilter               = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable        = VK_TRUE;
    samplerInfo.maxAnisotropy           = 16.0f;
    samplerInfo.maxLod                  = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &mDefaultSampler));

    // --- descriptor set layout (bindless sampled images) ---
    VkDescriptorSetLayoutBinding binding{};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = MAX_TEXTURES;
    binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorBindingFlags bindingFlags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
    flagsInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flagsInfo.bindingCount  = 1;
    flagsInfo.pBindingFlags = &bindingFlags;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext        = &flagsInfo;
    layoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings    = &binding;

    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &mLayout));

    // --- descriptor pool ---
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = MAX_TEXTURES;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets       = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;

    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &mPool));

    // --- allocate the single bindless set ---
    uint32_t variableCount = MAX_TEXTURES;
    VkDescriptorSetVariableDescriptorCountAllocateInfo variableInfo{};
    variableInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
    variableInfo.descriptorSetCount = 1;
    variableInfo.pDescriptorCounts  = &variableCount;

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.pNext              = &variableInfo;
    allocInfo.descriptorPool     = mPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &mLayout;

    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &mSet));

    LOG_INFO("DescriptorManager initialized (max {} textures)", MAX_TEXTURES);
}

void DescriptorManager::Shutdown(VkDevice device) {
    if (mDefaultSampler) vkDestroySampler(device, mDefaultSampler, nullptr);
    if (mPool)           vkDestroyDescriptorPool(device, mPool, nullptr);
    if (mLayout)         vkDestroyDescriptorSetLayout(device, mLayout, nullptr);
    mDefaultSampler = VK_NULL_HANDLE;
    mPool           = VK_NULL_HANDLE;
    mLayout         = VK_NULL_HANDLE;
    mSet            = VK_NULL_HANDLE;
    LOG_INFO("DescriptorManager destroyed");
}

uint32_t DescriptorManager::AllocateTextureIndex() {
    for (uint32_t i = mNextFree; i < MAX_TEXTURES; i++) {
        if (!mUsed[i]) {
            mUsed[i]  = true;
            mNextFree = i + 1;
            return i;
        }
    }
    for (uint32_t i = 0; i < mNextFree && i < MAX_TEXTURES; i++) {
        if (!mUsed[i]) {
            mUsed[i]  = true;
            mNextFree = i + 1;
            return i;
        }
    }
    LOG_ERROR("DescriptorManager: no free texture index");
    return UINT32_MAX;
}

void DescriptorManager::FreeTextureIndex(uint32_t index) {
    if (index < MAX_TEXTURES) {
        mUsed[index] = false;
        if (index < mNextFree) mNextFree = index;
    }
}

void DescriptorManager::UpdateTexture(VkDevice device, uint32_t index,
                                      VkImageView view, VkSampler sampler)
{
    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler     = sampler;
    imageInfo.imageView   = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = mSet;
    write.dstBinding      = 0;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo      = &imageInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}
