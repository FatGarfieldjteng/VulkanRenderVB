#pragma once

#include "Resource/VulkanImage.h"
#include "Resource/ShaderManager.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <cstdint>

namespace nrd { class Integration; struct CommonSettings; }

class NRDDenoiser {
public:
    NRDDenoiser();
    ~NRDDenoiser();
    void Initialize(VkInstance instance, VkDevice device, VmaAllocator allocator,
                    ShaderManager& shaders,
                    VkPhysicalDevice physicalDevice,
                    VkQueue queue, uint32_t queueFamilyIndex,
                    uint32_t width, uint32_t height);
    void Shutdown(VkDevice device, VmaAllocator allocator);
    void Resize(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h);

    void Denoise(VkCommandBuffer cmd,
                 VkImageView noisyColorView,
                 VkImageView normalView,
                 VkImageView depthView,
                 VkImageView motionView,
                 VkImageView albedoView,
                 VkImage depthImage,
                 VkImage normalImage,
                 const glm::mat4& invViewProj,
                 const glm::mat4& viewProj,
                 const glm::mat4& viewMat,
                 const glm::mat4& projMat,
                 const glm::mat4& viewMatPrev,
                 const glm::mat4& projMatPrev,
                 bool cameraMoved = false);

    VkImageView GetOutputView() const { return mOutput.GetView(); }
    VkImage     GetOutputImage() const { return mOutput.GetImage(); }

    /// Transition output from GENERAL (after NRD write) to SHADER_READ_ONLY_OPTIMAL for sampling
    void TransitionOutputForSampling(VkCommandBuffer cmd) const;

    void InvalidateHistory() { mHistoryValid = false; }

private:
    void CreatePrepackResources(uint32_t w, uint32_t h);
    void CreatePrepackPipeline(ShaderManager& shaders);
    void CreatePrepackDescriptors();
    void UpdatePrepackDescriptors(VkImageView colorView, VkImageView normalView,
                                   VkImageView depthView, VkImageView motionView,
                                   VkImageView albedoView);
    void RunPrepack(VkCommandBuffer cmd);

    void SetupNRDCommonSettings(nrd::CommonSettings& common, uint32_t frameIndex,
                                const glm::mat4& viewMat, const glm::mat4& projMat,
                                const glm::mat4& viewMatPrev, const glm::mat4& projMatPrev);
    void SetupNRDReblurSettings();

    VkInstance         mInstance  = VK_NULL_HANDLE;
    VkDevice           mDevice   = VK_NULL_HANDLE;
    VmaAllocator       mAllocator = VK_NULL_HANDLE;
    VkPhysicalDevice  mPhysicalDevice = VK_NULL_HANDLE;
    VkQueue           mQueue     = VK_NULL_HANDLE;
    uint32_t          mQueueFamilyIndex = 0;
    ShaderManager*    mShaders   = nullptr;
    uint32_t          mWidth = 0, mHeight = 0;
    bool         mHistoryValid = false;
    bool         mDescriptorsDirty = true;

    // NRI device (owned by us when using Vulkan)
    void* mNriDevice = nullptr;

    // NRD integration (pimpl; raw ptr - Shutdown() must be called before destruction)
    nrd::Integration* mNRD = nullptr;
    uint32_t mReblurDiffuseId = 0;

    // Prepack: PT G-buffer -> NRD format
    VulkanImage mPrepackRadianceHitDist;
    VulkanImage mPrepackNormalRoughness;
    VulkanImage mPrepackViewZ;
    VulkanImage mPrepackMotion;

    VkPipeline       mPrepackPipeline   = VK_NULL_HANDLE;
    VkPipelineLayout mPrepackPipeLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout mPrepackDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool mPrepackDescPool  = VK_NULL_HANDLE;
    VkDescriptorSet  mPrepackDescSet   = VK_NULL_HANDLE;

    // NRD output (we use prepack output as input; NRD writes to our output)
    VulkanImage mOutput;

    uint32_t mFrameIndex = 0;
    glm::mat4 mViewProjPrev{1.0f};
};
