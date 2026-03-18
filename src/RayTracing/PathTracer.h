#pragma once

#include "RayTracing/RTPipeline.h"
#include "RayTracing/ShaderBindingTable.h"
#include "RayTracing/AccelStructure.h"
#include "Resource/VulkanBuffer.h"
#include "Resource/VulkanImage.h"
#include "Resource/ShaderManager.h"
#include "GPU/MeshPool.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <cstdint>

class TransferManager;
class DescriptorManager;

class PathTracer {
public:
    void Initialize(VkDevice device, VmaAllocator allocator,
                    ShaderManager& shaders,
                    const TransferManager& transfer,
                    const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rtProps,
                    uint32_t width, uint32_t height);

    void Shutdown(VkDevice device, VmaAllocator allocator);
    void Resize(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h);

    void UpdateScene(VkDevice device, VmaAllocator allocator,
                     const TransferManager& transfer,
                     VkAccelerationStructureKHR tlas,
                     const MeshPool& meshPool,
                     const std::vector<RTInstanceInfo>& instanceInfos,
                     VkBuffer materialSSBO, VkDeviceSize materialSSBOSize,
                     VkDescriptorSet bindlessTexSet,
                     VkDescriptorSetLayout bindlessTexLayout,
                     VkImageView envCubeView, VkSampler cubeSampler,
                     VkImageView irradianceView,
                     VkImageView brdfLutView, VkSampler lutSampler);

    void Trace(VkCommandBuffer cmd,
               const glm::mat4& invViewProj,
               const glm::mat4& viewProj,
               const glm::vec3& cameraPos,
               const glm::vec3& sunDir,
               const glm::vec3& sunColor,
               float sunIntensity,
               float lightRadius,
               bool denoiserEnabled = false);  // unused, kept for API compatibility

    void ResetAccumulation() { mAccumFrames = 0; mAccumReset = true; }
    bool WasAccumulationReset() const { return mAccumReset; }

    VkImageView GetColorOutputView()  const { return mColorOutput.GetView(); }
    VkImageView GetAccumOutputView()  const { return mAccumBuffer.GetView(); }
    VkImageView GetAlbedoOutputView() const { return mAlbedoOutput.GetView(); }
    VkImageView GetNormalOutputView() const { return mNormalOutput.GetView(); }
    VkImageView GetDepthOutputView()  const { return mDepthOutput.GetView(); }
    VkImageView GetMotionOutputView() const { return mMotionOutput.GetView(); }

    VkImage GetColorOutputImage() const { return mColorOutput.GetImage(); }
    VkImage GetDepthOutputImage() const { return mDepthOutput.GetImage(); }
    VkImage GetNormalOutputImage() const { return mNormalOutput.GetImage(); }
    VkImage GetAccumOutputImage() const { return mAccumBuffer.GetImage(); }

    int      maxBounces   = 8;
    bool     enableMIS    = true;
    bool     progressive  = true;

private:
    void CreateImages(uint32_t w, uint32_t h);
    void CreatePipeline(ShaderManager& shaders);
    void CreateDescriptors();
    void UpdateImageDescriptors();

    VkDevice     mDevice    = VK_NULL_HANDLE;
    VmaAllocator mAllocator = VK_NULL_HANDLE;
    const TransferManager* mTransfer = nullptr;
    ShaderManager* mShaders = nullptr;

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR mRTProps{};

    RTPipeline         mPipeline;
    ShaderBindingTable mSBT;

    VulkanImage mColorOutput;
    VulkanImage mAlbedoOutput;
    VulkanImage mNormalOutput;
    VulkanImage mDepthOutput;
    VulkanImage mMotionOutput;
    VulkanImage mAccumBuffer;

    VulkanBuffer mInstanceInfoBuffer;
    VulkanBuffer mFrameUBO;

    VkDescriptorSetLayout mSceneDescLayout   = VK_NULL_HANDLE;
    VkDescriptorPool      mDescPool          = VK_NULL_HANDLE;
    VkDescriptorSet       mSceneDescSet      = VK_NULL_HANDLE;
    VkPipelineLayout      mPipelineLayout    = VK_NULL_HANDLE;
    VkDescriptorSetLayout mBindlessDescLayout = VK_NULL_HANDLE;
    VkDescriptorSet       mBindlessDescSet    = VK_NULL_HANDLE;

    uint32_t mWidth  = 0, mHeight = 0;
    uint32_t mAccumFrames   = 0;
    bool     mAccumReset    = false;
    uint32_t mSampleOffset  = 0;
    bool     mSceneDirty    = true;

    glm::mat4 mPrevViewProj{1.0f};

    struct PushConstants {
        glm::mat4  invViewProj;
        glm::vec4  cameraPosAndFrame;
        glm::vec4  sunDirAndRadius;
        glm::vec4  sunColorIntensity;
        glm::uvec4 params;
    };
    static_assert(sizeof(PushConstants) == 128, "PathTracer push constants must be 128 bytes");

    struct FrameUBOData {
        glm::mat4 viewProj;
        glm::mat4 prevViewProj;
    };
};
