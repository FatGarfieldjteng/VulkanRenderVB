#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <cstdint>

class TransferManager;

class IBLProcessor {
public:
    void Initialize(VmaAllocator allocator, VkDevice device,
                    const TransferManager& transfer, VkPipelineCache pipelineCache);

    /// Load an HDR file and bake IBL maps. If hdrPath is null or file not found,
    /// generates a procedural sky environment instead.
    void Process(const char* hdrPath = nullptr);

    void Shutdown(VmaAllocator allocator, VkDevice device);

    VkImageView GetIrradianceView() const { return mIrradianceCubeView; }
    VkImageView GetPrefilterView()  const { return mPrefilterCubeView; }
    VkImageView GetBRDFLutView()    const { return mBRDFLutView; }
    VkSampler   GetCubeSampler()    const { return mCubeSampler; }
    VkSampler   GetLutSampler()     const { return mLutSampler; }
    bool        IsReady()           const { return mReady; }

    static constexpr uint32_t ENV_SIZE            = 512;
    static constexpr uint32_t IRR_SIZE            = 32;
    static constexpr uint32_t PREFILTER_SIZE      = 128;
    static constexpr uint32_t PREFILTER_MIP_LEVELS = 5;
    static constexpr uint32_t BRDF_SIZE           = 512;

private:
    void CreateCubemapImages();
    void CreateSamplers();
    void UploadEquirectangular(const float* pixels, uint32_t w, uint32_t h);
    void GenerateProceduralSky();
    void BakeIBL();

    VmaAllocator            mAllocator     = VK_NULL_HANDLE;
    VkDevice                mDevice        = VK_NULL_HANDLE;
    const TransferManager*  mTransfer      = nullptr;
    VkPipelineCache         mPipelineCache = VK_NULL_HANDLE;

    VkImage       mEnvCubemap         = VK_NULL_HANDLE;
    VmaAllocation mEnvCubemapAlloc    = VK_NULL_HANDLE;
    VkImageView   mEnvCubeView        = VK_NULL_HANDLE;

    VkImage       mIrradianceCubemap      = VK_NULL_HANDLE;
    VmaAllocation mIrradianceCubemapAlloc  = VK_NULL_HANDLE;
    VkImageView   mIrradianceCubeView     = VK_NULL_HANDLE;

    VkImage       mPrefilterCubemap       = VK_NULL_HANDLE;
    VmaAllocation mPrefilterCubemapAlloc   = VK_NULL_HANDLE;
    VkImageView   mPrefilterCubeView      = VK_NULL_HANDLE;

    VkImage       mBRDFLut          = VK_NULL_HANDLE;
    VmaAllocation mBRDFLutAlloc     = VK_NULL_HANDLE;
    VkImageView   mBRDFLutView      = VK_NULL_HANDLE;

    VkImage       mEquirectImage    = VK_NULL_HANDLE;
    VmaAllocation mEquirectAlloc    = VK_NULL_HANDLE;
    VkImageView   mEquirectView     = VK_NULL_HANDLE;

    VkSampler     mCubeSampler      = VK_NULL_HANDLE;
    VkSampler     mLutSampler       = VK_NULL_HANDLE;

    bool mReady = false;
};
