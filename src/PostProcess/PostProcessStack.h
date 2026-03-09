#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <cstdint>
#include <memory>

class ShaderManager;
class RenderGraph;

class AutoExposure;
class SSAO;
class Bloom;
class ToneMapping;
class ColorGrading;

struct PostProcessSettings {
    bool autoExposureEnabled = true;
    bool ssaoEnabled         = true;
    bool bloomEnabled        = true;
    bool colorGradingEnabled = true;

    float bloomStrength = 0.04f;

    // Auto exposure
    float exposureMinEV   = -10.0f;
    float exposureMaxEV   = 20.0f;
    float adaptSpeed      = 1.5f;

    // SSAO
    float ssaoRadius    = 30.0f;
    float ssaoBias      = 0.05f;
    float ssaoIntensity = 1.5f;

    // Tone mapping
    enum class ToneCurve : uint32_t { ACES = 0, AgX = 1 };
    ToneCurve toneCurve    = ToneCurve::ACES;
    float exposureBias     = 0.0f;

    // ACES params
    float acesShoulderStrength = 2.51f;
    float acesLinearStrength   = 0.03f;
    float acesLinearAngle      = 2.43f;
    float acesToeStrength      = 0.59f;
    float acesWhitePoint       = 11.2f;

    // AgX params
    float agxSaturation = 1.0f;
    float agxPunch      = 0.0f;

    // Color grading
    float lutStrength          = 0.0f;
    float vignetteIntensity    = 0.3f;
    float vignetteRadius       = 0.5f;
    float grainStrength        = 0.02f;
    float chromaticAberration  = 0.0f;
};

class PostProcessStack {
public:
    PostProcessStack();
    ~PostProcessStack();

    void Initialize(VkDevice device, VmaAllocator allocator, ShaderManager& shaders,
                    VkFormat swapchainFormat, uint32_t width, uint32_t height);
    void Shutdown(VkDevice device, VmaAllocator allocator);
    void Resize(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height);

    void RegisterPasses(RenderGraph& graph, uint32_t swapRes, uint32_t depthRes,
                        uint32_t hdrRes, uint32_t forwardPassH, float deltaTime);

    VkImage     GetHDRImage()     const { return mHDRImage; }
    VkImageView GetHDRView()      const { return mHDRView; }
    VkFormat    GetHDRFormat()     const { return VK_FORMAT_R16G16B16A16_SFLOAT; }

    VkImage     GetLDRImage()     const { return mLDRImage; }
    VkImageView GetLDRView()      const { return mLDRView; }

    VkImageView GetWhitePlaceholderView() const { return mWhitePlaceholderView; }
    VkImageView GetBlackPlaceholderView() const { return mBlackPlaceholderView; }

    void TransitionPlaceholders(VkCommandBuffer cmd);

    void SetMSAASampleCount(VkDevice device, VmaAllocator allocator, VkSampleCountFlagBits samples);
    VkSampleCountFlagBits GetMSAASampleCount() const { return mMSAASamples; }
    VkImage     GetMSAAColorImage() const { return mMSAAColorImage; }
    VkImageView GetMSAAColorView()  const { return mMSAAColorView; }
    VkImage     GetMSAADepthImage() const { return mMSAADepthImage; }
    VkImageView GetMSAADepthView()  const { return mMSAADepthView; }

    PostProcessSettings& GetSettings() { return mSettings; }
    const PostProcessSettings& GetSettings() const { return mSettings; }

    AutoExposure* GetAutoExposure() { return mAutoExposure.get(); }
    SSAO*         GetSSAO()         { return mSSAO.get(); }
    Bloom*        GetBloom()        { return mBloom.get(); }
    ToneMapping*  GetToneMapping()  { return mToneMapping.get(); }
    ColorGrading* GetColorGrading() { return mColorGrading.get(); }

private:
    void CreateHDRImage(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height);
    void DestroyHDRImage(VkDevice device, VmaAllocator allocator);
    void CreateLDRImage(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height, VkFormat fmt);
    void DestroyLDRImage(VkDevice device, VmaAllocator allocator);
    void CreatePlaceholders(VkDevice device, VmaAllocator allocator);
    void DestroyPlaceholders(VkDevice device, VmaAllocator allocator);
    void CreateMSAAImages(VkDevice device, VmaAllocator allocator);
    void DestroyMSAAImages(VkDevice device, VmaAllocator allocator);

    PostProcessSettings mSettings;

    VkImage       mHDRImage  = VK_NULL_HANDLE;
    VkImageView   mHDRView   = VK_NULL_HANDLE;
    VmaAllocation mHDRAlloc  = VK_NULL_HANDLE;

    VkImage       mLDRImage  = VK_NULL_HANDLE;
    VkImageView   mLDRView   = VK_NULL_HANDLE;
    VmaAllocation mLDRAlloc  = VK_NULL_HANDLE;
    VkFormat      mSwapFormat = VK_FORMAT_B8G8R8A8_SRGB;

    VkImage       mWhitePlaceholder     = VK_NULL_HANDLE;
    VkImageView   mWhitePlaceholderView = VK_NULL_HANDLE;
    VmaAllocation mWhitePlaceholderAlloc = VK_NULL_HANDLE;
    VkImage       mBlackPlaceholder     = VK_NULL_HANDLE;
    VkImageView   mBlackPlaceholderView = VK_NULL_HANDLE;
    VmaAllocation mBlackPlaceholderAlloc = VK_NULL_HANDLE;

    bool          mPlaceholdersReady = false;
    uint32_t      mWidth     = 0;
    uint32_t      mHeight    = 0;

    VkSampleCountFlagBits mMSAASamples     = VK_SAMPLE_COUNT_1_BIT;
    VkImage       mMSAAColorImage  = VK_NULL_HANDLE;
    VkImageView   mMSAAColorView   = VK_NULL_HANDLE;
    VmaAllocation mMSAAColorAlloc  = VK_NULL_HANDLE;
    VkImage       mMSAADepthImage  = VK_NULL_HANDLE;
    VkImageView   mMSAADepthView   = VK_NULL_HANDLE;
    VmaAllocation mMSAADepthAlloc  = VK_NULL_HANDLE;

    std::unique_ptr<AutoExposure> mAutoExposure;
    std::unique_ptr<SSAO>         mSSAO;
    std::unique_ptr<Bloom>        mBloom;
    std::unique_ptr<ToneMapping>  mToneMapping;
    std::unique_ptr<ColorGrading> mColorGrading;
};
