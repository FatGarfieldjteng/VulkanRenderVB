#pragma once

#include "Core/Window.h"
#include "Core/InputManager.h"
#include "Core/ThreadPool.h"
#include "Core/SubmitThread.h"
#include "RHI/VulkanInstance.h"
#include "RHI/VulkanDevice.h"
#include "RHI/VulkanSwapchain.h"
#include "RHI/VulkanCommandBuffer.h"
#include "RHI/VulkanSync.h"
#include "RHI/VulkanMemory.h"
#include "Resource/TransferManager.h"
#include "Resource/VulkanBuffer.h"
#include "Resource/VulkanImage.h"
#include "Resource/DescriptorManager.h"
#include "Resource/ShaderManager.h"
#include "Resource/PipelineManager.h"
#include "Asset/ModelLoader.h"
#include "Scene/Camera.h"
#include "Scene/Scene.h"
#include "Scene/ECS.h"
#include "Lighting/CascadedShadowMap.h"
#include "IBL/IBLProcessor.h"
#include "ImageCache/ImageCache.h"
#include "RenderGraph/RenderGraph.h"
#include "GPU/MeshPool.h"
#include "GPU/IndirectRenderer.h"
#include "GPU/HiZBuffer.h"
#include "GPU/ComputeCulling.h"
#include "PostProcess/PostProcessStack.h"
#include "VisualUI/DebugUI.h"
#include "VisualUI/GPUProfiler.h"
#include "VisualUI/DebugVisualization.h"
#include "VisualUI/PipelineStatistics.h"
#include "RayTracing/AccelStructure.h"
#include "RayTracing/RTShadows.h"
#include "RayTracing/RTReflections.h"
#include "RayTracing/PathTracer.h"
#include "RayTracing/NRDDenoiser.h"

#include <optional>
#include <string>
#include <vector>

class Application {
public:
    ~Application();
    void Run();
    void RunBenchmark(uint32_t frameCount, bool gpuDriven, bool occlusionCulling);
    void SetScenePath(const std::string& path) { mScenePathOverride = path; }
    void SetInitialRenderMode(DebugUIState::RenderMode mode) { mInitialRenderMode = mode; }
    void SetInitialDenoiser(bool on) { mInitialDenoiser = on; }

private:
    void InitWindow();
    void InitVulkan();
    void CreateDefaultTextures();
    void LoadScene();
    void CreateDepthBuffer();
    void CreateFrameDescriptors();
    void CreatePipelines();
    void MainLoop();
    void DrawFrame();
    void DrawFrameMultiThreaded();
    void BuildAndExecuteRenderGraph(VkCommandBuffer cmd, uint32_t imageIndex);
    void RecreateSwapchain();
    void CleanupVulkan();

    void InitGPUDriven();
    void ShutdownGPUDriven();
    void ExtractFrustumPlanes(const glm::mat4& vp, glm::vec4 planes[6]);

    void ClearScene();
    void LoadTestScene();
    void ReloadScene(SceneType newType);

    void InitDebugUI();
    void ShutdownDebugUI();
    void SyncUIState();
    void LabelVulkanObjects();

    static constexpr uint32_t WINDOW_WIDTH     = 1280;
    static constexpr uint32_t WINDOW_HEIGHT    = 720;
    static constexpr uint32_t FRAMES_IN_FLIGHT = 2;

    // --- core ---
    Window              mWindow;
    InputManager        mInput;
    VulkanInstance      mVulkanInstance;
    VkSurfaceKHR        mSurface = VK_NULL_HANDLE;
    VulkanDevice        mDevice;
    VulkanMemory        mMemory;
    VulkanSwapchain     mSwapchain;
    VulkanSync          mSync;
    VulkanCommandBuffer mCommandBuffers;

    // --- resource managers ---
    TransferManager   mTransfer;
    DescriptorManager mDescriptors;
    ShaderManager     mShaders;
    PipelineManager   mPipelines;

    // --- image cache & render graph ---
    ImageCache  mImageCache;
    RenderGraph mRenderGraph;

    // --- scene (ECS) ---
    Camera           mCamera;
    Registry         mRegistry;
    Entity           mSunEntity = INVALID_ENTITY;
    CascadedShadowMap mCSM;
    IBLProcessor      mIBL;
    float            mLightAzimuth   = 53.0f;
    float            mLightElevation = 58.0f;
    bool             mCSMEnabled     = true;

    // --- depth ---
    VulkanImage mDepthImage;

    // --- GPU scene data ---
    ModelData                    mModelData;
    std::vector<VulkanImage>     mGPUTextures;
    std::vector<uint32_t>        mTextureDescriptorIndices;
    std::vector<GPUMaterialData> mGPUMaterials;

    // --- default textures ---
    VulkanImage mWhiteTexture;
    VulkanImage mBlackTexture;
    VulkanImage mDefaultNormalTexture;
    uint32_t    mWhiteTexDescIdx         = 0;
    uint32_t    mBlackTexDescIdx         = 0;
    uint32_t    mDefaultNormalDescIdx    = 0;

    // --- material SSBO ---
    VulkanBuffer mMaterialSSBO;

    // --- per-frame UBOs ---
    std::vector<VulkanBuffer> mFrameUBOs;

    // --- frame descriptor resources (set 1: 7 bindings) ---
    VkDescriptorSetLayout        mFrameSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool             mFrameDescPool  = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> mFrameDescSets;

    // --- pipelines ---
    VkPipelineLayout mPBRPipelineLayout        = VK_NULL_HANDLE;
    VkPipeline       mPBRPipeline              = VK_NULL_HANDLE;
    VkPipelineLayout mShadowPipelineLayout     = VK_NULL_HANDLE;
    VkPipeline       mShadowPipeline           = VK_NULL_HANDLE;

    // --- GPU-driven rendering (Phase 6) ---
    bool             mGPUDriven = true;
    bool             mOcclusionCulling = true;
    float            mOccluderRatio = 0.2f;
    MeshPool         mMeshPool;
    IndirectRenderer mIndirectRenderer;
    HiZBuffer        mHiZBuffer;
    ComputeCulling   mComputeCulling;

    VkPipelineLayout mPBRIndirectPipelineLayout    = VK_NULL_HANDLE;
    VkPipeline       mPBRIndirectPipeline          = VK_NULL_HANDLE;
    VkPipelineLayout mShadowIndirectPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       mShadowIndirectPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout mDepthPrepassPipelineLayout   = VK_NULL_HANDLE;
    VkPipeline       mDepthPrepassPipeline         = VK_NULL_HANDLE;

    VkDescriptorSetLayout mShadowIndirectDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool      mShadowIndirectDescPool   = VK_NULL_HANDLE;
    VkDescriptorSet       mShadowIndirectDescSet    = VK_NULL_HANDLE;

    // --- Post-processing (Phase 8) ---
    PostProcessStack mPostProcess;

    // --- Ray Tracing (Phase 9) ---
    bool             mRayTracingEnabled = false;
    bool             mRTShadowsEnabled  = true;
    bool             mRTReflEnabled     = true;
    float            mRTShadowStrength  = 1.0f;
    float            mRTReflStrength    = 0.5f;
    float            mRTReflRoughness   = 0.15f;
    float            mRTLightRadius     = 0.02f;
    bool             mRTDebugShadowVis = false;
    AccelStructure   mAccelStructure;
    RTShadows        mRTShadows;
    RTReflections    mRTReflections;


    VkPipeline       mRTCompositePipeline    = VK_NULL_HANDLE;
    VkPipelineLayout mRTCompositePipeLayout  = VK_NULL_HANDLE;
    VkDescriptorSetLayout mRTCompositeDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool mRTCompositeDescPool    = VK_NULL_HANDLE;
    VkDescriptorSet  mRTCompositeDescSet     = VK_NULL_HANDLE;
    VkSampler        mRTDepthSampler         = VK_NULL_HANDLE;
    bool             mRTCompositeDescDirty   = true;

    void InitRayTracing();
    void ShutdownRayTracing();

    // --- Ray Tracing Pipeline (Phase 10) ---
    bool            mRTPipelineSupported = false;
    PathTracer      mPathTracer;
    NRDDenoiser     mNRDDenoiser;

    VkPipeline       mPTCompositePipeline   = VK_NULL_HANDLE;
    VkPipelineLayout mPTCompositePipeLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout mPTCompositeDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool mPTCompositeDescPool   = VK_NULL_HANDLE;
    VkDescriptorSet  mPTCompositeDescSet    = VK_NULL_HANDLE;

    // Denoise composite: uses imageLoad for B10G11R11 (texture() can produce wrong colors)
    VkPipeline       mPTCompositeDenoisePipeline   = VK_NULL_HANDLE;
    VkPipelineLayout mPTCompositeDenoisePipeLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout mPTCompositeDenoiseDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool mPTCompositeDenoiseDescPool   = VK_NULL_HANDLE;
    VkDescriptorSet  mPTCompositeDenoiseDescSet    = VK_NULL_HANDLE;

    // Compare composite: split-screen L=denoised, R=raw (for debugging)
    VkPipeline       mPTCompositeComparePipeline   = VK_NULL_HANDLE;
    VkPipelineLayout mPTCompositeComparePipeLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout mPTCompositeCompareDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool mPTCompositeCompareDescPool   = VK_NULL_HANDLE;
    VkDescriptorSet  mPTCompositeCompareDescSet    = VK_NULL_HANDLE;

    VkSampler        mPTSampler             = VK_NULL_HANDLE;
    bool             mPTCompositeDescDirty  = true;

    // NRD needs prev frame view/proj for temporal reprojection
    glm::mat4        mPTViewMatPrev{1.0f};
    glm::mat4        mPTProjMatPrev{1.0f};
    bool             mPTFirstNRDFrame       = true;

    DebugUIState::RenderMode mActiveRenderMode = DebugUIState::RenderMode::Rasterization;

    void InitRTPipeline();
    void ShutdownRTPipeline();
    void UpdatePTCompositeDescriptors();
    void RebuildRenderGraphForMode(DebugUIState::RenderMode mode);

    // --- MSAA ---
    std::vector<VkSampleCountFlagBits> mSupportedMSAA;
    VkSampleCountFlagBits mCurrentMSAA = VK_SAMPLE_COUNT_1_BIT;
    void RecreatePBRPipelines(VkSampleCountFlagBits samples);

    // --- Visual UI (Phase 7) ---
    DebugUI              mDebugUI;
    GPUProfiler          mGPUProfiler;
    DebugVisualization   mDebugVis;
    PipelineStatistics   mPipelineStats;
    bool                 mShowUI = true;
    float                mDeltaTime = 0.0f;

    // --- multithreaded rendering ---
    bool             mMultiThreading = false;
    ThreadPool       mThreadPool;
    SubmitThread     mSubmitThread;
    std::vector<VkCommandPool>   mWorkerCommandPools;
    std::vector<VkCommandBuffer> mSecondaryCommandBuffers;

    // --- sync ---
    std::vector<VkFence> mImageFences;
    uint32_t mFrameIndex         = 0;
    uint32_t mFrameNumber        = 0;
    bool     mFramebufferResized = false;

    // --- timing ---
    double mLastFrameTime = 0.0;

    // --- scene override ---
    std::string mScenePathOverride;
    SceneType   mCurrentScene = SceneType::TestScene;

    // --- CLI overrides for render mode / denoiser ---
    std::optional<DebugUIState::RenderMode> mInitialRenderMode;
    std::optional<bool> mInitialDenoiser;
};
