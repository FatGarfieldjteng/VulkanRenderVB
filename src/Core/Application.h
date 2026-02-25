#pragma once

#include "Core/Window.h"
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
#include "Lighting/CascadedShadowMap.h"

#include <vector>

struct GPUMesh {
    VulkanBuffer vertexBuffer;
    VulkanBuffer indexBuffer;
    uint32_t     indexCount = 0;
};

class Application {
public:
    void Run();

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
    void RecordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void RecreateSwapchain();
    void CleanupVulkan();

    static constexpr uint32_t WINDOW_WIDTH     = 1280;
    static constexpr uint32_t WINDOW_HEIGHT    = 720;
    static constexpr uint32_t FRAMES_IN_FLIGHT = 2;

    // --- core ---
    Window              mWindow;
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

    // --- scene ---
    Camera            mCamera;
    Scene             mScene;
    CascadedShadowMap mCSM;

    // --- depth ---
    VulkanImage mDepthImage;

    // --- GPU scene data ---
    ModelData                    mModelData;
    std::vector<GPUMesh>         mGPUMeshes;
    std::vector<VulkanImage>     mGPUTextures;
    std::vector<uint32_t>        mTextureDescriptorIndices;
    std::vector<GPUMaterialData> mGPUMaterials;

    // --- default textures ---
    VulkanImage mWhiteTexture;
    VulkanImage mBlackTexture;
    uint32_t    mWhiteTexDescIdx = 0;
    uint32_t    mBlackTexDescIdx = 0;

    // --- material SSBO ---
    VulkanBuffer mMaterialSSBO;

    // --- per-frame UBOs ---
    std::vector<VulkanBuffer> mFrameUBOs;

    // --- frame descriptor resources (set 1) ---
    VkDescriptorSetLayout        mFrameSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool             mFrameDescPool  = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> mFrameDescSets;

    // --- pipelines ---
    VkPipelineLayout mPBRPipelineLayout    = VK_NULL_HANDLE;
    VkPipeline       mPBRPipeline          = VK_NULL_HANDLE;
    VkPipelineLayout mShadowPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       mShadowPipeline       = VK_NULL_HANDLE;

    // --- sync ---
    std::vector<VkFence> mImageFences;
    uint32_t mFrameIndex         = 0;
    bool     mFramebufferResized = false;

    // --- timing ---
    double mLastFrameTime = 0.0;
};
