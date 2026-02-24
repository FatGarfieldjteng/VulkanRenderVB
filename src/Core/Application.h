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

#include <vector>

struct GPUMesh {
    VulkanBuffer vertexBuffer;
    VulkanBuffer indexBuffer;
    uint32_t     indexCount     = 0;
    uint32_t     textureIndex  = 0; // bindless descriptor index
    int          materialIndex = -1;
};

class Application {
public:
    void Run();

private:
    void InitWindow();
    void InitVulkan();
    void LoadScene();
    void CreateDepthBuffer();
    void CreatePipeline();
    void MainLoop();
    void DrawFrame();
    void RecordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void RecreateSwapchain();
    void CleanupVulkan();

    static constexpr uint32_t WINDOW_WIDTH  = 1280;
    static constexpr uint32_t WINDOW_HEIGHT = 720;

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

    // --- depth ---
    VulkanImage mDepthImage;

    // --- scene ---
    ModelData              mModelData;
    std::vector<GPUMesh>   mGPUMeshes;
    std::vector<VulkanImage> mGPUTextures;
    std::vector<uint32_t>  mTextureDescriptorIndices;

    // --- pipeline ---
    VkPipelineLayout mPipelineLayout   = VK_NULL_HANDLE;
    VkPipeline       mGraphicsPipeline = VK_NULL_HANDLE;

    // --- sync ---
    std::vector<VkFence> mImageFences;
    uint32_t mFrameIndex         = 0;
    bool     mFramebufferResized = false;

    float mRotationAngle = 0.0f;
};
