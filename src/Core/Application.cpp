#include "Core/Application.h"
#include "Core/Logger.h"
#include "RHI/VulkanUtils.h"
#include "RenderGraph/Passes/ShadowPass.h"
#include "RenderGraph/Passes/ForwardPass.h"
#include "RenderGraph/Passes/PresentPass.h"
#include "RenderGraph/Passes/FrustumCullPass.h"
#include "RenderGraph/Passes/OccluderDepthPass.h"
#include "RenderGraph/Passes/HiZBuildPass.h"
#include "RenderGraph/Passes/OcclusionTestPass.h"
#include "RenderGraph/Passes/PostProcessPass.h"
#include "RenderGraph/Passes/RayTracingPass.h"
#include "GPU/MeshPool.h"
#include "GPU/IndirectRenderer.h"
#include "GPU/HiZBuffer.h"
#include "GPU/ComputeCulling.h"
#include "VisualUI/DebugUI.h"
#include "VisualUI/ImGuiPass.h"
#include "VisualUI/GPUProfiler.h"
#include "VisualUI/ObjectLabeling.h"
#include "VisualUI/DebugVisualization.h"
#include "VisualUI/PipelineStatistics.h"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <cstring>
#include <algorithm>

#include <filesystem>
#include <array>
#include <memory>

// =======================================================================
void Application::Run() {
    InitWindow();
    InitVulkan();
    MainLoop();
    CleanupVulkan();
}

void Application::RunBenchmark(uint32_t frameCount, bool gpuDriven, bool occlusionCulling) {
    InitWindow();
    InitVulkan();

    mGPUDriven        = gpuDriven;
    mOcclusionCulling  = occlusionCulling;
    mShowUI            = false;

    constexpr uint32_t kWarmup = 30;
    uint32_t totalFrames = kWarmup + frameCount;

    double benchStart = 0.0;
    double totalCpu   = 0.0;
    float  minMs = 1e9f, maxMs = 0.0f;

    std::printf("=== BENCHMARK: %u frames, GPU-driven=%d, occlusion=%d ===\n",
                frameCount, gpuDriven, occlusionCulling);

    for (uint32_t i = 0; i < totalFrames && !mWindow.ShouldClose(); i++) {
        mWindow.PollEvents();
        double now = glfwGetTime();
        float dt   = static_cast<float>(now - mLastFrameTime);
        mLastFrameTime = now;
        dt = std::min(dt, 0.1f);
        mDeltaTime = dt;

        mInput.Update(mWindow);
        mWindow.ResetInputDeltas();
        mRegistry.UpdateTransforms();

        if (i == kWarmup) benchStart = glfwGetTime();

        DrawFrame();

        if (i >= kWarmup) {
            double frameEnd = glfwGetTime();
            float frameMs = static_cast<float>((frameEnd - now) * 1000.0);
            totalCpu += frameMs;
            minMs = std::min(minMs, frameMs);
            maxMs = std::max(maxMs, frameMs);
        }
    }

    mDevice.WaitIdle();

    double benchEnd   = glfwGetTime();
    double wallTimeS  = benchEnd - benchStart;
    float  avgMs      = static_cast<float>(totalCpu / frameCount);
    float  avgFps     = static_cast<float>(frameCount / wallTimeS);

    const char* modeName = gpuDriven
        ? (occlusionCulling ? "GPU-Driven + Occlusion" : "GPU-Driven (no occlusion)")
        : "CPU Draw Calls";

    std::printf("=== BENCHMARK RESULTS ===\n");
    std::printf("  Mode:         %s\n", modeName);
    std::printf("  Frames:       %u\n", frameCount);
    std::printf("  Wall time:    %.2f s\n", wallTimeS);
    std::printf("  Avg FPS:      %.1f\n", avgFps);
    std::printf("  Avg frame:    %.3f ms\n", avgMs);
    std::printf("  Min frame:    %.3f ms\n", minMs);
    std::printf("  Max frame:    %.3f ms\n", maxMs);

    mGPUProfiler.CollectResults(mDevice.GetHandle(), mFrameIndex);
    const auto& gpuResults = mGPUProfiler.GetResults();
    if (!gpuResults.empty()) {
        std::printf("  GPU total:    %.3f ms\n", mGPUProfiler.GetTotalMs());
        for (const auto& r : gpuResults)
            std::printf("    %s: %.3f ms\n", r.name.c_str(), r.durationMs);
    }
    std::printf("=========================\n");
    std::fflush(stdout);

    CleanupVulkan();
}

// =======================================================================
// Init
// =======================================================================
void Application::InitWindow() {
    Logger::Initialize();
    mWindow.Initialize(WINDOW_WIDTH, WINDOW_HEIGHT, "VulkanRenderVB");
    mWindow.SetResizeCallback([this](uint32_t, uint32_t) {
        mFramebufferResized = true;
    });
    mInput.Initialize();
}

void Application::InitVulkan() {
    mVulkanInstance.Initialize("VulkanRenderVB");
    mSurface = mVulkanInstance.CreateSurface(mWindow.GetHandle());
    mDevice.Initialize(mVulkanInstance.GetHandle(), mSurface);
    mMemory.Initialize(mVulkanInstance.GetHandle(), mDevice.GetPhysicalDevice(), mDevice.GetHandle());
    mSwapchain.Initialize(mDevice.GetHandle(), mDevice.GetPhysicalDevice(),
                          mSurface, mWindow.GetHandle(), mDevice.GetQueueFamilyIndices());
    mSync.Initialize(mDevice.GetHandle(), FRAMES_IN_FLIGHT, mSwapchain.GetImageCount());
    mCommandBuffers.Initialize(mDevice.GetHandle(), mDevice.GetQueueFamilyIndices().graphicsFamily,
                               mSwapchain.GetImageCount());
    mImageFences.resize(mSwapchain.GetImageCount(), VK_NULL_HANDLE);

    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(mDevice.GetPhysicalDevice(), &props);
        VkSampleCountFlags counts = props.limits.framebufferColorSampleCounts
                                  & props.limits.framebufferDepthSampleCounts;
        mSupportedMSAA.push_back(VK_SAMPLE_COUNT_1_BIT);
        if (counts & VK_SAMPLE_COUNT_2_BIT) mSupportedMSAA.push_back(VK_SAMPLE_COUNT_2_BIT);
        if (counts & VK_SAMPLE_COUNT_4_BIT) mSupportedMSAA.push_back(VK_SAMPLE_COUNT_4_BIT);
        if (counts & VK_SAMPLE_COUNT_8_BIT) mSupportedMSAA.push_back(VK_SAMPLE_COUNT_8_BIT);
        LOG_INFO("Supported MSAA up to: {}x", static_cast<int>(mSupportedMSAA.back()));
    }

    mTransfer.Initialize(mDevice.GetHandle(), mDevice.GetQueueFamilyIndices().graphicsFamily,
                         mDevice.GetGraphicsQueue());
    mDescriptors.Initialize(mDevice.GetHandle());
    mShaders.Initialize(mDevice.GetHandle());
    mPipelines.Initialize(mDevice.GetHandle());
    mPipelines.LoadCache("pipeline_cache.bin");

    mCSM.Initialize(mMemory.GetAllocator(), mDevice.GetHandle());

    mIBL.Initialize(mMemory.GetAllocator(), mDevice.GetHandle(), mTransfer, mPipelines.GetCache());
    mIBL.Process();

    mImageCache.Initialize(mDevice.GetHandle(), mMemory.GetAllocator());
    mRenderGraph.Initialize(mDevice.GetHandle(), &mImageCache);

    CreateDefaultTextures();
    if (mCurrentScene == SceneType::TestScene)
        LoadTestScene();
    else
        LoadScene();
    CreateDepthBuffer();
    CreateFrameDescriptors();
    CreatePipelines();

    InitGPUDriven();
    InitRayTracing();

    {
        auto extent = mSwapchain.GetExtent();
        mPostProcess.Initialize(mDevice.GetHandle(), mMemory.GetAllocator(), mShaders,
                                mSwapchain.GetImageFormat(), extent.width, extent.height);
    }

    InitDebugUI();

    mModelData = ModelData{};

    if (mMultiThreading) {
        mThreadPool.Initialize();
        mSubmitThread.Initialize(mDevice.GetGraphicsQueue(), mDevice.GetPresentQueue());

        uint32_t workerCount = mThreadPool.GetThreadCount();
        mWorkerCommandPools.resize(workerCount);
        mSecondaryCommandBuffers.resize(workerCount);
        for (uint32_t i = 0; i < workerCount; i++) {
            VkCommandPoolCreateInfo poolCI{};
            poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolCI.queueFamilyIndex = mDevice.GetQueueFamilyIndices().graphicsFamily;
            VK_CHECK(vkCreateCommandPool(mDevice.GetHandle(), &poolCI, nullptr, &mWorkerCommandPools[i]));

            VkCommandBufferAllocateInfo allocCI{};
            allocCI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocCI.commandPool        = mWorkerCommandPools[i];
            allocCI.level              = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
            allocCI.commandBufferCount = 1;
            VK_CHECK(vkAllocateCommandBuffers(mDevice.GetHandle(), &allocCI, &mSecondaryCommandBuffers[i]));
        }
    }

    mCamera.Init(glm::vec3(0, 1.6f, 0), glm::vec3(0, 1.6f, -1.0f), 45.0f, 0.01f, 100.0f);
    mLastFrameTime = glfwGetTime();
    mInput.LoadBindings("input_bindings.cfg");

    LOG_INFO("Vulkan initialization complete (Phase 7 - Debug Tools & Profiling)");
}

// =======================================================================
// Default textures
// =======================================================================
void Application::CreateDefaultTextures() {
    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();

    uint8_t whitePixels[4] = {255, 255, 255, 255};
    mWhiteTexture.CreateTexture2D(allocator, device, mTransfer, 1, 1,
                                  VK_FORMAT_R8G8B8A8_UNORM, whitePixels);
    mWhiteTexDescIdx = mDescriptors.AllocateTextureIndex();
    mDescriptors.UpdateTexture(device, mWhiteTexDescIdx, mWhiteTexture.GetView(),
                               mDescriptors.GetDefaultSampler());

    uint8_t blackPixels[4] = {0, 0, 0, 255};
    mBlackTexture.CreateTexture2D(allocator, device, mTransfer, 1, 1,
                                  VK_FORMAT_R8G8B8A8_UNORM, blackPixels);
    mBlackTexDescIdx = mDescriptors.AllocateTextureIndex();
    mDescriptors.UpdateTexture(device, mBlackTexDescIdx, mBlackTexture.GetView(),
                               mDescriptors.GetDefaultSampler());

    uint8_t normalPixels[4] = {128, 128, 255, 255};
    mDefaultNormalTexture.CreateTexture2D(allocator, device, mTransfer, 1, 1,
                                          VK_FORMAT_R8G8B8A8_UNORM, normalPixels);
    mDefaultNormalDescIdx = mDescriptors.AllocateTextureIndex();
    mDescriptors.UpdateTexture(device, mDefaultNormalDescIdx, mDefaultNormalTexture.GetView(),
                               mDescriptors.GetDefaultSampler());

    LOG_INFO("Default textures created (white={}, black={}, normal={})",
             mWhiteTexDescIdx, mBlackTexDescIdx, mDefaultNormalDescIdx);
}

// =======================================================================
// Scene loading (ECS-based)
// =======================================================================
void Application::LoadScene() {
    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();

    mSunEntity = mRegistry.CreateEntity();
    mRegistry.AddTransform(mSunEntity);
    auto& sunLight = mRegistry.AddLight(mSunEntity);
    sunLight.direction = glm::normalize(glm::vec3(-0.4f, -0.8f, -0.3f));
    sunLight.color     = glm::vec3(1.0f, 0.95f, 0.85f);
    sunLight.intensity = 3.5f;

    bool loaded = false;

    if (!mScenePathOverride.empty()) {
        if (std::filesystem::exists(mScenePathOverride)) {
            loaded = ModelLoader::LoadGLTF(mScenePathOverride.c_str(), mModelData);
            if (loaded)
                LOG_INFO("Loaded glTF model (override): {}", mScenePathOverride);
            else
                LOG_ERROR("Failed to load override scene: {}", mScenePathOverride);
        } else {
            LOG_ERROR("Override scene path does not exist: {}", mScenePathOverride);
        }
    }

    if (!loaded) {
        const char* modelPaths[] = {
            "assets/Sponza/Sponza.gltf",
            "assets/Sponza.glb",
            "assets/Bistro/Bistro.gltf",
            "assets/Bistro/bistro.gltf",
            "assets/DamagedHelmet.glb",
            "assets/DamagedHelmet/DamagedHelmet.gltf",
            "assets/model.glb",
        };
        for (const char* p : modelPaths) {
            if (std::filesystem::exists(p)) {
                loaded = ModelLoader::LoadGLTF(p, mModelData);
                if (loaded) {
                    LOG_INFO("Loaded glTF model: {}", p);
                    break;
                }
            }
        }
    }

    if (loaded) {
        ModelLoader::SortMeshesByVolume(mModelData.meshes);

        std::vector<bool> isLinear(mModelData.textures.size(), false);
        for (const auto& mat : mModelData.materials) {
            if (mat.metallicRoughnessTextureIndex >= 0)
                isLinear[mat.metallicRoughnessTextureIndex] = true;
            if (mat.normalTextureIndex >= 0)
                isLinear[mat.normalTextureIndex] = true;
            if (mat.occlusionTextureIndex >= 0)
                isLinear[mat.occlusionTextureIndex] = true;
        }

        for (size_t i = 0; i < mModelData.textures.size(); i++) {
            const auto& texData = mModelData.textures[i];
            VkFormat fmt = isLinear[i] ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R8G8B8A8_SRGB;
            VulkanImage gpuTex;
            gpuTex.CreateTexture2D(allocator, device, mTransfer,
                                   texData.width, texData.height, fmt, texData.pixels.data());
            uint32_t descIdx = mDescriptors.AllocateTextureIndex();
            mDescriptors.UpdateTexture(device, descIdx, gpuTex.GetView(),
                                       mDescriptors.GetDefaultSampler());
            mGPUTextures.push_back(std::move(gpuTex));
            mTextureDescriptorIndices.push_back(descIdx);
        }

        auto resolveIdx = [&](int texIdx, uint32_t fallback) -> uint32_t {
            if (texIdx >= 0 && texIdx < static_cast<int>(mTextureDescriptorIndices.size()))
                return mTextureDescriptorIndices[texIdx];
            return fallback;
        };

        for (const auto& mat : mModelData.materials) {
            GPUMaterialData g{};
            g.baseColorFactor         = mat.baseColorFactor;
            g.metallicFactor          = mat.metallicFactor;
            g.roughnessFactor         = mat.roughnessFactor;
            g.baseColorTexIdx         = resolveIdx(mat.baseColorTextureIndex, mWhiteTexDescIdx);
            g.normalTexIdx            = resolveIdx(mat.normalTextureIndex, mDefaultNormalDescIdx);
            g.metallicRoughnessTexIdx = resolveIdx(mat.metallicRoughnessTextureIndex, mWhiteTexDescIdx);
            g.aoTexIdx                = resolveIdx(mat.occlusionTextureIndex, mWhiteTexDescIdx);
            g.emissiveTexIdx          = resolveIdx(mat.emissiveTextureIndex, mBlackTexDescIdx);
            mGPUMaterials.push_back(g);
        }

        for (const auto& inst : mModelData.instances) {
            if (inst.meshIndex < 0 || inst.meshIndex >= static_cast<int>(mModelData.meshes.size()))
                continue;
            Entity e = mRegistry.CreateEntity();
            auto& tc = mRegistry.AddTransform(e);
            tc.localPosition = inst.translation;
            tc.localRotation = inst.rotation;
            tc.localScale    = inst.scale;
            mRegistry.AddMesh(e).meshIndex = inst.meshIndex;
            mRegistry.AddMaterial(e).materialIndex = std::max(0, mModelData.meshes[inst.meshIndex].materialIndex);
        }

    } else {
        LOG_INFO("No glTF model found, generating procedural scene");

        constexpr uint32_t texW = 512, texH = 512, tileSize = 32;
        std::vector<uint8_t> checkerPixels(texW * texH * 4);
        for (uint32_t y = 0; y < texH; y++) {
            for (uint32_t x = 0; x < texW; x++) {
                bool white = ((x / tileSize) + (y / tileSize)) % 2 == 0;
                uint8_t c  = white ? 200 : 80;
                uint32_t idx = (y * texW + x) * 4;
                checkerPixels[idx + 0] = c;
                checkerPixels[idx + 1] = c;
                checkerPixels[idx + 2] = c;
                checkerPixels[idx + 3] = 255;
            }
        }
        VulkanImage checkerImg;
        checkerImg.CreateTexture2D(allocator, device, mTransfer,
                                   texW, texH, VK_FORMAT_R8G8B8A8_SRGB, checkerPixels.data());
        uint32_t checkerDescIdx = mDescriptors.AllocateTextureIndex();
        mDescriptors.UpdateTexture(device, checkerDescIdx, checkerImg.GetView(),
                                   mDescriptors.GetDefaultSampler());
        mGPUTextures.push_back(std::move(checkerImg));
        mTextureDescriptorIndices.push_back(checkerDescIdx);

        {
            MeshData groundMesh;
            ModelLoader::GenerateGroundPlane(groundMesh, 20.0f);
            mModelData.meshes.push_back(std::move(groundMesh));
        }
        {
            ModelData cubeData;
            ModelLoader::GenerateProceduralCube(cubeData);
            mModelData.meshes.push_back(std::move(cubeData.meshes[0]));
        }

        auto pushMat = [&](glm::vec3 color, float metallic, float roughness, uint32_t baseTexIdx) {
            GPUMaterialData m{};
            m.baseColorFactor         = glm::vec4(color, 1.0f);
            m.metallicFactor          = metallic;
            m.roughnessFactor         = roughness;
            m.baseColorTexIdx         = baseTexIdx;
            m.normalTexIdx            = mDefaultNormalDescIdx;
            m.metallicRoughnessTexIdx = mWhiteTexDescIdx;
            m.aoTexIdx                = mWhiteTexDescIdx;
            m.emissiveTexIdx          = mBlackTexDescIdx;
            mGPUMaterials.push_back(m);
        };

        pushMat({0.5f, 0.5f, 0.5f}, 0.0f, 0.9f, checkerDescIdx);
        pushMat({0.8f, 0.15f, 0.15f}, 0.0f, 0.3f, mWhiteTexDescIdx);
        pushMat({0.1f, 0.7f, 0.1f}, 0.9f, 0.15f, mWhiteTexDescIdx);
        pushMat({0.15f, 0.15f, 0.8f}, 0.0f, 0.7f, mWhiteTexDescIdx);
        pushMat({1.0f, 0.766f, 0.336f}, 1.0f, 0.1f, mWhiteTexDescIdx);

        auto makeObj = [&](int mesh, int mat, glm::vec3 pos) {
            Entity e = mRegistry.CreateEntity();
            auto& tc = mRegistry.AddTransform(e);
            tc.localPosition = pos;
            mRegistry.AddMesh(e).meshIndex = mesh;
            mRegistry.AddMaterial(e).materialIndex = mat;
        };
        makeObj(0, 0, {0, 0, 0});
        makeObj(1, 1, {-3.0f, 0.5f, 0.0f});
        makeObj(1, 2, {-1.0f, 0.5f, -1.0f});
        makeObj(1, 3, {1.0f, 0.5f, 0.0f});
        makeObj(1, 4, {3.0f, 0.5f, 1.0f});
    }

    if (mGPUMaterials.empty()) {
        GPUMaterialData def{};
        def.baseColorTexIdx         = mWhiteTexDescIdx;
        def.normalTexIdx            = mDefaultNormalDescIdx;
        def.metallicRoughnessTexIdx = mWhiteTexDescIdx;
        def.aoTexIdx                = mWhiteTexDescIdx;
        def.emissiveTexIdx          = mBlackTexDescIdx;
        mGPUMaterials.push_back(def);
    }

    // Demo scene objects for RT (mirror sphere + glossy floor plane)
    if (mDevice.IsRayTracingSupported()) {
        uint32_t sphereMeshIdx = static_cast<uint32_t>(mModelData.meshes.size());
        MeshData sphereMesh;
        ModelLoader::GenerateUVSphere(sphereMesh, 0.8f, 64, 32);
        mModelData.meshes.push_back(std::move(sphereMesh));

        uint32_t glossyFloorIdx = static_cast<uint32_t>(mModelData.meshes.size());
        MeshData floorMesh;
        ModelLoader::GenerateGroundPlane(floorMesh, 3.0f);
        mModelData.meshes.push_back(std::move(floorMesh));

        // Mirror material (metallic = 1, roughness ~0)
        uint32_t mirrorMatIdx = static_cast<uint32_t>(mGPUMaterials.size());
        {
            GPUMaterialData m{};
            m.baseColorFactor         = glm::vec4(0.95f, 0.95f, 0.97f, 1.0f);
            m.metallicFactor          = 1.0f;
            m.roughnessFactor         = 0.02f;
            m.baseColorTexIdx         = mWhiteTexDescIdx;
            m.normalTexIdx            = mDefaultNormalDescIdx;
            m.metallicRoughnessTexIdx = mWhiteTexDescIdx;
            m.aoTexIdx                = mWhiteTexDescIdx;
            m.emissiveTexIdx          = mBlackTexDescIdx;
            mGPUMaterials.push_back(m);
        }

        // Glossy floor material
        uint32_t glossyMatIdx = static_cast<uint32_t>(mGPUMaterials.size());
        {
            GPUMaterialData m{};
            m.baseColorFactor         = glm::vec4(0.7f, 0.7f, 0.72f, 1.0f);
            m.metallicFactor          = 0.3f;
            m.roughnessFactor         = 0.15f;
            m.baseColorTexIdx         = mWhiteTexDescIdx;
            m.normalTexIdx            = mDefaultNormalDescIdx;
            m.metallicRoughnessTexIdx = mWhiteTexDescIdx;
            m.aoTexIdx                = mWhiteTexDescIdx;
            m.emissiveTexIdx          = mBlackTexDescIdx;
            mGPUMaterials.push_back(m);
        }

        // Place mirror sphere in the Sponza courtyard
        {
            Entity e = mRegistry.CreateEntity();
            auto& tc = mRegistry.AddTransform(e);
            tc.localPosition = glm::vec3(0.0f, 1.2f, 0.0f);
            tc.localScale    = glm::vec3(1.0f);
            mRegistry.AddMesh(e).meshIndex = static_cast<int>(sphereMeshIdx);
            mRegistry.AddMaterial(e).materialIndex = static_cast<int>(mirrorMatIdx);
        }

        // Place glossy floor plane
        {
            Entity e = mRegistry.CreateEntity();
            auto& tc = mRegistry.AddTransform(e);
            tc.localPosition = glm::vec3(0.0f, 0.01f, 0.0f);
            mRegistry.AddMesh(e).meshIndex = static_cast<int>(glossyFloorIdx);
            mRegistry.AddMaterial(e).materialIndex = static_cast<int>(glossyMatIdx);
        }

        LOG_INFO("RT demo objects added: mirror sphere + glossy floor");
    }

    mMaterialSSBO.CreateDeviceLocal(allocator, mTransfer,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        mGPUMaterials.data(), mGPUMaterials.size() * sizeof(GPUMaterialData));

    if (mDevice.IsRayTracingSupported()) {
        mMeshPool.Upload(allocator, mTransfer, mModelData.meshes,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    } else {
        mMeshPool.Upload(allocator, mTransfer, mModelData.meshes);
    }

    LOG_INFO("Scene loaded: {} meshes, {} textures, {} materials, {} entities",
             mMeshPool.GetMeshCount(), mGPUTextures.size(), mGPUMaterials.size(),
             mRegistry.EntityCount());
}

// =======================================================================
// Depth buffer
// =======================================================================
void Application::CreateDepthBuffer() {
    auto extent = mSwapchain.GetExtent();
    mDepthImage.CreateDepth(mMemory.GetAllocator(), mDevice.GetHandle(),
                            extent.width, extent.height);
}

// =======================================================================
// Frame descriptors (set 1: 7 bindings -- UBO + mat SSBO + shadow + IBL + object SSBO)
// =======================================================================
void Application::CreateFrameDescriptors() {
    auto device     = mDevice.GetHandle();
    uint32_t frames = FRAMES_IN_FLIGHT;

    VkDescriptorSetLayoutBinding bindings[7]{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[4] = {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[5] = {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[6] = {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                   VK_SHADER_STAGE_VERTEX_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = 7;
    layoutCI.pBindings    = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &mFrameSetLayout));

    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         frames },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         frames * 2 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, frames * 4 },
    };
    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets       = frames;
    poolCI.poolSizeCount = 3;
    poolCI.pPoolSizes    = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(device, &poolCI, nullptr, &mFrameDescPool));

    mFrameDescSets.resize(frames);
    mFrameUBOs.resize(frames);

    for (uint32_t i = 0; i < frames; i++) {
        VkDescriptorSetAllocateInfo allocCI{};
        allocCI.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocCI.descriptorPool     = mFrameDescPool;
        allocCI.descriptorSetCount = 1;
        allocCI.pSetLayouts        = &mFrameSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &allocCI, &mFrameDescSets[i]));

        mFrameUBOs[i].CreateHostVisible(mMemory.GetAllocator(),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        sizeof(FrameData));

        VkDescriptorBufferInfo uboInfo{mFrameUBOs[i].GetHandle(), 0, sizeof(FrameData)};
        VkDescriptorBufferInfo matInfo{mMaterialSSBO.GetHandle(), 0, mMaterialSSBO.GetSize()};

        VkDescriptorImageInfo shadowInfo{mCSM.GetShadowSampler(), mCSM.GetArrayView(),
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo irrInfo{mIBL.GetCubeSampler(), mIBL.GetIrradianceView(),
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo prefInfo{mIBL.GetCubeSampler(), mIBL.GetPrefilterView(),
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo brdfInfo{mIBL.GetLutSampler(), mIBL.GetBRDFLutView(),
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

        VkWriteDescriptorSet writes[6]{};
        for (int w = 0; w < 6; w++) {
            writes[w].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[w].dstSet          = mFrameDescSets[i];
            writes[w].dstBinding      = static_cast<uint32_t>(w);
            writes[w].descriptorCount = 1;
        }
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo    = &uboInfo;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo    = &matInfo;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo     = &shadowInfo;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].pImageInfo     = &irrInfo;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[4].pImageInfo     = &prefInfo;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[5].pImageInfo     = &brdfInfo;

        vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
    }

    LOG_INFO("Frame descriptors created ({} sets, 7 bindings each)", frames);
}

// =======================================================================
// Pipelines (PBR + Shadow)
// =======================================================================
void Application::CreatePipelines() {
    auto device = mDevice.GetHandle();

    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding   = 0;
    bindingDesc.stride    = sizeof(MeshVertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDescs[4]{};
    attrDescs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(MeshVertex, position)};
    attrDescs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(MeshVertex, normal)};
    attrDescs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(MeshVertex, texCoord)};
    attrDescs[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(MeshVertex, tangent)};

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = 4;
    vertexInput.pVertexAttributeDescriptions    = attrDescs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates    = dynStates;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // -------- PBR pipeline --------
    {
        VkShaderModule vert = mShaders.GetOrLoad("shaders/pbr.vert.spv");
        VkShaderModule frag = mShaders.GetOrLoad("shaders/pbr.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode    = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.lineWidth   = 1.0f;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable  = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState blendAtt{};
        blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlend{};
        colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlend.attachmentCount = 1;
        colorBlend.pAttachments    = &blendAtt;

        VkDescriptorSetLayout setLayouts[] = { mDescriptors.GetLayout(), mFrameSetLayout };
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset     = 0;
        pushRange.size       = static_cast<uint32_t>(sizeof(glm::mat4) + sizeof(uint32_t));

        VkPipelineLayoutCreateInfo layoutCI{};
        layoutCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCI.setLayoutCount         = 2;
        layoutCI.pSetLayouts            = setLayouts;
        layoutCI.pushConstantRangeCount = 1;
        layoutCI.pPushConstantRanges    = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(device, &layoutCI, nullptr, &mPBRPipelineLayout));

        VkFormat hdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        VkPipelineRenderingCreateInfo renderInfo{};
        renderInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderInfo.colorAttachmentCount    = 1;
        renderInfo.pColorAttachmentFormats = &hdrFormat;
        renderInfo.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

        VkGraphicsPipelineCreateInfo ci{};
        ci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        ci.pNext               = &renderInfo;
        ci.stageCount          = 2;
        ci.pStages             = stages;
        ci.pVertexInputState   = &vertexInput;
        ci.pInputAssemblyState = &inputAssembly;
        ci.pViewportState      = &viewportState;
        ci.pRasterizationState = &rasterizer;
        ci.pMultisampleState   = &multisampling;
        ci.pDepthStencilState  = &depthStencil;
        ci.pColorBlendState    = &colorBlend;
        ci.pDynamicState       = &dynamicState;
        ci.layout              = mPBRPipelineLayout;

        VK_CHECK(vkCreateGraphicsPipelines(device, mPipelines.GetCache(), 1, &ci, nullptr, &mPBRPipeline));
    }

    // -------- Shadow pipeline --------
    {
        VkShaderModule vert = mShaders.GetOrLoad("shaders/shadow.vert.spv");
        VkShaderModule frag = mShaders.GetOrLoad("shaders/shadow.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode                = VK_CULL_MODE_FRONT_BIT;
        rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.lineWidth               = 1.0f;
        rasterizer.depthBiasEnable         = VK_TRUE;
        rasterizer.depthBiasConstantFactor = 1.25f;
        rasterizer.depthBiasSlopeFactor    = 1.75f;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable  = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendStateCreateInfo colorBlend{};
        colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlend.attachmentCount = 0;

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushRange.offset     = 0;
        pushRange.size       = sizeof(glm::mat4);

        VkPipelineLayoutCreateInfo layoutCI{};
        layoutCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCI.pushConstantRangeCount = 1;
        layoutCI.pPushConstantRanges    = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(device, &layoutCI, nullptr, &mShadowPipelineLayout));

        VkPipelineRenderingCreateInfo renderInfo{};
        renderInfo.sType                = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

        VkGraphicsPipelineCreateInfo ci{};
        ci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        ci.pNext               = &renderInfo;
        ci.stageCount          = 2;
        ci.pStages             = stages;
        ci.pVertexInputState   = &vertexInput;
        ci.pInputAssemblyState = &inputAssembly;
        ci.pViewportState      = &viewportState;
        ci.pRasterizationState = &rasterizer;
        ci.pMultisampleState   = &multisampling;
        ci.pDepthStencilState  = &depthStencil;
        ci.pColorBlendState    = &colorBlend;
        ci.pDynamicState       = &dynamicState;
        ci.layout              = mShadowPipelineLayout;

        VK_CHECK(vkCreateGraphicsPipelines(device, mPipelines.GetCache(), 1, &ci, nullptr, &mShadowPipeline));
    }

    // -------- PBR indirect pipeline --------
    {
        VkShaderModule vert = mShaders.GetOrLoad("shaders/pbr_indirect.vert.spv");
        VkShaderModule frag = mShaders.GetOrLoad("shaders/pbr_indirect.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode    = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.lineWidth   = 1.0f;

        VkPipelineDepthStencilStateCreateInfo depthStencilI{};
        depthStencilI.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencilI.depthTestEnable  = VK_TRUE;
        depthStencilI.depthWriteEnable = VK_TRUE;
        depthStencilI.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState blendAttI{};
        blendAttI.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlendI{};
        colorBlendI.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendI.attachmentCount = 1;
        colorBlendI.pAttachments    = &blendAttI;

        VkDescriptorSetLayout setLayoutsI[] = { mDescriptors.GetLayout(), mFrameSetLayout };

        VkPipelineLayoutCreateInfo layoutCII{};
        layoutCII.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCII.setLayoutCount = 2;
        layoutCII.pSetLayouts    = setLayoutsI;
        VK_CHECK(vkCreatePipelineLayout(device, &layoutCII, nullptr, &mPBRIndirectPipelineLayout));

        VkFormat hdrFormatI = VK_FORMAT_R16G16B16A16_SFLOAT;
        VkPipelineRenderingCreateInfo renderInfoI{};
        renderInfoI.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderInfoI.colorAttachmentCount    = 1;
        renderInfoI.pColorAttachmentFormats = &hdrFormatI;
        renderInfoI.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

        VkGraphicsPipelineCreateInfo ciI{};
        ciI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        ciI.pNext               = &renderInfoI;
        ciI.stageCount          = 2;
        ciI.pStages             = stages;
        ciI.pVertexInputState   = &vertexInput;
        ciI.pInputAssemblyState = &inputAssembly;
        ciI.pViewportState      = &viewportState;
        ciI.pRasterizationState = &rasterizer;
        ciI.pMultisampleState   = &multisampling;
        ciI.pDepthStencilState  = &depthStencilI;
        ciI.pColorBlendState    = &colorBlendI;
        ciI.pDynamicState       = &dynamicState;
        ciI.layout              = mPBRIndirectPipelineLayout;

        VK_CHECK(vkCreateGraphicsPipelines(device, mPipelines.GetCache(), 1, &ciI, nullptr, &mPBRIndirectPipeline));
    }

    // -------- Depth pre-pass pipeline (for occluder draw) --------
    {
        VkShaderModule vert = mShaders.GetOrLoad("shaders/pbr_indirect.vert.spv");
        VkShaderModule frag = mShaders.GetOrLoad("shaders/depth_prepass.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode    = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.lineWidth   = 1.0f;

        VkPipelineDepthStencilStateCreateInfo depthStencilDP{};
        depthStencilDP.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencilDP.depthTestEnable  = VK_TRUE;
        depthStencilDP.depthWriteEnable = VK_TRUE;
        depthStencilDP.depthCompareOp   = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendStateCreateInfo colorBlendDP{};
        colorBlendDP.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendDP.attachmentCount = 0;

        VkDescriptorSetLayout setLayoutsDP[] = { mDescriptors.GetLayout(), mFrameSetLayout };
        VkPipelineLayoutCreateInfo layoutCIDP{};
        layoutCIDP.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCIDP.setLayoutCount = 2;
        layoutCIDP.pSetLayouts    = setLayoutsDP;
        VK_CHECK(vkCreatePipelineLayout(device, &layoutCIDP, nullptr, &mDepthPrepassPipelineLayout));

        VkPipelineRenderingCreateInfo renderInfoDP{};
        renderInfoDP.sType                = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderInfoDP.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

        VkGraphicsPipelineCreateInfo ciDP{};
        ciDP.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        ciDP.pNext               = &renderInfoDP;
        ciDP.stageCount          = 2;
        ciDP.pStages             = stages;
        ciDP.pVertexInputState   = &vertexInput;
        ciDP.pInputAssemblyState = &inputAssembly;
        ciDP.pViewportState      = &viewportState;
        ciDP.pRasterizationState = &rasterizer;
        ciDP.pMultisampleState   = &multisampling;
        ciDP.pDepthStencilState  = &depthStencilDP;
        ciDP.pColorBlendState    = &colorBlendDP;
        ciDP.pDynamicState       = &dynamicState;
        ciDP.layout              = mDepthPrepassPipelineLayout;

        VK_CHECK(vkCreateGraphicsPipelines(device, mPipelines.GetCache(), 1, &ciDP, nullptr, &mDepthPrepassPipeline));
    }

    // -------- Shadow indirect pipeline --------
    {
        VkShaderModule vert = mShaders.GetOrLoad("shaders/shadow_indirect.vert.spv");
        VkShaderModule frag = mShaders.GetOrLoad("shaders/shadow.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode                = VK_CULL_MODE_FRONT_BIT;
        rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.lineWidth               = 1.0f;
        rasterizer.depthBiasEnable         = VK_TRUE;
        rasterizer.depthBiasConstantFactor = 1.25f;
        rasterizer.depthBiasSlopeFactor    = 1.75f;

        VkPipelineDepthStencilStateCreateInfo depthStencilS{};
        depthStencilS.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencilS.depthTestEnable  = VK_TRUE;
        depthStencilS.depthWriteEnable = VK_TRUE;
        depthStencilS.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendStateCreateInfo colorBlendS{};
        colorBlendS.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendS.attachmentCount = 0;

        VkDescriptorSetLayoutBinding shadowIndBinding{};
        shadowIndBinding.binding         = 0;
        shadowIndBinding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        shadowIndBinding.descriptorCount = 1;
        shadowIndBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo shadowIndLayoutCI{};
        shadowIndLayoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        shadowIndLayoutCI.bindingCount = 1;
        shadowIndLayoutCI.pBindings    = &shadowIndBinding;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &shadowIndLayoutCI, nullptr, &mShadowIndirectDescLayout));

        VkPushConstantRange pushRangeS{};
        pushRangeS.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushRangeS.offset     = 0;
        pushRangeS.size       = sizeof(glm::mat4);

        VkPipelineLayoutCreateInfo layoutCIS{};
        layoutCIS.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCIS.setLayoutCount         = 1;
        layoutCIS.pSetLayouts            = &mShadowIndirectDescLayout;
        layoutCIS.pushConstantRangeCount = 1;
        layoutCIS.pPushConstantRanges    = &pushRangeS;
        VK_CHECK(vkCreatePipelineLayout(device, &layoutCIS, nullptr, &mShadowIndirectPipelineLayout));

        VkPipelineRenderingCreateInfo renderInfoS{};
        renderInfoS.sType                = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderInfoS.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

        VkGraphicsPipelineCreateInfo ciS{};
        ciS.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        ciS.pNext               = &renderInfoS;
        ciS.stageCount          = 2;
        ciS.pStages             = stages;
        ciS.pVertexInputState   = &vertexInput;
        ciS.pInputAssemblyState = &inputAssembly;
        ciS.pViewportState      = &viewportState;
        ciS.pRasterizationState = &rasterizer;
        ciS.pMultisampleState   = &multisampling;
        ciS.pDepthStencilState  = &depthStencilS;
        ciS.pColorBlendState    = &colorBlendS;
        ciS.pDynamicState       = &dynamicState;
        ciS.layout              = mShadowIndirectPipelineLayout;

        VK_CHECK(vkCreateGraphicsPipelines(device, mPipelines.GetCache(), 1, &ciS, nullptr, &mShadowIndirectPipeline));
    }

    LOG_INFO("Pipelines created (PBR + Shadow + Indirect variants)");
}

// =======================================================================
// Main loop
// =======================================================================
void Application::MainLoop() {
    LOG_INFO("Entering main loop (Phase 7 - Debug Tools & Profiling)");
    while (!mWindow.ShouldClose()) {
        mWindow.PollEvents();

        double now = glfwGetTime();
        float dt   = static_cast<float>(now - mLastFrameTime);
        mLastFrameTime = now;
        dt = std::min(dt, 0.1f);
        mDeltaTime = dt;

        mInput.Update(mWindow);

        if (mInput.WasPressed(InputManager::Action::ToggleUI))
            mShowUI = !mShowUI;

        bool uiWantsMouse = mDebugUI.WantCaptureMouse();

        if (!uiWantsMouse)
            mCamera.Update(mInput, dt);

        mWindow.ResetInputDeltas();
        mRegistry.UpdateTransforms();

        SyncUIState();

        if (mShowUI) {
            mDebugUI.BeginFrame();
            mDebugUI.BuildUI(dt, &mGPUProfiler, &mPipelineStats, &mRegistry, &mGPUMaterials,
                           &mPostProcess.GetSettings(), &mSupportedMSAA);
            mDebugUI.EndFrame();
        }

        DrawFrame();
    }
    mDevice.WaitIdle();
}

// =======================================================================
// Draw frame
// =======================================================================
void Application::DrawFrame() {
    auto device    = mDevice.GetHandle();
    auto frameFence = mSync.GetFence(mFrameIndex);

    vkWaitForFences(device, 1, &frameFence, VK_TRUE, UINT64_MAX);

    VkSemaphore acquireSem = mSync.GetImageAvailableSemaphore(mFrameIndex);
    uint32_t imageIndex = 0;
    VkResult result = vkAcquireNextImageKHR(
        device, mSwapchain.GetHandle(), UINT64_MAX, acquireSem, VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) { RecreateSwapchain(); return; }

    if (mImageFences[imageIndex] != VK_NULL_HANDLE && mImageFences[imageIndex] != frameFence)
        vkWaitForFences(device, 1, &mImageFences[imageIndex], VK_TRUE, UINT64_MAX);
    mImageFences[imageIndex] = frameFence;

    vkResetFences(device, 1, &frameFence);

    VkExtent2D extent = mSwapchain.GetExtent();
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    glm::mat4 view = mCamera.GetViewMatrix();
    glm::mat4 proj = mCamera.GetProjectionMatrix(aspect);

    const auto* sunLight = mRegistry.GetLight(mSunEntity);
    glm::vec3 sunColor = sunLight ? sunLight->color     : glm::vec3(1.0f);
    float sunIntensity  = sunLight ? sunLight->intensity : 1.0f;

    float az = glm::radians(mLightAzimuth);
    float el = glm::radians(mLightElevation);
    glm::vec3 sunDir = glm::normalize(glm::vec3(
        -glm::cos(el) * glm::sin(az), -glm::sin(el), -glm::cos(el) * glm::cos(az)));

    if (mCSMEnabled)
        mCSM.Update(view, proj, mCamera.GetNear(), mCamera.GetFar(), sunDir);

    FrameData fd{};
    fd.view           = view;
    fd.projection     = proj;
    fd.viewProjection = proj * view;
    fd.cameraPos      = glm::vec4(mCamera.GetPosition(), 0.0f);
    fd.sunDirection   = glm::vec4(sunDir, mCSMEnabled ? 1.0f : 0.0f);
    fd.sunColor       = glm::vec4(sunColor, sunIntensity);
    for (uint32_t c = 0; c < CascadedShadowMap::CASCADE_COUNT; c++)
        fd.cascadeViewProj[c] = mCSM.GetViewProj(c);
    fd.cascadeSplits = mCSM.GetSplits();
    std::memcpy(mFrameUBOs[mFrameIndex].GetMappedData(), &fd, sizeof(FrameData));

    mImageCache.EvictUnused(mFrameNumber, 60);

    mGPUProfiler.CollectResults(device, mFrameIndex);
    mPipelineStats.CollectResults(device, mFrameIndex);

    auto cmd = mCommandBuffers.Begin(device, imageIndex);

    mGPUProfiler.BeginFrame(cmd, mFrameIndex);
    BuildAndExecuteRenderGraph(cmd, imageIndex);
    mGPUProfiler.EndFrame(cmd, mFrameIndex);

    mCommandBuffers.End(imageIndex);

    VkSemaphore          waitSems[]   = { acquireSem };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore          signalSems[] = { mSync.GetRenderFinishedSemaphore(imageIndex) };
    VkCommandBuffer      cmdBuf       = mCommandBuffers.Get(imageIndex);

    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = waitSems;
    submitInfo.pWaitDstStageMask    = waitStages;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &cmdBuf;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = signalSems;

    VK_CHECK(vkQueueSubmit(mDevice.GetGraphicsQueue(), 1, &submitInfo, frameFence));

    VkSwapchainKHR swapchains[] = { mSwapchain.GetHandle() };
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = signalSems;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = swapchains;
    presentInfo.pImageIndices      = &imageIndex;

    result = vkQueuePresentKHR(mDevice.GetPresentQueue(), &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || mFramebufferResized) {
        mFramebufferResized = false;
        RecreateSwapchain();
    }

    mFrameIndex = (mFrameIndex + 1) % FRAMES_IN_FLIGHT;
    mFrameNumber++;
}

// =======================================================================
// Build and execute render graph
// =======================================================================
void Application::BuildAndExecuteRenderGraph(VkCommandBuffer cmd, uint32_t imageIndex) {
    constexpr uint32_t CC = CascadedShadowMap::CASCADE_COUNT;
    VkExtent2D extent = mSwapchain.GetExtent();

    bool useGPU       = mGPUDriven && mIndirectRenderer.GetDrawCount() > 0;
    bool useOcclusion = useGPU && mOcclusionCulling;

    CullParams cullParams{};
    if (useGPU) {
        float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        glm::mat4 view = mCamera.GetViewMatrix();
        glm::mat4 proj = mCamera.GetProjectionMatrix(aspect);
        glm::mat4 viewProj = proj * view;

        cullParams.viewProjection = viewProj;
        ExtractFrustumPlanes(viewProj, cullParams.frustumPlanes);
        cullParams.hiZSize        = glm::vec2(float(mHiZBuffer.GetWidth()), float(mHiZBuffer.GetHeight()));
        cullParams.nearPlane      = mCamera.GetNear();
        cullParams.farPlane       = mCamera.GetFar();
        cullParams.drawCount      = mIndirectRenderer.GetDrawCount();
        cullParams.occluderCount  = useOcclusion ? mIndirectRenderer.GetOccluderCount()
                                                 : mIndirectRenderer.GetDrawCount();
        cullParams.candidateCount = 0;

        if (mFrameNumber == 0)
            LOG_INFO("Culling: {} draws, {} occluders (ratio {}), occlusion {}",
                     cullParams.drawCount, cullParams.occluderCount, mOccluderRatio,
                     useOcclusion ? "ON" : "OFF");
    }

    mRenderGraph.BeginFrame(mFrameNumber);

    // ----- 1. Resources -----
    auto swapRes = mRenderGraph.AddImage("Swapchain",
        mSwapchain.GetImages()[imageIndex], mSwapchain.GetImageViews()[imageIndex],
        VK_IMAGE_LAYOUT_UNDEFINED);

    auto csmRes = mRenderGraph.AddImage("CSM",
        mCSM.GetImage(), mCSM.GetArrayView(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_ASPECT_DEPTH_BIT, CC);

    auto depthRes = mRenderGraph.AddImage("Depth",
        mDepthImage.GetImage(), mDepthImage.GetView(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    // ----- 2. Passes -----

    RenderGraph::PassHandle frustumCullPassH   = RenderGraph::INVALID_PASS;
    RenderGraph::PassHandle occlusionTestPassH = RenderGraph::INVALID_PASS;

    if (useGPU) {
        FrustumCullPass::Desc fcDesc{};
        fcDesc.culling = &mComputeCulling;
        fcDesc.params  = &cullParams;
        frustumCullPassH = mRenderGraph.AddPass(std::make_unique<FrustumCullPass>(fcDesc));

        if (useOcclusion) {
            OccluderDepthPass::Desc odDesc{};
            odDesc.depthResource         = depthRes;
            odDesc.frustumCullPassHandle = frustumCullPassH;
            odDesc.extent                = extent;
            odDesc.depthView             = mDepthImage.GetView();
            odDesc.pipeline              = mDepthPrepassPipeline;
            odDesc.pipelineLayout        = mDepthPrepassPipelineLayout;
            odDesc.bindlessSet           = mDescriptors.GetSet();
            odDesc.frameDescSet          = mFrameDescSets[mFrameIndex];
            odDesc.meshPool              = &mMeshPool;
            odDesc.culling               = &mComputeCulling;
            odDesc.maxDrawCount          = mIndirectRenderer.GetDrawCount();
            auto occDepthPassH = mRenderGraph.AddPass(std::make_unique<OccluderDepthPass>(odDesc));

            HiZBuildPass::Desc hzDesc{};
            hzDesc.depthResource           = depthRes;
            hzDesc.occluderDepthPassHandle = occDepthPassH;
            hzDesc.hiZ                     = &mHiZBuffer;
            auto hiZPassH = mRenderGraph.AddPass(std::make_unique<HiZBuildPass>(hzDesc));

            OcclusionTestPass::Desc otDesc{};
            otDesc.hiZBuildPassHandle = hiZPassH;
            otDesc.depthResource      = depthRes;
            otDesc.culling            = &mComputeCulling;
            otDesc.params             = &cullParams;
            occlusionTestPassH = mRenderGraph.AddPass(std::make_unique<OcclusionTestPass>(otDesc));
        }
    }

    ShadowPass::Desc shadowDesc{};
    shadowDesc.csmResource   = csmRes;
    shadowDesc.csm           = &mCSM;
    shadowDesc.skip          = !mCSMEnabled;
    shadowDesc.pipeline      = mShadowPipeline;
    shadowDesc.pipelineLayout = mShadowPipelineLayout;
    shadowDesc.registry      = &mRegistry;
    shadowDesc.meshPool      = &mMeshPool;
    if (useGPU) {
        shadowDesc.gpuDriven                  = true;
        shadowDesc.indirectPipeline           = mShadowIndirectPipeline;
        shadowDesc.indirectPipelineLayout     = mShadowIndirectPipelineLayout;
        shadowDesc.indirectDescSet            = mShadowIndirectDescSet;
        shadowDesc.indirectBuffer             = mIndirectRenderer.GetIndirectBuffer();
        shadowDesc.countBuffer                = mIndirectRenderer.GetCountBuffer();
        shadowDesc.maxDrawCount               = mIndirectRenderer.GetDrawCount();
    }
    auto shadowPassH = mRenderGraph.AddPass(std::make_unique<ShadowPass>(shadowDesc));

    auto hdrRes = mRenderGraph.AddImage("HDRColor",
        mPostProcess.GetHDRImage(), mPostProcess.GetHDRView(),
        VK_IMAGE_LAYOUT_UNDEFINED);

    ForwardPass::Desc fwdDesc{};
    fwdDesc.csmResource        = csmRes;
    fwdDesc.depthResource      = depthRes;
    fwdDesc.colorResource      = hdrRes;
    fwdDesc.shadowPassHandle   = shadowPassH;
    fwdDesc.extent             = extent;
    fwdDesc.colorView          = mPostProcess.GetHDRView();
    fwdDesc.depthView          = mDepthImage.GetView();
    fwdDesc.pipeline           = mPBRPipeline;
    fwdDesc.pipelineLayout     = mPBRPipelineLayout;
    fwdDesc.bindlessSet        = mDescriptors.GetSet();
    fwdDesc.frameDescSet       = mFrameDescSets[mFrameIndex];
    fwdDesc.registry           = &mRegistry;
    fwdDesc.meshPool           = &mMeshPool;
    fwdDesc.gpuMaterials       = &mGPUMaterials;
    if (mCurrentMSAA != VK_SAMPLE_COUNT_1_BIT && mPostProcess.GetMSAAColorView()) {
        fwdDesc.msaaSamples       = mCurrentMSAA;
        fwdDesc.msaaColorImage    = mPostProcess.GetMSAAColorImage();
        fwdDesc.msaaColorView     = mPostProcess.GetMSAAColorView();
        fwdDesc.msaaDepthImage    = mPostProcess.GetMSAADepthImage();
        fwdDesc.msaaDepthView     = mPostProcess.GetMSAADepthView();
        fwdDesc.resolveColorView  = mPostProcess.GetHDRView();
        fwdDesc.resolveDepthImage = mDepthImage.GetImage();
        fwdDesc.resolveDepthView  = mDepthImage.GetView();
    }
    if (useGPU) {
        fwdDesc.gpuDriven                = true;
        fwdDesc.occlusionEnabled         = useOcclusion;
        fwdDesc.indirectPipeline         = mPBRIndirectPipeline;
        fwdDesc.indirectPipelineLayout   = mPBRIndirectPipelineLayout;
        fwdDesc.occluderBuffer           = mComputeCulling.GetOccluderIndirectBuffer();
        fwdDesc.occluderCountBuffer      = mComputeCulling.GetOccluderCountBuffer();
        fwdDesc.maxOccluderCount         = mIndirectRenderer.GetDrawCount();
        fwdDesc.visibleBuffer            = mComputeCulling.GetVisibleIndirectBuffer();
        fwdDesc.visibleCountBuffer       = mComputeCulling.GetVisibleCountBuffer();
        fwdDesc.maxVisibleCount          = mIndirectRenderer.GetDrawCount();
        fwdDesc.occlusionTestPassHandle  = occlusionTestPassH;
        fwdDesc.frustumCullPassHandle    = frustumCullPassH;
    }
    auto forwardPassH = mRenderGraph.AddPass(std::make_unique<ForwardPass>(fwdDesc));

    // --- Ray tracing pass (between forward and post-process) ---
    RenderGraph::PassHandle rtPassH = RenderGraph::INVALID_PASS;
    if (mRayTracingEnabled && (mRTShadowsEnabled || mRTReflEnabled)) {
        float aspect_ = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        glm::mat4 viewMat_ = mCamera.GetViewMatrix();
        glm::mat4 projMat_ = mCamera.GetProjectionMatrix(aspect_);
        glm::mat4 viewProj_ = projMat_ * viewMat_;
        glm::mat4 invVP_ = glm::inverse(viewProj_);

        const auto* sunLight = mRegistry.GetLight(mSunEntity);

        mRTShadows.SetEnabled(mRTShadowsEnabled);
        mRTReflections.SetEnabled(mRTReflEnabled);

        // Update composite descriptor set (only when resources change)
        if (mRTCompositeDescDirty) {
            VkDescriptorImageInfo hdrInfo{VK_NULL_HANDLE, mPostProcess.GetHDRView(), VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo shadowInfo{mRTDepthSampler, mRTShadows.GetOutputView(), VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo reflInfo{mRTDepthSampler, mRTReflections.GetOutputView(), VK_IMAGE_LAYOUT_GENERAL};

            VkWriteDescriptorSet writes[3] = {};
            writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mRTCompositeDescSet,
                          0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrInfo};
            writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mRTCompositeDescSet,
                          1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &shadowInfo};
            writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mRTCompositeDescSet,
                          2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &reflInfo};
            vkUpdateDescriptorSets(mDevice.GetHandle(), 3, writes, 0, nullptr);
            mRTCompositeDescDirty = false;
        }

        RayTracingPass::Desc rtDesc{};
        rtDesc.depthResource     = depthRes;
        rtDesc.colorResource     = hdrRes;
        rtDesc.forwardPassHandle = forwardPassH;
        rtDesc.shadows           = &mRTShadows;
        rtDesc.reflections       = &mRTReflections;
        rtDesc.accel             = &mAccelStructure;
        rtDesc.depthView         = mDepthImage.GetView();
        rtDesc.depthSampler      = mRTDepthSampler;
        rtDesc.extent            = extent;
        rtDesc.invViewProj       = invVP_;
        float rtAz = glm::radians(mLightAzimuth);
        float rtEl = glm::radians(mLightElevation);
        glm::vec3 rtSunDir = glm::normalize(glm::vec3(
            -glm::cos(rtEl) * glm::sin(rtAz), -glm::sin(rtEl), -glm::cos(rtEl) * glm::cos(rtAz)));
        rtDesc.lightDir          = -rtSunDir;
        rtDesc.lightRadius       = mRTLightRadius;
        rtDesc.cameraPos         = mCamera.GetPosition();
        rtDesc.roughness         = mRTReflRoughness;
        rtDesc.compositePipeline   = mRTCompositePipeline;
        rtDesc.compositePipeLayout = mRTCompositePipeLayout;
        rtDesc.compositeDescSet    = mRTCompositeDescSet;
        rtDesc.shadowStrength      = mRTShadowStrength;
        rtDesc.reflectionStrength  = mRTReflStrength;
        rtDesc.debugShadowVis      = mRTDebugShadowVis;

        rtPassH = mRenderGraph.AddPass(std::make_unique<RayTracingPass>(rtDesc));
    }

    // Post-processing: HDR → swapchain
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    glm::mat4 invProj = glm::inverse(mCamera.GetProjectionMatrix(aspect));
    glm::mat4 proj    = mCamera.GetProjectionMatrix(aspect);
    float projInfoData[4] = {
        2.0f / (extent.width * proj[0][0]),
        2.0f / (extent.height * proj[1][1]),
        -(1.0f - proj[2][0]) / proj[0][0],
        -(1.0f + proj[2][1]) / proj[1][1]
    };

    PostProcessPass::Desc ppDesc{};
    ppDesc.hdrResource        = hdrRes;
    ppDesc.depthResource      = depthRes;
    ppDesc.swapchainResource  = swapRes;
    ppDesc.forwardPassHandle  = (rtPassH != RenderGraph::INVALID_PASS) ? rtPassH : forwardPassH;
    ppDesc.stack              = &mPostProcess;
    ppDesc.swapchainView      = mSwapchain.GetImageViews()[imageIndex];
    ppDesc.depthView          = mDepthImage.GetView();
    ppDesc.extent             = extent;
    ppDesc.deltaTime          = mDeltaTime;
    ppDesc.invProjection      = &invProj[0][0];
    ppDesc.projInfo           = projInfoData;
    ppDesc.farPlane           = mCamera.GetFar();
    auto postProcessPassH = mRenderGraph.AddPass(std::make_unique<PostProcessPass>(ppDesc));

    RenderGraph::PassHandle lastPassBeforePresent = postProcessPassH;

    if (mShowUI) {
        ImGuiPass::Desc imguiDesc{};
        imguiDesc.swapchainResource  = swapRes;
        imguiDesc.previousPassHandle = postProcessPassH;
        imguiDesc.swapchainView      = mSwapchain.GetImageViews()[imageIndex];
        imguiDesc.extent             = extent;
        imguiDesc.debugUI            = &mDebugUI;
        lastPassBeforePresent = mRenderGraph.AddPass(std::make_unique<ImGuiPass>(imguiDesc));
    }

    mRenderGraph.AddPass(std::make_unique<PresentPass>(PresentPass::Desc{
        swapRes, lastPassBeforePresent
    }));

    // ----- 3. Compile & execute -----
    mRenderGraph.Compile();
    mRenderGraph.Execute(cmd, &mGPUProfiler, mFrameIndex, &mPipelineStats);
}

// =======================================================================
// GPU-Driven rendering init/shutdown
// =======================================================================
void Application::InitGPUDriven() {
    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();

    if (!mGPUDriven || mMeshPool.GetMeshCount() == 0) {
        mGPUDriven = false;
        return;
    }

    mIndirectRenderer.Initialize(allocator, device);
    mRegistry.UpdateTransforms();
    mIndirectRenderer.BuildCommands(allocator, mTransfer, mMeshPool, mRegistry, mOccluderRatio);

    mHiZBuffer.Initialize(device, allocator, mShaders);
    auto extent = mSwapchain.GetExtent();
    mHiZBuffer.Resize(device, allocator, extent.width, extent.height);
    mHiZBuffer.SetSourceDepth(mDepthImage.GetView());

    mComputeCulling.Initialize(device, allocator, mShaders);
    mComputeCulling.UpdateBuffers(allocator,
        mIndirectRenderer.GetIndirectBuffer(),
        mIndirectRenderer.GetDrawCount(),
        mIndirectRenderer.GetObjectBuffer(),
        mHiZBuffer.GetView(),
        mHiZBuffer.GetSampler());

    // Object SSBO descriptor for frame sets (binding 6)
    if (mIndirectRenderer.GetObjectBuffer() != VK_NULL_HANDLE) {
        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo objInfo{mIndirectRenderer.GetObjectBuffer(), 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet write{};
            write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet          = mFrameDescSets[i];
            write.dstBinding      = 6;
            write.descriptorCount = 1;
            write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo     = &objInfo;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }
    }

    // Shadow indirect descriptor set
    {
        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.maxSets       = 1;
        poolCI.poolSizeCount = 1;
        poolCI.pPoolSizes    = &poolSize;
        VK_CHECK(vkCreateDescriptorPool(device, &poolCI, nullptr, &mShadowIndirectDescPool));

        VkDescriptorSetAllocateInfo allocCI{};
        allocCI.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocCI.descriptorPool     = mShadowIndirectDescPool;
        allocCI.descriptorSetCount = 1;
        allocCI.pSetLayouts        = &mShadowIndirectDescLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &allocCI, &mShadowIndirectDescSet));

        VkDescriptorBufferInfo objInfo{mIndirectRenderer.GetObjectBuffer(), 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = mShadowIndirectDescSet;
        write.dstBinding      = 0;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo     = &objInfo;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    LOG_INFO("GPU-Driven rendering initialized ({} indirect draws)", mIndirectRenderer.GetDrawCount());
}

// =======================================================================
// Debug UI (Phase 7)
// =======================================================================
void Application::InitDebugUI() {
    auto device = mDevice.GetHandle();

    mDebugUI.Initialize(
        mVulkanInstance.GetHandle(), device, mDevice.GetPhysicalDevice(),
        mDevice.GetQueueFamilyIndices().graphicsFamily,
        mDevice.GetGraphicsQueue(), mWindow.GetHandle(),
        mSwapchain.GetImageFormat(), FRAMES_IN_FLIGHT);

    auto& uiState = mDebugUI.GetState();
    uiState.gpuDriven        = mGPUDriven;
    uiState.occlusionCulling = mOcclusionCulling;
    uiState.occluderRatio    = mOccluderRatio;
    uiState.sceneType        = mCurrentScene;
    uiState.lightAzimuth     = mLightAzimuth;
    uiState.lightElevation   = mLightElevation;
    uiState.csmEnabled       = mCSMEnabled;
    uiState.rtAvailable      = mRayTracingEnabled;
    uiState.rtShadowsEnabled = mRTShadowsEnabled;
    uiState.rtReflEnabled     = mRTReflEnabled;
    uiState.rtShadowStrength  = mRTShadowStrength;
    uiState.rtReflStrength    = mRTReflStrength;
    uiState.rtReflRoughness   = mRTReflRoughness;
    uiState.rtLightRadius     = mRTLightRadius;

    mGPUProfiler.Initialize(device, mDevice.GetPhysicalDevice(), FRAMES_IN_FLIGHT, 32);
    mPipelineStats.Initialize(device, FRAMES_IN_FLIGHT);

    mDebugVis.Initialize(device, mShaders, mPipelines,
                         mSwapchain.GetImageFormat(),
                         mDescriptors.GetLayout(), mFrameSetLayout);

    LabelVulkanObjects();

    LOG_INFO("Debug UI and profiling initialized");
}

void Application::ShutdownDebugUI() {
    auto device = mDevice.GetHandle();
    mDebugVis.Shutdown(device);
    mPipelineStats.Shutdown(device);
    mGPUProfiler.Shutdown(device);
    mDebugUI.Shutdown(device);
}

void Application::SyncUIState() {
    auto& uiState = mDebugUI.GetState();

    bool gpuChanged = (mGPUDriven != uiState.gpuDriven) ||
                      (mOcclusionCulling != uiState.occlusionCulling);

    mGPUDriven        = uiState.gpuDriven;
    mOcclusionCulling = uiState.occlusionCulling;
    mOccluderRatio    = uiState.occluderRatio;

    mPipelineStats.SetEnabled(uiState.pipelineStatsEnabled);

    // Light direction and CSM
    mLightAzimuth   = uiState.lightAzimuth;
    mLightElevation = uiState.lightElevation;
    mCSMEnabled     = uiState.csmEnabled;

    // RT state sync
    uiState.rtAvailable     = mRayTracingEnabled;
    mRTShadowsEnabled       = uiState.rtShadowsEnabled;
    mRTReflEnabled           = uiState.rtReflEnabled;
    mRTShadowStrength        = uiState.rtShadowStrength;
    mRTReflStrength          = uiState.rtReflStrength;
    mRTReflRoughness         = uiState.rtReflRoughness;
    mRTLightRadius           = uiState.rtLightRadius;
    mRTDebugShadowVis        = uiState.rtDebugShadowVis;

    if (gpuChanged) {
        LOG_INFO("Render mode changed: GPU-driven={}, occlusion={}",
                 mGPUDriven, mOcclusionCulling);
    }

    if (uiState.msaaChanged) {
        uiState.msaaChanged = false;
        int idx = std::clamp(uiState.msaaIndex, 0, static_cast<int>(mSupportedMSAA.size()) - 1);
        VkSampleCountFlagBits newSamples = mSupportedMSAA[idx];
        if (newSamples != mCurrentMSAA) {
            mDevice.WaitIdle();
            mPostProcess.SetMSAASampleCount(mDevice.GetHandle(), mMemory.GetAllocator(), newSamples);
            RecreatePBRPipelines(newSamples);
        }
    }

    if (uiState.sceneChanged) {
        uiState.sceneChanged = false;
        if (uiState.sceneType != mCurrentScene) {
            ReloadScene(uiState.sceneType);
        }
    }
}

void Application::LabelVulkanObjects() {
    auto device = mDevice.GetHandle();

    ObjectLabeling::NameImage(device, mDepthImage.GetImage(), "DepthBuffer");
    ObjectLabeling::NameImageView(device, mDepthImage.GetView(), "DepthBuffer_View");

    ObjectLabeling::NamePipeline(device, mPBRPipeline, "PBR_Pipeline");
    ObjectLabeling::NamePipeline(device, mShadowPipeline, "Shadow_Pipeline");

    if (mPBRIndirectPipeline)
        ObjectLabeling::NamePipeline(device, mPBRIndirectPipeline, "PBR_Indirect_Pipeline");
    if (mShadowIndirectPipeline)
        ObjectLabeling::NamePipeline(device, mShadowIndirectPipeline, "Shadow_Indirect_Pipeline");
    if (mDepthPrepassPipeline)
        ObjectLabeling::NamePipeline(device, mDepthPrepassPipeline, "DepthPrepass_Pipeline");

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
        std::string name = "FrameUBO_" + std::to_string(i);
        ObjectLabeling::NameBuffer(device, mFrameUBOs[i].GetHandle(), name.c_str());
        std::string dsName = "FrameDescSet_" + std::to_string(i);
        ObjectLabeling::NameDescriptorSet(device, mFrameDescSets[i], dsName.c_str());
    }

    ObjectLabeling::NameBuffer(device, mMaterialSSBO.GetHandle(), "MaterialSSBO");
}

void Application::ShutdownGPUDriven() {
    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();

    mComputeCulling.Shutdown(device, allocator);
    mHiZBuffer.Shutdown(device, allocator);
    mIndirectRenderer.Shutdown(allocator);

    if (mShadowIndirectDescPool) { vkDestroyDescriptorPool(device, mShadowIndirectDescPool, nullptr); mShadowIndirectDescPool = VK_NULL_HANDLE; }
}

// =======================================================================
// Ray Tracing (Phase 9)
// =======================================================================
void Application::InitRayTracing() {
    if (!mDevice.IsRayTracingSupported() || mMeshPool.GetMeshCount() == 0) {
        mRayTracingEnabled = false;
        LOG_INFO("Ray tracing: disabled ({})", mDevice.IsRayTracingSupported() ? "no meshes" : "not supported");
        return;
    }

    mRayTracingEnabled = true;
    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();
    auto extent    = mSwapchain.GetExtent();

    mAccelStructure.Initialize(device, allocator, mTransfer);
    mAccelStructure.BuildBLAS(mMeshPool);
    mRegistry.UpdateTransforms();
    mAccelStructure.BuildTLAS(mRegistry, mMeshPool);

    mRTShadows.Initialize(device, allocator, mShaders, extent.width, extent.height);
    mRTReflections.Initialize(device, allocator, mShaders, extent.width, extent.height);

    // Depth sampler for RT passes
    VkSamplerCreateInfo samplerCI{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerCI.magFilter    = VK_FILTER_NEAREST;
    samplerCI.minFilter    = VK_FILTER_NEAREST;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device, &samplerCI, nullptr, &mRTDepthSampler));

    // Composite pipeline
    {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT};

        VkDescriptorSetLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutCI.bindingCount = 3;
        layoutCI.pBindings    = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &mRTCompositeDescLayout));

        VkDescriptorPoolSize poolSizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        };
        VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolCI.maxSets       = 1;
        poolCI.poolSizeCount = 2;
        poolCI.pPoolSizes    = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(device, &poolCI, nullptr, &mRTCompositeDescPool));

        VkDescriptorSetAllocateInfo allocCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocCI.descriptorPool     = mRTCompositeDescPool;
        allocCI.descriptorSetCount = 1;
        allocCI.pSetLayouts        = &mRTCompositeDescLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &allocCI, &mRTCompositeDescSet));

        struct CompositePushConstants {
            glm::uvec2 resolution;
            float shadowStrength;
            float reflectionStrength;
            uint32_t enableShadows;
            uint32_t enableReflections;
            uint32_t debugShadowVis;
        };
        VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CompositePushConstants)};
        VkPipelineLayoutCreateInfo pipeLayoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeLayoutCI.setLayoutCount         = 1;
        pipeLayoutCI.pSetLayouts            = &mRTCompositeDescLayout;
        pipeLayoutCI.pushConstantRangeCount = 1;
        pipeLayoutCI.pPushConstantRanges    = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(device, &pipeLayoutCI, nullptr, &mRTCompositePipeLayout));

        VkShaderModule mod = mShaders.GetOrLoad("shaders/rt_composite.comp.spv");
        VkComputePipelineCreateInfo pipeCI{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeCI.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeCI.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeCI.stage.module = mod;
        pipeCI.stage.pName  = "main";
        pipeCI.layout       = mRTCompositePipeLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &mRTCompositePipeline));
    }

    mRTCompositeDescDirty = true;

    LOG_INFO("Ray tracing initialized: BLAS {:.1f} KB, TLAS {:.1f} KB",
             mAccelStructure.GetTotalBLASMemory() / 1024.0f,
             mAccelStructure.GetTLASMemory() / 1024.0f);
}

void Application::ShutdownRayTracing() {
    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();

    mRTShadows.Shutdown(device, allocator);
    mRTReflections.Shutdown(device, allocator);
    mAccelStructure.Shutdown(allocator);

    if (mRTCompositePipeline)   vkDestroyPipeline(device, mRTCompositePipeline, nullptr);
    if (mRTCompositePipeLayout) vkDestroyPipelineLayout(device, mRTCompositePipeLayout, nullptr);
    if (mRTCompositeDescPool)   vkDestroyDescriptorPool(device, mRTCompositeDescPool, nullptr);
    if (mRTCompositeDescLayout) vkDestroyDescriptorSetLayout(device, mRTCompositeDescLayout, nullptr);
    if (mRTDepthSampler)        vkDestroySampler(device, mRTDepthSampler, nullptr);

    mRTCompositePipeline   = VK_NULL_HANDLE;
    mRTCompositePipeLayout = VK_NULL_HANDLE;
    mRTCompositeDescPool   = VK_NULL_HANDLE;
    mRTCompositeDescLayout = VK_NULL_HANDLE;
    mRTDepthSampler        = VK_NULL_HANDLE;
}

// =======================================================================
// Scene management (switching)
// =======================================================================
void Application::ClearScene() {
    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();

    ShutdownRayTracing();
    ShutdownGPUDriven();

    mMeshPool.Destroy(allocator);

    for (auto& tex : mGPUTextures)
        tex.Destroy(allocator, device);
    mGPUTextures.clear();

    for (uint32_t idx : mTextureDescriptorIndices)
        mDescriptors.FreeTextureIndex(idx);
    mTextureDescriptorIndices.clear();

    mGPUMaterials.clear();
    mMaterialSSBO.Destroy(allocator);

    mRegistry.Clear();
    mSunEntity = INVALID_ENTITY;
    mModelData = ModelData{};
    mRayTracingEnabled = false;
}

void Application::LoadTestScene() {
    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();

    // --- Sun light (side-front for clear shadow patterns) ---
    mSunEntity = mRegistry.CreateEntity();
    mRegistry.AddTransform(mSunEntity);
    auto& sunLight = mRegistry.AddLight(mSunEntity);
    sunLight.direction = glm::normalize(glm::vec3(-0.5f, -0.7f, -0.3f));
    sunLight.color     = glm::vec3(1.0f, 0.97f, 0.9f);
    sunLight.intensity = 4.0f;

    // --- Checker texture for ground ---
    constexpr uint32_t texW = 512, texH = 512, tileSize = 32;
    std::vector<uint8_t> checkerPixels(texW * texH * 4);
    for (uint32_t y = 0; y < texH; y++) {
        for (uint32_t x = 0; x < texW; x++) {
            bool white = ((x / tileSize) + (y / tileSize)) % 2 == 0;
            uint8_t c  = white ? 200 : 80;
            uint32_t idx = (y * texW + x) * 4;
            checkerPixels[idx + 0] = c;
            checkerPixels[idx + 1] = c;
            checkerPixels[idx + 2] = c;
            checkerPixels[idx + 3] = 255;
        }
    }
    VulkanImage checkerImg;
    checkerImg.CreateTexture2D(allocator, device, mTransfer,
                               texW, texH, VK_FORMAT_R8G8B8A8_SRGB, checkerPixels.data());
    uint32_t checkerDescIdx = mDescriptors.AllocateTextureIndex();
    mDescriptors.UpdateTexture(device, checkerDescIdx, checkerImg.GetView(),
                               mDescriptors.GetDefaultSampler());
    mGPUTextures.push_back(std::move(checkerImg));
    mTextureDescriptorIndices.push_back(checkerDescIdx);

    // --- Meshes ---
    // 0: large ground plane
    {
        MeshData mesh;
        ModelLoader::GenerateGroundPlane(mesh, 30.0f);
        mModelData.meshes.push_back(std::move(mesh));
    }
    // 1: sphere
    {
        MeshData mesh;
        ModelLoader::GenerateUVSphere(mesh, 1.0f, 64, 32);
        mModelData.meshes.push_back(std::move(mesh));
    }
    // 2: cube
    {
        ModelData cubeData;
        ModelLoader::GenerateProceduralCube(cubeData);
        mModelData.meshes.push_back(std::move(cubeData.meshes[0]));
    }
    // 3: small mirror plane
    {
        MeshData mesh;
        ModelLoader::GenerateGroundPlane(mesh, 4.0f);
        mModelData.meshes.push_back(std::move(mesh));
    }

    // --- Materials ---
    auto pushMat = [&](glm::vec3 color, float metallic, float roughness, uint32_t baseTexIdx) {
        GPUMaterialData m{};
        m.baseColorFactor         = glm::vec4(color, 1.0f);
        m.metallicFactor          = metallic;
        m.roughnessFactor         = roughness;
        m.baseColorTexIdx         = baseTexIdx;
        m.normalTexIdx            = mDefaultNormalDescIdx;
        m.metallicRoughnessTexIdx = mWhiteTexDescIdx;
        m.aoTexIdx                = mWhiteTexDescIdx;
        m.emissiveTexIdx          = mBlackTexDescIdx;
        mGPUMaterials.push_back(m);
    };
    // 0: ground (checker, non-metallic, medium roughness)
    pushMat({0.6f, 0.6f, 0.6f}, 0.0f, 0.8f, checkerDescIdx);
    // 1: shiny metallic copper sphere
    pushMat({0.955f, 0.638f, 0.538f}, 1.0f, 0.25f, mWhiteTexDescIdx);
    // 2: rough red plastic sphere
    pushMat({0.8f, 0.1f, 0.08f}, 0.0f, 0.7f, mWhiteTexDescIdx);
    // 3: blue cube (dielectric, medium roughness)
    pushMat({0.1f, 0.2f, 0.85f}, 0.0f, 0.5f, mWhiteTexDescIdx);
    // 4: wood sphere (dielectric, medium-high roughness)
    pushMat({0.55f, 0.35f, 0.18f}, 0.0f, 0.65f, mWhiteTexDescIdx);
    // 7: smooth green plastic sphere
    pushMat({0.1f, 0.75f, 0.15f}, 0.0f, 0.15f, mWhiteTexDescIdx);
    // 5: mirror floor (metallic, very smooth)
    pushMat({0.6f, 0.1f, 0.6f}, 1.0f, 0.02f, mWhiteTexDescIdx);
    // 6: white wall / tall pillar (dielectric)
    pushMat({0.9f, 0.88f, 0.85f}, 0.0f, 0.7f, mWhiteTexDescIdx);

    // --- Entities ---
    auto makeObj = [&](int meshIdx, int matIdx, glm::vec3 pos, glm::vec3 scale = glm::vec3(1.0f)) {
        Entity e = mRegistry.CreateEntity();
        auto& tc = mRegistry.AddTransform(e);
        tc.localPosition = pos;
        tc.localScale    = scale;
        mRegistry.AddMesh(e).meshIndex = meshIdx;
        mRegistry.AddMaterial(e).materialIndex = matIdx;
    };

    // Ground
    makeObj(0, 0, {0.0f, 0.0f, 0.0f});
    // Shiny copper sphere (center, floating for clear shadow)
    makeObj(1, 1, {0.0f, 1.5f, 0.0f});
    // Rough red plastic sphere (right side)
    makeObj(1, 2, {4.0f, 1.0f, -2.0f});
    // Blue cube (left side, tall for clear rectangular shadow)
    makeObj(2, 3, {-4.0f, 1.5f, -1.0f}, {1.5f, 3.0f, 1.5f});
    // Wood sphere (front-right)
    makeObj(1, 4, {2.0f, 0.8f, 3.0f}, {0.8f, 0.8f, 0.8f});
    // Mirror floor patch (slightly elevated to avoid z-fighting)
    makeObj(3, 5, {0.0f, 0.02f, 4.0f});
    // White pillar (back, tall shadow caster)
    makeObj(2, 6, {-2.0f, 2.0f, -5.0f}, {1.0f, 4.0f, 1.0f});
    // Smooth green plastic sphere near mirror plane
    makeObj(1, 7, {1.0f, 0.6f, 4.5f}, {0.6f, 0.6f, 0.6f});

    // --- Upload scene data ---
    mMaterialSSBO.CreateDeviceLocal(allocator, mTransfer,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        mGPUMaterials.data(), mGPUMaterials.size() * sizeof(GPUMaterialData));

    if (mDevice.IsRayTracingSupported()) {
        mMeshPool.Upload(allocator, mTransfer, mModelData.meshes,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    } else {
        mMeshPool.Upload(allocator, mTransfer, mModelData.meshes);
    }

    mModelData = ModelData{};

    LOG_INFO("Test scene loaded: {} meshes, {} materials, {} entities",
             mMeshPool.GetMeshCount(), mGPUMaterials.size(), mRegistry.EntityCount());
}

void Application::ReloadScene(SceneType newType) {
    mDevice.WaitIdle();
    mCurrentScene = newType;

    ClearScene();

    if (newType == SceneType::TestScene) {
        LoadTestScene();
    } else {
        LoadScene();
    }

    // Update material SSBO binding in frame descriptor sets
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo matInfo{mMaterialSSBO.GetHandle(), 0, mMaterialSSBO.GetSize()};
        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = mFrameDescSets[i];
        write.dstBinding      = 1;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo     = &matInfo;
        vkUpdateDescriptorSets(mDevice.GetHandle(), 1, &write, 0, nullptr);
    }

    mGPUDriven = true;
    InitGPUDriven();
    InitRayTracing();

    // Reset camera for test scene
    if (newType == SceneType::TestScene) {
        mCamera.Init(glm::vec3(0.0f, 6.0f, 12.0f), glm::vec3(0.0f, 0.0f, 0.0f), 45.0f, 0.01f, 100.0f);
    } else {
        mCamera.Init(glm::vec3(0, 1.6f, 0), glm::vec3(0, 1.6f, -1.0f), 45.0f, 0.01f, 100.0f);
    }

    LOG_INFO("Scene switched to: {}", newType == SceneType::TestScene ? "Test Scene" : "Sponza");
}

// =======================================================================
// Compute culling
// =======================================================================
void Application::ExtractFrustumPlanes(const glm::mat4& vp, glm::vec4 planes[6]) {
    planes[0] = glm::vec4(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]); // left
    planes[1] = glm::vec4(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]); // right
    planes[2] = glm::vec4(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]); // bottom
    planes[3] = glm::vec4(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]); // top
    planes[4] = glm::vec4(vp[0][3] + vp[0][2], vp[1][3] + vp[1][2], vp[2][3] + vp[2][2], vp[3][3] + vp[3][2]); // near
    planes[5] = glm::vec4(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]); // far

    for (int i = 0; i < 6; i++) {
        float len = glm::length(glm::vec3(planes[i]));
        planes[i] /= len;
    }
}


// =======================================================================
// Swapchain recreation
// =======================================================================
void Application::RecreateSwapchain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(mWindow.GetHandle(), &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(mWindow.GetHandle(), &width, &height);
        mWindow.WaitEvents();
    }

    mDevice.WaitIdle();

    mDepthImage.Destroy(mMemory.GetAllocator(), mDevice.GetHandle());
    mSwapchain.Recreate(mDevice.GetHandle(), mDevice.GetPhysicalDevice(),
                        mSurface, mWindow.GetHandle(), mDevice.GetQueueFamilyIndices());
    CreateDepthBuffer();

    if (mGPUDriven) {
        auto extent = mSwapchain.GetExtent();
        mHiZBuffer.Resize(mDevice.GetHandle(), mMemory.GetAllocator(), extent.width, extent.height);
        mHiZBuffer.SetSourceDepth(mDepthImage.GetView());

        mComputeCulling.UpdateBuffers(mMemory.GetAllocator(),
            mIndirectRenderer.GetIndirectBuffer(),
            mIndirectRenderer.GetDrawCount(),
            mIndirectRenderer.GetObjectBuffer(),
            mHiZBuffer.GetView(),
            mHiZBuffer.GetSampler());
    }

    {
        auto extent = mSwapchain.GetExtent();
        mPostProcess.Resize(mDevice.GetHandle(), mMemory.GetAllocator(), extent.width, extent.height);
    }

    if (mRayTracingEnabled) {
        auto extent = mSwapchain.GetExtent();
        mRTShadows.Resize(mDevice.GetHandle(), mMemory.GetAllocator(), extent.width, extent.height);
        mRTReflections.Resize(mDevice.GetHandle(), mMemory.GetAllocator(), extent.width, extent.height);
        mRTCompositeDescDirty = true;
    }

    LOG_INFO("Swapchain recreated (MSAA: {}x)", static_cast<int>(mCurrentMSAA));
}

// =======================================================================
// Cleanup
// =======================================================================
void Application::CleanupVulkan() {
    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();

    mPipelines.SaveCache("pipeline_cache.bin");
    mInput.SaveBindings("input_bindings.cfg");

    ShutdownDebugUI();

    if (mMultiThreading) {
        mSubmitThread.Drain();
        mSubmitThread.Shutdown();
        for (auto pool : mWorkerCommandPools)
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
        mWorkerCommandPools.clear();
        mSecondaryCommandBuffers.clear();
        mThreadPool.Shutdown();
    }

    mPostProcess.Shutdown(device, allocator);

    ShutdownRayTracing();
    ShutdownGPUDriven();

    mMeshPool.Destroy(allocator);

    for (auto& tex : mGPUTextures)
        tex.Destroy(allocator, device);
    mGPUTextures.clear();

    for (uint32_t idx : mTextureDescriptorIndices)
        mDescriptors.FreeTextureIndex(idx);
    mTextureDescriptorIndices.clear();

    mWhiteTexture.Destroy(allocator, device);
    mBlackTexture.Destroy(allocator, device);
    mDefaultNormalTexture.Destroy(allocator, device);
    mDescriptors.FreeTextureIndex(mWhiteTexDescIdx);
    mDescriptors.FreeTextureIndex(mBlackTexDescIdx);
    mDescriptors.FreeTextureIndex(mDefaultNormalDescIdx);

    mMaterialSSBO.Destroy(allocator);

    for (auto& ubo : mFrameUBOs)
        ubo.Destroy(allocator);
    mFrameUBOs.clear();

    if (mFrameDescPool)   vkDestroyDescriptorPool(device, mFrameDescPool, nullptr);
    if (mFrameSetLayout)  vkDestroyDescriptorSetLayout(device, mFrameSetLayout, nullptr);
    mFrameDescPool  = VK_NULL_HANDLE;
    mFrameSetLayout = VK_NULL_HANDLE;

    mRenderGraph.Shutdown();
    mImageCache.Shutdown();
    mIBL.Shutdown(allocator, device);
    mCSM.Shutdown(allocator, device);
    mDepthImage.Destroy(allocator, device);

    if (mPBRPipeline)              vkDestroyPipeline(device, mPBRPipeline, nullptr);
    if (mPBRPipelineLayout)        vkDestroyPipelineLayout(device, mPBRPipelineLayout, nullptr);
    if (mShadowPipeline)           vkDestroyPipeline(device, mShadowPipeline, nullptr);
    if (mShadowPipelineLayout)     vkDestroyPipelineLayout(device, mShadowPipelineLayout, nullptr);
    if (mPBRIndirectPipeline)      vkDestroyPipeline(device, mPBRIndirectPipeline, nullptr);
    if (mPBRIndirectPipelineLayout) vkDestroyPipelineLayout(device, mPBRIndirectPipelineLayout, nullptr);
    if (mShadowIndirectPipeline)       vkDestroyPipeline(device, mShadowIndirectPipeline, nullptr);
    if (mShadowIndirectPipelineLayout) vkDestroyPipelineLayout(device, mShadowIndirectPipelineLayout, nullptr);
    if (mShadowIndirectDescLayout)     vkDestroyDescriptorSetLayout(device, mShadowIndirectDescLayout, nullptr);
    if (mDepthPrepassPipeline)         vkDestroyPipeline(device, mDepthPrepassPipeline, nullptr);
    if (mDepthPrepassPipelineLayout)   vkDestroyPipelineLayout(device, mDepthPrepassPipelineLayout, nullptr);

    mShaders.Shutdown();
    mPipelines.Shutdown();
    mDescriptors.Shutdown(device);
    mTransfer.Shutdown();

    mCommandBuffers.Shutdown(device);
    mSync.Shutdown(device);
    mSwapchain.Shutdown(device);
    mMemory.Shutdown();
    mDevice.Shutdown();

    if (mSurface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(mVulkanInstance.GetHandle(), mSurface, nullptr);
        mSurface = VK_NULL_HANDLE;
    }

    mVulkanInstance.Shutdown();
    mWindow.Shutdown();

    LOG_INFO("Cleanup complete");
}

void Application::RecreatePBRPipelines(VkSampleCountFlagBits samples) {
    auto device = mDevice.GetHandle();

    if (mPBRPipeline)         { vkDestroyPipeline(device, mPBRPipeline, nullptr);         mPBRPipeline = VK_NULL_HANDLE; }
    if (mPBRIndirectPipeline) { vkDestroyPipeline(device, mPBRIndirectPipeline, nullptr); mPBRIndirectPipeline = VK_NULL_HANDLE; }

    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding   = 0;
    bindingDesc.stride    = sizeof(MeshVertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDescs[4]{};
    attrDescs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(MeshVertex, position)};
    attrDescs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(MeshVertex, normal)};
    attrDescs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(MeshVertex, texCoord)};
    attrDescs[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(MeshVertex, tangent)};

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = 4;
    vertexInput.pVertexAttributeDescriptions    = attrDescs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates    = dynStates;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = samples;
    if (samples != VK_SAMPLE_COUNT_1_BIT) {
        multisampling.sampleShadingEnable  = VK_TRUE;
        multisampling.minSampleShading     = 0.25f;
    }

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &blendAtt;

    VkFormat hdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkPipelineRenderingCreateInfo renderInfo{};
    renderInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderInfo.colorAttachmentCount    = 1;
    renderInfo.pColorAttachmentFormats = &hdrFormat;
    renderInfo.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

    {
        VkShaderModule vert = mShaders.GetOrLoad("shaders/pbr.vert.spv");
        VkShaderModule frag = mShaders.GetOrLoad("shaders/pbr.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode    = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.lineWidth   = 1.0f;

        VkGraphicsPipelineCreateInfo ci{};
        ci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        ci.pNext               = &renderInfo;
        ci.stageCount          = 2;
        ci.pStages             = stages;
        ci.pVertexInputState   = &vertexInput;
        ci.pInputAssemblyState = &inputAssembly;
        ci.pViewportState      = &viewportState;
        ci.pRasterizationState = &rasterizer;
        ci.pMultisampleState   = &multisampling;
        ci.pDepthStencilState  = &depthStencil;
        ci.pColorBlendState    = &colorBlend;
        ci.pDynamicState       = &dynamicState;
        ci.layout              = mPBRPipelineLayout;

        VK_CHECK(vkCreateGraphicsPipelines(device, mPipelines.GetCache(), 1, &ci, nullptr, &mPBRPipeline));
    }

    {
        VkShaderModule vert = mShaders.GetOrLoad("shaders/pbr_indirect.vert.spv");
        VkShaderModule frag = mShaders.GetOrLoad("shaders/pbr_indirect.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode    = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.lineWidth   = 1.0f;

        VkPipelineDepthStencilStateCreateInfo depthStencilI{};
        depthStencilI.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencilI.depthTestEnable  = VK_TRUE;
        depthStencilI.depthWriteEnable = VK_TRUE;
        depthStencilI.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkGraphicsPipelineCreateInfo ci{};
        ci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        ci.pNext               = &renderInfo;
        ci.stageCount          = 2;
        ci.pStages             = stages;
        ci.pVertexInputState   = &vertexInput;
        ci.pInputAssemblyState = &inputAssembly;
        ci.pViewportState      = &viewportState;
        ci.pRasterizationState = &rasterizer;
        ci.pMultisampleState   = &multisampling;
        ci.pDepthStencilState  = &depthStencilI;
        ci.pColorBlendState    = &colorBlend;
        ci.pDynamicState       = &dynamicState;
        ci.layout              = mPBRIndirectPipelineLayout;

        VK_CHECK(vkCreateGraphicsPipelines(device, mPipelines.GetCache(), 1, &ci, nullptr, &mPBRIndirectPipeline));
    }

    mCurrentMSAA = samples;
    LOG_INFO("PBR pipelines recreated with {}x MSAA", static_cast<int>(samples));
}
