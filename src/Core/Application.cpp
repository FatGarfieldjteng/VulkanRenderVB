#include "Core/Application.h"
#include "Core/Logger.h"
#include "RHI/VulkanUtils.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstring>
#include <algorithm>
#include <filesystem>
#include <array>

struct PBRPushConstants {
    glm::mat4 model;
    uint32_t  materialIndex;
};

// =======================================================================
void Application::Run() {
    InitWindow();
    InitVulkan();
    MainLoop();
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
}

void Application::InitVulkan() {
    mVulkanInstance.Initialize("VulkanRenderVB");
    mSurface = mVulkanInstance.CreateSurface(mWindow.GetHandle());
    mDevice.Initialize(mVulkanInstance.GetHandle(), mSurface);
    mMemory.Initialize(mVulkanInstance.GetHandle(), mDevice.GetPhysicalDevice(), mDevice.GetHandle());
    mSwapchain.Initialize(mDevice.GetHandle(), mDevice.GetPhysicalDevice(),
                          mSurface, mWindow.GetHandle(), mDevice.GetQueueFamilyIndices());
    mSync.Initialize(mDevice.GetHandle(), FRAMES_IN_FLIGHT);
    mCommandBuffers.Initialize(mDevice.GetHandle(), mDevice.GetQueueFamilyIndices().graphicsFamily,
                               mSwapchain.GetImageCount());
    mImageFences.resize(mSwapchain.GetImageCount(), VK_NULL_HANDLE);

    mTransfer.Initialize(mDevice.GetHandle(), mDevice.GetQueueFamilyIndices().graphicsFamily,
                         mDevice.GetGraphicsQueue());
    mDescriptors.Initialize(mDevice.GetHandle());
    mShaders.Initialize(mDevice.GetHandle());
    mPipelines.Initialize(mDevice.GetHandle());
    mPipelines.LoadCache("pipeline_cache.bin");

    mCSM.Initialize(mMemory.GetAllocator(), mDevice.GetHandle());

    CreateDefaultTextures();
    LoadScene();
    CreateDepthBuffer();
    CreateFrameDescriptors();
    CreatePipelines();

    mCamera.Init(glm::vec3(0, 3, 8), glm::vec3(0, 0, 0), 45.0f, 0.1f, 150.0f);
    mLastFrameTime = glfwGetTime();

    LOG_INFO("Vulkan initialization complete (Phase 3)");
}

// =======================================================================
// Default textures (1x1 white + black for material fallbacks)
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

    LOG_INFO("Default textures created (white={}, black={})", mWhiteTexDescIdx, mBlackTexDescIdx);
}

// =======================================================================
// Scene loading
// =======================================================================
void Application::LoadScene() {
    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();

    bool loaded = false;
    const char* modelPaths[] = {
        "assets/DamagedHelmet.glb",
        "assets/DamagedHelmet/DamagedHelmet.gltf",
        "assets/model.glb",
    };
    for (const char* p : modelPaths) {
        if (std::filesystem::exists(p)) {
            loaded = ModelLoader::LoadGLTF(p, mModelData);
            if (loaded) break;
        }
    }

    if (loaded) {
        // Determine sRGB vs UNORM per texture based on material usage
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

        for (auto& meshData : mModelData.meshes) {
            GPUMesh gpu;
            gpu.indexCount = static_cast<uint32_t>(meshData.indices.size());
            gpu.vertexBuffer.CreateDeviceLocal(allocator, mTransfer,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                meshData.vertices.data(), meshData.vertices.size() * sizeof(MeshVertex));
            gpu.indexBuffer.CreateDeviceLocal(allocator, mTransfer,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                meshData.indices.data(), meshData.indices.size() * sizeof(uint32_t));
            mGPUMeshes.push_back(std::move(gpu));
        }

        auto resolveIdx = [&](int texIdx, uint32_t fallback) -> uint32_t {
            if (texIdx >= 0 && texIdx < static_cast<int>(mTextureDescriptorIndices.size()))
                return mTextureDescriptorIndices[texIdx];
            return fallback;
        };

        for (const auto& mat : mModelData.materials) {
            GPUMaterialData g{};
            g.baseColorFactor        = mat.baseColorFactor;
            g.metallicFactor         = mat.metallicFactor;
            g.roughnessFactor        = mat.roughnessFactor;
            g.baseColorTexIdx        = resolveIdx(mat.baseColorTextureIndex, mWhiteTexDescIdx);
            g.normalTexIdx           = resolveIdx(mat.normalTextureIndex, mWhiteTexDescIdx);
            g.metallicRoughnessTexIdx = resolveIdx(mat.metallicRoughnessTextureIndex, mWhiteTexDescIdx);
            g.aoTexIdx               = resolveIdx(mat.occlusionTextureIndex, mWhiteTexDescIdx);
            g.emissiveTexIdx         = resolveIdx(mat.emissiveTextureIndex, mBlackTexDescIdx);
            mGPUMaterials.push_back(g);
        }

        for (size_t i = 0; i < mModelData.meshes.size(); i++) {
            auto& obj = mScene.AddObject();
            obj.meshIndex     = static_cast<int>(i);
            obj.materialIndex = std::max(0, mModelData.meshes[i].materialIndex);
        }

    } else {
        LOG_INFO("No glTF model found, generating procedural scene");

        // Checkerboard texture for the ground
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

        // Mesh 0: ground plane
        {
            MeshData groundMesh;
            ModelLoader::GenerateGroundPlane(groundMesh, 20.0f);
            GPUMesh gpu;
            gpu.indexCount = static_cast<uint32_t>(groundMesh.indices.size());
            gpu.vertexBuffer.CreateDeviceLocal(allocator, mTransfer,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                groundMesh.vertices.data(), groundMesh.vertices.size() * sizeof(MeshVertex));
            gpu.indexBuffer.CreateDeviceLocal(allocator, mTransfer,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                groundMesh.indices.data(), groundMesh.indices.size() * sizeof(uint32_t));
            mGPUMeshes.push_back(std::move(gpu));
        }

        // Mesh 1: cube
        {
            ModelData cubeData;
            ModelLoader::GenerateProceduralCube(cubeData);
            const auto& cm = cubeData.meshes[0];
            GPUMesh gpu;
            gpu.indexCount = static_cast<uint32_t>(cm.indices.size());
            gpu.vertexBuffer.CreateDeviceLocal(allocator, mTransfer,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                cm.vertices.data(), cm.vertices.size() * sizeof(MeshVertex));
            gpu.indexBuffer.CreateDeviceLocal(allocator, mTransfer,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                cm.indices.data(), cm.indices.size() * sizeof(uint32_t));
            mGPUMeshes.push_back(std::move(gpu));
        }

        // Material helper
        auto pushMat = [&](glm::vec3 color, float metallic, float roughness, uint32_t baseTexIdx) {
            GPUMaterialData m{};
            m.baseColorFactor         = glm::vec4(color, 1.0f);
            m.metallicFactor          = metallic;
            m.roughnessFactor         = roughness;
            m.baseColorTexIdx         = baseTexIdx;
            m.normalTexIdx            = mWhiteTexDescIdx;
            m.metallicRoughnessTexIdx = mWhiteTexDescIdx;
            m.aoTexIdx                = mWhiteTexDescIdx;
            m.emissiveTexIdx          = mBlackTexDescIdx;
            mGPUMaterials.push_back(m);
        };

        // Mat 0: grey ground (checkerboard)
        pushMat({0.5f, 0.5f, 0.5f}, 0.0f, 0.9f, checkerDescIdx);
        // Mat 1: red dielectric
        pushMat({0.8f, 0.15f, 0.15f}, 0.0f, 0.3f, mWhiteTexDescIdx);
        // Mat 2: green metallic
        pushMat({0.1f, 0.7f, 0.1f}, 0.9f, 0.15f, mWhiteTexDescIdx);
        // Mat 3: blue rough
        pushMat({0.15f, 0.15f, 0.8f}, 0.0f, 0.7f, mWhiteTexDescIdx);
        // Mat 4: gold
        pushMat({1.0f, 0.766f, 0.336f}, 1.0f, 0.1f, mWhiteTexDescIdx);

        // Scene objects
        { auto& o = mScene.AddObject(); o.meshIndex = 0; o.materialIndex = 0; }
        { auto& o = mScene.AddObject(); o.meshIndex = 1; o.materialIndex = 1;
          o.position = {-3.0f, 0.5f, 0.0f}; }
        { auto& o = mScene.AddObject(); o.meshIndex = 1; o.materialIndex = 2;
          o.position = {-1.0f, 0.5f, -1.0f}; }
        { auto& o = mScene.AddObject(); o.meshIndex = 1; o.materialIndex = 3;
          o.position = {1.0f, 0.5f, 0.0f}; }
        { auto& o = mScene.AddObject(); o.meshIndex = 1; o.materialIndex = 4;
          o.position = {3.0f, 0.5f, 1.0f}; }
    }

    if (mGPUMaterials.empty()) {
        GPUMaterialData def{};
        def.baseColorTexIdx         = mWhiteTexDescIdx;
        def.normalTexIdx            = mWhiteTexDescIdx;
        def.metallicRoughnessTexIdx = mWhiteTexDescIdx;
        def.aoTexIdx                = mWhiteTexDescIdx;
        def.emissiveTexIdx          = mBlackTexDescIdx;
        mGPUMaterials.push_back(def);
    }

    mMaterialSSBO.CreateDeviceLocal(allocator, mTransfer,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        mGPUMaterials.data(), mGPUMaterials.size() * sizeof(GPUMaterialData));

    LOG_INFO("Scene loaded: {} meshes, {} textures, {} materials, {} objects",
             mGPUMeshes.size(), mGPUTextures.size(), mGPUMaterials.size(),
             mScene.GetObjects().size());
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
// Frame descriptors (set 1: UBO + material SSBO + shadow map)
// =======================================================================
void Application::CreateFrameDescriptors() {
    auto device     = mDevice.GetHandle();
    uint32_t frames = FRAMES_IN_FLIGHT;

    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[2].binding         = 2;
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = 3;
    layoutCI.pBindings    = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &mFrameSetLayout));

    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         frames },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         frames },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, frames },
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

        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = mFrameUBOs[i].GetHandle();
        uboInfo.offset = 0;
        uboInfo.range  = sizeof(FrameData);

        VkDescriptorBufferInfo ssboInfo{};
        ssboInfo.buffer = mMaterialSSBO.GetHandle();
        ssboInfo.offset = 0;
        ssboInfo.range  = mMaterialSSBO.GetSize();

        VkDescriptorImageInfo shadowInfo{};
        shadowInfo.sampler     = mCSM.GetShadowSampler();
        shadowInfo.imageView   = mCSM.GetArrayView();
        shadowInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet writes[3]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = mFrameDescSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo     = &uboInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = mFrameDescSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo     = &ssboInfo;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = mFrameDescSets[i];
        writes[2].dstBinding      = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo      = &shadowInfo;

        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    }

    LOG_INFO("Frame descriptors created ({} sets)", frames);
}

// =======================================================================
// Pipelines (PBR + Shadow)
// =======================================================================
void Application::CreatePipelines() {
    auto device = mDevice.GetHandle();

    // Shared vertex input
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding   = 0;
    bindingDesc.stride    = sizeof(MeshVertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDescs[3]{};
    attrDescs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, position)};
    attrDescs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, normal)};
    attrDescs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(MeshVertex, texCoord)};

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = 3;
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

        VkFormat colorFormat = mSwapchain.GetImageFormat();
        VkPipelineRenderingCreateInfo renderInfo{};
        renderInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderInfo.colorAttachmentCount    = 1;
        renderInfo.pColorAttachmentFormats = &colorFormat;
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

    LOG_INFO("Pipelines created (PBR + Shadow)");
}

// =======================================================================
// Main loop
// =======================================================================
void Application::MainLoop() {
    LOG_INFO("Entering main loop (Phase 3)");
    while (!mWindow.ShouldClose()) {
        mWindow.PollEvents();

        double now = glfwGetTime();
        float dt   = static_cast<float>(now - mLastFrameTime);
        mLastFrameTime = now;
        dt = std::min(dt, 0.1f);

        mCamera.Update(mWindow, dt);
        mWindow.ResetInputDeltas();

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

    // Update CSM cascades
    VkExtent2D extent = mSwapchain.GetExtent();
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    glm::mat4 view = mCamera.GetViewMatrix();
    glm::mat4 proj = mCamera.GetProjectionMatrix(aspect);
    const auto& sun = mScene.GetSun();
    mCSM.Update(view, proj, mCamera.GetNear(), mCamera.GetFar(), sun.direction);

    // Update per-frame UBO
    FrameData fd{};
    fd.view           = view;
    fd.projection     = proj;
    fd.viewProjection = proj * view;
    fd.cameraPos      = glm::vec4(mCamera.GetPosition(), 0.0f);
    fd.sunDirection   = glm::vec4(sun.direction, 0.0f);
    fd.sunColor       = glm::vec4(sun.color, sun.intensity);
    for (uint32_t c = 0; c < CascadedShadowMap::CASCADE_COUNT; c++)
        fd.cascadeViewProj[c] = mCSM.GetViewProj(c);
    fd.cascadeSplits = mCSM.GetSplits();
    std::memcpy(mFrameUBOs[mFrameIndex].GetMappedData(), &fd, sizeof(FrameData));

    auto cmd = mCommandBuffers.Begin(device, imageIndex);
    RecordCommandBuffer(cmd, imageIndex);
    mCommandBuffers.End(imageIndex);

    VkSemaphore          waitSems[]   = { acquireSem };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore          signalSems[] = { mSync.GetRenderFinishedSemaphore(mFrameIndex) };
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
}

// =======================================================================
// Record command buffer (shadow passes + main PBR pass)
// =======================================================================
void Application::RecordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    constexpr uint32_t SD = CascadedShadowMap::SHADOW_DIM;
    constexpr uint32_t CC = CascadedShadowMap::CASCADE_COUNT;

    // ===== SHADOW PASSES =====

    TransitionImage(cmd, mCSM.GetImage(),
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, CC);

    for (uint32_t c = 0; c < CC; c++) {
        VkRenderingAttachmentInfo depthAtt{};
        depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAtt.imageView   = mCSM.GetLayerView(c);
        depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        depthAtt.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo ri{};
        ri.sType            = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea       = {{0, 0}, {SD, SD}};
        ri.layerCount       = 1;
        ri.pDepthAttachment = &depthAtt;

        vkCmdBeginRendering(cmd, &ri);

        VkViewport vp{0, 0, static_cast<float>(SD), static_cast<float>(SD), 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, {SD, SD}};
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mShadowPipeline);

        for (const auto& obj : mScene.GetObjects()) {
            if (obj.meshIndex < 0 || obj.meshIndex >= static_cast<int>(mGPUMeshes.size())) continue;
            glm::mat4 mvp = mCSM.GetViewProj(c) * obj.GetModelMatrix();
            vkCmdPushConstants(cmd, mShadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(glm::mat4), &mvp);
            const auto& mesh = mGPUMeshes[obj.meshIndex];
            VkBuffer     vb[]  = { mesh.vertexBuffer.GetHandle() };
            VkDeviceSize off[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
            vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.GetHandle(), 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
        }

        vkCmdEndRendering(cmd);
    }

    TransitionImage(cmd, mCSM.GetImage(),
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, CC);

    // ===== MAIN PBR PASS =====

    VkImage    swapImg = mSwapchain.GetImages()[imageIndex];
    VkExtent2D extent  = mSwapchain.GetExtent();

    TransitionImage(cmd, swapImg,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    TransitionImage(cmd, mDepthImage.GetImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView   = mSwapchain.GetImageViews()[imageIndex];
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue.color = {{0.02f, 0.02f, 0.04f, 1.0f}};

    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView   = mDepthImage.GetView();
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea           = {{0, 0}, extent};
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &colorAtt;
    ri.pDepthAttachment     = &depthAtt;

    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mPBRPipeline);

    VkDescriptorSet bindlessSet = mDescriptors.GetSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            mPBRPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            mPBRPipelineLayout, 1, 1, &mFrameDescSets[mFrameIndex], 0, nullptr);

    for (const auto& obj : mScene.GetObjects()) {
        if (obj.meshIndex < 0 || obj.meshIndex >= static_cast<int>(mGPUMeshes.size())) continue;

        int matIdx = std::clamp(obj.materialIndex, 0, static_cast<int>(mGPUMaterials.size()) - 1);

        PBRPushConstants pc{};
        pc.model         = obj.GetModelMatrix();
        pc.materialIndex = static_cast<uint32_t>(matIdx);

        vkCmdPushConstants(cmd, mPBRPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, static_cast<uint32_t>(sizeof(glm::mat4) + sizeof(uint32_t)), &pc);

        const auto& mesh = mGPUMeshes[obj.meshIndex];
        VkBuffer     vb[]  = { mesh.vertexBuffer.GetHandle() };
        VkDeviceSize off[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.GetHandle(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);

    TransitionImage(cmd, swapImg,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
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

    LOG_INFO("Swapchain recreated");
}

// =======================================================================
// Cleanup
// =======================================================================
void Application::CleanupVulkan() {
    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();

    mPipelines.SaveCache("pipeline_cache.bin");

    for (auto& mesh : mGPUMeshes) {
        mesh.vertexBuffer.Destroy(allocator);
        mesh.indexBuffer.Destroy(allocator);
    }
    mGPUMeshes.clear();

    for (auto& tex : mGPUTextures)
        tex.Destroy(allocator, device);
    mGPUTextures.clear();

    for (uint32_t idx : mTextureDescriptorIndices)
        mDescriptors.FreeTextureIndex(idx);
    mTextureDescriptorIndices.clear();

    mWhiteTexture.Destroy(allocator, device);
    mBlackTexture.Destroy(allocator, device);
    mDescriptors.FreeTextureIndex(mWhiteTexDescIdx);
    mDescriptors.FreeTextureIndex(mBlackTexDescIdx);

    mMaterialSSBO.Destroy(allocator);

    for (auto& ubo : mFrameUBOs)
        ubo.Destroy(allocator);
    mFrameUBOs.clear();

    if (mFrameDescPool)   vkDestroyDescriptorPool(device, mFrameDescPool, nullptr);
    if (mFrameSetLayout)  vkDestroyDescriptorSetLayout(device, mFrameSetLayout, nullptr);
    mFrameDescPool  = VK_NULL_HANDLE;
    mFrameSetLayout = VK_NULL_HANDLE;

    mCSM.Shutdown(allocator, device);
    mDepthImage.Destroy(allocator, device);

    if (mPBRPipeline)          vkDestroyPipeline(device, mPBRPipeline, nullptr);
    if (mPBRPipelineLayout)    vkDestroyPipelineLayout(device, mPBRPipelineLayout, nullptr);
    if (mShadowPipeline)       vkDestroyPipeline(device, mShadowPipeline, nullptr);
    if (mShadowPipelineLayout) vkDestroyPipelineLayout(device, mShadowPipelineLayout, nullptr);

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
