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

// Push constants sent to vertex + fragment shaders
struct MeshPushConstants {
    glm::mat4 mvp;
    uint32_t  textureIndex;
};

// =======================================================================
// Application
// =======================================================================

void Application::Run() {
    InitWindow();
    InitVulkan();
    MainLoop();
    CleanupVulkan();
}

// -----------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------
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
    mSync.Initialize(mDevice.GetHandle(), mSwapchain.GetImageCount());
    mCommandBuffers.Initialize(mDevice.GetHandle(), mDevice.GetQueueFamilyIndices().graphicsFamily,
                               mSwapchain.GetImageCount());

    mImageFences.resize(mSwapchain.GetImageCount(), VK_NULL_HANDLE);

    // Resource managers
    mTransfer.Initialize(mDevice.GetHandle(), mDevice.GetQueueFamilyIndices().graphicsFamily,
                         mDevice.GetGraphicsQueue());
    mDescriptors.Initialize(mDevice.GetHandle());
    mShaders.Initialize(mDevice.GetHandle());
    mPipelines.Initialize(mDevice.GetHandle());
    mPipelines.LoadCache("pipeline_cache.bin");

    LoadScene();
    CreateDepthBuffer();
    CreatePipeline();

    LOG_INFO("Vulkan initialization complete (Phase 2)");
}

// -----------------------------------------------------------------------
// Scene loading: glTF or procedural cube fallback
// -----------------------------------------------------------------------
void Application::LoadScene() {
    bool loaded = false;

    // Try common asset paths
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

    if (!loaded) {
        LOG_INFO("No glTF model found, generating procedural cube");
        ModelLoader::GenerateProceduralCube(mModelData);
    }

    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();

    // Upload textures
    for (auto& texData : mModelData.textures) {
        VulkanImage gpuTex;
        gpuTex.CreateTexture2D(allocator, device, mTransfer,
                               texData.width, texData.height,
                               VK_FORMAT_R8G8B8A8_SRGB, texData.pixels.data());

        uint32_t descIdx = mDescriptors.AllocateTextureIndex();
        mDescriptors.UpdateTexture(device, descIdx, gpuTex.GetView(),
                                   mDescriptors.GetDefaultSampler());

        mGPUTextures.push_back(std::move(gpuTex));
        mTextureDescriptorIndices.push_back(descIdx);
    }

    // Upload meshes
    for (auto& meshData : mModelData.meshes) {
        GPUMesh gpu;
        gpu.materialIndex = meshData.materialIndex;
        gpu.indexCount     = static_cast<uint32_t>(meshData.indices.size());

        gpu.vertexBuffer.CreateDeviceLocal(allocator, mTransfer,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            meshData.vertices.data(),
            meshData.vertices.size() * sizeof(MeshVertex));

        gpu.indexBuffer.CreateDeviceLocal(allocator, mTransfer,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            meshData.indices.data(),
            meshData.indices.size() * sizeof(uint32_t));

        // Resolve which bindless texture index to use
        gpu.textureIndex = 0;
        if (gpu.materialIndex >= 0 &&
            gpu.materialIndex < static_cast<int>(mModelData.materials.size()))
        {
            int texIdx = mModelData.materials[gpu.materialIndex].baseColorTextureIndex;
            if (texIdx >= 0 && texIdx < static_cast<int>(mTextureDescriptorIndices.size()))
                gpu.textureIndex = mTextureDescriptorIndices[texIdx];
        }

        mGPUMeshes.push_back(std::move(gpu));
    }

    LOG_INFO("Scene loaded: {} GPU meshes, {} GPU textures",
             mGPUMeshes.size(), mGPUTextures.size());
}

// -----------------------------------------------------------------------
// Depth buffer
// -----------------------------------------------------------------------
void Application::CreateDepthBuffer() {
    auto extent = mSwapchain.GetExtent();
    mDepthImage.CreateDepth(mMemory.GetAllocator(), mDevice.GetHandle(),
                            extent.width, extent.height);
}

// -----------------------------------------------------------------------
// Graphics pipeline
// -----------------------------------------------------------------------
void Application::CreatePipeline() {
    auto device = mDevice.GetHandle();

    VkShaderModule vertModule = mShaders.GetOrLoad("shaders/mesh.vert.spv");
    VkShaderModule fragModule = mShaders.GetOrLoad("shaders/mesh.frag.spv");

    VkPipelineShaderStageCreateInfo shaderStages[2]{};
    shaderStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = vertModule;
    shaderStages[0].pName  = "main";
    shaderStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = fragModule;
    shaderStages[1].pName  = "main";

    // Vertex input: MeshVertex { vec3 pos, vec3 normal, vec2 uv }
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding   = 0;
    bindingDesc.stride    = sizeof(MeshVertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDescs[3]{};
    attrDescs[0].binding  = 0;
    attrDescs[0].location = 0;
    attrDescs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrDescs[0].offset   = offsetof(MeshVertex, position);
    attrDescs[1].binding  = 0;
    attrDescs[1].location = 1;
    attrDescs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrDescs[1].offset   = offsetof(MeshVertex, normal);
    attrDescs[2].binding  = 0;
    attrDescs[2].location = 2;
    attrDescs[2].format   = VK_FORMAT_R32G32_SFLOAT;
    attrDescs[2].offset   = offsetof(MeshVertex, texCoord);

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

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode    = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &blendAttachment;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates    = dynamicStates;

    // Pipeline layout: one bindless descriptor set + push constants
    VkDescriptorSetLayout setLayouts[] = { mDescriptors.GetLayout() };

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(MeshPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = setLayouts;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pushRange;
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &mPipelineLayout));

    // Dynamic rendering: color + depth
    VkFormat colorFormat = mSwapchain.GetImageFormat();
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;
    renderingInfo.depthAttachmentFormat   = depthFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext               = &renderingInfo;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = shaderStages;
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlend;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = mPipelineLayout;
    pipelineInfo.renderPass          = VK_NULL_HANDLE;

    VK_CHECK(vkCreateGraphicsPipelines(device, mPipelines.GetCache(), 1,
                                       &pipelineInfo, nullptr, &mGraphicsPipeline));

    LOG_INFO("Graphics pipeline created (Phase 2, with depth + bindless)");
}

// -----------------------------------------------------------------------
// Main loop
// -----------------------------------------------------------------------
void Application::MainLoop() {
    LOG_INFO("Entering main loop");
    while (!mWindow.ShouldClose()) {
        mWindow.PollEvents();
        DrawFrame();
    }
    mDevice.WaitIdle();
}

// -----------------------------------------------------------------------
// Draw frame
// -----------------------------------------------------------------------
void Application::DrawFrame() {
    auto device    = mDevice.GetHandle();
    auto frameFence = mSync.GetFence(mFrameIndex);

    vkWaitForFences(device, 1, &frameFence, VK_TRUE, UINT64_MAX);

    VkSemaphore acquireSemaphore = mSync.GetImageAvailableSemaphore(mFrameIndex);
    uint32_t imageIndex = 0;
    VkResult result = vkAcquireNextImageKHR(
        device, mSwapchain.GetHandle(), UINT64_MAX,
        acquireSemaphore, VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        RecreateSwapchain();
        return;
    }

    if (mImageFences[imageIndex] != VK_NULL_HANDLE && mImageFences[imageIndex] != frameFence) {
        vkWaitForFences(device, 1, &mImageFences[imageIndex], VK_TRUE, UINT64_MAX);
    }
    mImageFences[imageIndex] = frameFence;

    vkResetFences(device, 1, &frameFence);

    auto cmd = mCommandBuffers.Begin(device, imageIndex);
    RecordCommandBuffer(cmd, imageIndex);
    mCommandBuffers.End(imageIndex);

    VkSemaphore          waitSemaphores[]   = { acquireSemaphore };
    VkPipelineStageFlags waitStages[]       = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore          signalSemaphores[] = { mSync.GetRenderFinishedSemaphore(imageIndex) };
    VkCommandBuffer      cmdBuf             = mCommandBuffers.Get(imageIndex);

    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = waitSemaphores;
    submitInfo.pWaitDstStageMask    = waitStages;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &cmdBuf;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = signalSemaphores;

    VK_CHECK(vkQueueSubmit(mDevice.GetGraphicsQueue(), 1, &submitInfo, frameFence));

    VkSwapchainKHR swapchains[] = { mSwapchain.GetHandle() };
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = signalSemaphores;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = swapchains;
    presentInfo.pImageIndices      = &imageIndex;

    result = vkQueuePresentKHR(mDevice.GetPresentQueue(), &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || mFramebufferResized) {
        mFramebufferResized = false;
        RecreateSwapchain();
    }

    mFrameIndex = (mFrameIndex + 1) % mSync.GetCount();
}

// -----------------------------------------------------------------------
// Record command buffer
// -----------------------------------------------------------------------
void Application::RecordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkImage swapchainImage = mSwapchain.GetImages()[imageIndex];
    VkExtent2D extent      = mSwapchain.GetExtent();

    // Color: Undefined -> ColorAttachmentOptimal
    TransitionImage(cmd, swapchainImage,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Depth: Undefined -> DepthAttachmentOptimal
    TransitionImage(cmd, mDepthImage.GetImage(),
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_ASPECT_DEPTH_BIT);

    // Begin dynamic rendering
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView   = mSwapchain.GetImageViews()[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.01f, 0.01f, 0.02f, 1.0f}};

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView   = mDepthImage.GetView();
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea           = {{0, 0}, extent};
    renderingInfo.layerCount           = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;
    renderingInfo.pDepthAttachment     = &depthAttachment;

    vkCmdBeginRendering(cmd, &renderingInfo);

    // Viewport & scissor
    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipeline);
    VkDescriptorSet bindlessSet = mDescriptors.GetSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            mPipelineLayout, 0, 1,
                            &bindlessSet, 0, nullptr);

    // Compute MVP (simple orbital camera)
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    mRotationAngle += 0.005f;

    glm::mat4 model = glm::rotate(glm::mat4(1.0f), mRotationAngle, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 view  = glm::lookAt(glm::vec3(0.0f, 1.0f, 3.0f),
                                   glm::vec3(0.0f, 0.0f, 0.0f),
                                   glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 proj  = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    proj[1][1] *= -1.0f; // Flip Y for Vulkan's coordinate system

    glm::mat4 mvp = proj * view * model;

    // Draw each submesh
    for (const auto& mesh : mGPUMeshes) {
        MeshPushConstants pc{};
        pc.mvp          = mvp;
        pc.textureIndex = mesh.textureIndex;

        vkCmdPushConstants(cmd, mPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(MeshPushConstants), &pc);

        VkBuffer     vertexBuffers[] = { mesh.vertexBuffer.GetHandle() };
        VkDeviceSize offsets[]       = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.GetHandle(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);

    // ColorAttachmentOptimal -> PresentSrcKHR
    TransitionImage(cmd, swapchainImage,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

// -----------------------------------------------------------------------
// Swapchain recreation
// -----------------------------------------------------------------------
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

// -----------------------------------------------------------------------
// Cleanup
// -----------------------------------------------------------------------
void Application::CleanupVulkan() {
    auto device    = mDevice.GetHandle();
    auto allocator = mMemory.GetAllocator();

    // Pipeline cache persistence
    mPipelines.SaveCache("pipeline_cache.bin");

    // Scene resources
    for (auto& mesh : mGPUMeshes) {
        mesh.vertexBuffer.Destroy(allocator);
        mesh.indexBuffer.Destroy(allocator);
    }
    mGPUMeshes.clear();

    for (auto& tex : mGPUTextures) {
        tex.Destroy(allocator, device);
    }
    mGPUTextures.clear();

    for (uint32_t idx : mTextureDescriptorIndices)
        mDescriptors.FreeTextureIndex(idx);
    mTextureDescriptorIndices.clear();

    mDepthImage.Destroy(allocator, device);

    if (mGraphicsPipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, mGraphicsPipeline, nullptr);
    if (mPipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, mPipelineLayout, nullptr);

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
