#include "Core/Application.h"
#include "Core/Logger.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <cstring>
#include <fstream>
#include <vector>
#include <array>

// -----------------------------------------------------------------------
// Vertex definition
// -----------------------------------------------------------------------
struct Vertex {
    glm::vec2 position;
    glm::vec3 color;
};

static const std::array<Vertex, 3> kTriangleVertices = {{
    {{ 0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{ 0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
}};

// -----------------------------------------------------------------------
// Utility: read binary file
// -----------------------------------------------------------------------
static std::vector<char> ReadBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open file: {}", path);
        return {};
    }
    auto fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    return buffer;
}

// -----------------------------------------------------------------------
// Utility: insert an image memory barrier (synchronization2)
// -----------------------------------------------------------------------
static void TransitionImage(VkCommandBuffer cmd, VkImage image,
                            VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                            VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                            VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask        = srcStage;
    barrier.srcAccessMask       = srcAccess;
    barrier.dstStageMask        = dstStage;
    barrier.dstAccessMask       = dstAccess;
    barrier.oldLayout           = oldLayout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo dep{};
    dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount  = 1;
    dep.pImageMemoryBarriers     = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

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
    mWindow.SetResizeCallback([this](uint32_t /*w*/, uint32_t /*h*/) {
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

    CreateTriangleResources();
    CreatePipeline();

    LOG_INFO("Vulkan initialization complete");
}

// -----------------------------------------------------------------------
// Triangle vertex buffer (host-visible for simplicity in Phase 1)
// -----------------------------------------------------------------------
void Application::CreateTriangleResources() {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size  = sizeof(kTriangleVertices);
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                      VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo allocationInfo{};
    VK_CHECK(vmaCreateBuffer(mMemory.GetAllocator(), &bufferInfo, &allocInfo,
                             &mVertexBuffer, &mVertexBufferAllocation, &allocationInfo));

    std::memcpy(allocationInfo.pMappedData, kTriangleVertices.data(), sizeof(kTriangleVertices));

    LOG_INFO("Triangle vertex buffer created ({} bytes)", sizeof(kTriangleVertices));
}

// -----------------------------------------------------------------------
// Graphics pipeline (dynamic rendering, no render pass)
// -----------------------------------------------------------------------
void Application::CreatePipeline() {
    auto vertCode = ReadBinaryFile("shaders/triangle.vert.spv");
    auto fragCode = ReadBinaryFile("shaders/triangle.frag.spv");
    if (vertCode.empty() || fragCode.empty()) {
        LOG_ERROR("Failed to load shader files");
        return;
    }

    VkShaderModule vertModule = CreateShaderModule("shaders/triangle.vert.spv");
    VkShaderModule fragModule = CreateShaderModule("shaders/triangle.frag.spv");

    VkPipelineShaderStageCreateInfo shaderStages[2]{};
    shaderStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = vertModule;
    shaderStages[0].pName  = "main";
    shaderStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = fragModule;
    shaderStages[1].pName  = "main";

    // Vertex input
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding   = 0;
    bindingDesc.stride    = sizeof(Vertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 2> attrDescs{};
    attrDescs[0].binding  = 0;
    attrDescs[0].location = 0;
    attrDescs[0].format   = VK_FORMAT_R32G32_SFLOAT;
    attrDescs[0].offset   = offsetof(Vertex, position);
    attrDescs[1].binding  = 0;
    attrDescs[1].location = 1;
    attrDescs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrDescs[1].offset   = offsetof(Vertex, color);

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInput.pVertexAttributeDescriptions    = attrDescs.data();

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
    rasterizer.frontFace   = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

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

    // Pipeline layout (empty — no descriptors or push constants yet)
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    VK_CHECK(vkCreatePipelineLayout(mDevice.GetHandle(), &layoutInfo, nullptr, &mPipelineLayout));

    // Dynamic rendering — specify color attachment format, no render pass
    VkFormat colorFormat = mSwapchain.GetImageFormat();
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;

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
    pipelineInfo.pColorBlendState    = &colorBlend;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = mPipelineLayout;
    pipelineInfo.renderPass          = VK_NULL_HANDLE;

    VK_CHECK(vkCreateGraphicsPipelines(mDevice.GetHandle(), VK_NULL_HANDLE, 1,
                                       &pipelineInfo, nullptr, &mGraphicsPipeline));

    vkDestroyShaderModule(mDevice.GetHandle(), fragModule, nullptr);
    vkDestroyShaderModule(mDevice.GetHandle(), vertModule, nullptr);

    LOG_INFO("Graphics pipeline created");
}

VkShaderModule Application::CreateShaderModule(const std::string& filepath) const {
    auto code = ReadBinaryFile(filepath);
    if (code.empty()) return VK_NULL_HANDLE;

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(mDevice.GetHandle(), &createInfo, nullptr, &module));
    return module;
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
    auto device = mDevice.GetHandle();

    // Pick acquire semaphore from a rolling ring (indexed independently of image).
    VkSemaphore acquireSemaphore = mSync.GetImageAvailableSemaphore(mAcquireSemaphoreIndex);
    mAcquireSemaphoreIndex = (mAcquireSemaphoreIndex + 1) % mSync.GetCount();

    uint32_t imageIndex = 0;
    VkResult result = vkAcquireNextImageKHR(
        device, mSwapchain.GetHandle(), UINT64_MAX,
        acquireSemaphore, VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        RecreateSwapchain();
        return;
    }

    // Wait for any previous work on THIS image to finish.
    // When acquire returns imageIndex, the presentation engine has released it,
    // so renderFinishedSemaphore[imageIndex] from the last present is consumed and safe to reuse.
    VkFence imageFence = mSync.GetFence(imageIndex);
    vkWaitForFences(device, 1, &imageFence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &imageFence);

    auto cmd = mCommandBuffers.Begin(device, imageIndex);
    RecordCommandBuffer(cmd, imageIndex);
    mCommandBuffers.End(imageIndex);

    // Submit — indexed by acquired image
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

    VK_CHECK(vkQueueSubmit(mDevice.GetGraphicsQueue(), 1, &submitInfo, imageFence));

    // Present
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
}

// -----------------------------------------------------------------------
// Record command buffer for one frame
// -----------------------------------------------------------------------
void Application::RecordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkImage swapchainImage = mSwapchain.GetImages()[imageIndex];

    // Undefined -> ColorAttachmentOptimal
    TransitionImage(cmd, swapchainImage,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Begin dynamic rendering
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView   = mSwapchain.GetImageViews()[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.01f, 0.01f, 0.02f, 1.0f}};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea           = {{0, 0}, mSwapchain.GetExtent()};
    renderingInfo.layerCount           = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;

    vkCmdBeginRendering(cmd, &renderingInfo);

    // Viewport & scissor
    VkViewport viewport{};
    viewport.width    = static_cast<float>(mSwapchain.GetExtent().width);
    viewport.height   = static_cast<float>(mSwapchain.GetExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, mSwapchain.GetExtent()};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Draw the triangle
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipeline);
    VkBuffer     vertexBuffers[] = { mVertexBuffer };
    VkDeviceSize offsets[]       = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdDraw(cmd, static_cast<uint32_t>(kTriangleVertices.size()), 1, 0, 0);

    vkCmdEndRendering(cmd);

    // ColorAttachmentOptimal -> PresentSrcKHR
    TransitionImage(cmd, swapchainImage,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

// -----------------------------------------------------------------------
// Swapchain recreation (resize / suboptimal)
// -----------------------------------------------------------------------
void Application::RecreateSwapchain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(mWindow.GetHandle(), &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(mWindow.GetHandle(), &width, &height);
        mWindow.WaitEvents();
    }

    mDevice.WaitIdle();
    mSwapchain.Recreate(mDevice.GetHandle(), mDevice.GetPhysicalDevice(),
                        mSurface, mWindow.GetHandle(), mDevice.GetQueueFamilyIndices());
    LOG_INFO("Swapchain recreated");
}

// -----------------------------------------------------------------------
// Cleanup (reverse order of creation)
// -----------------------------------------------------------------------
void Application::CleanupVulkan() {
    auto device = mDevice.GetHandle();

    if (mVertexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(mMemory.GetAllocator(), mVertexBuffer, mVertexBufferAllocation);
    }
    if (mGraphicsPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, mGraphicsPipeline, nullptr);
    }
    if (mPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, mPipelineLayout, nullptr);
    }

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
