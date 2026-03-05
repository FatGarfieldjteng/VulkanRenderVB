#include "VisualUI/DebugUI.h"
#include "VisualUI/GPUProfiler.h"
#include "VisualUI/PipelineStatistics.h"
#include "Scene/ECS.h"
#include "Scene/Scene.h"
#include "Core/Logger.h"

#include <volk.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstring>
#include <cmath>

static void CheckVkResult(VkResult err) {
    if (err != VK_SUCCESS) {
        LOG_ERROR("ImGui Vulkan error: {}", static_cast<int>(err));
    }
}

void DebugUI::Initialize(VkInstance instance, VkDevice device, VkPhysicalDevice physicalDevice,
                          uint32_t graphicsFamily, VkQueue graphicsQueue,
                          GLFWwindow* window, VkFormat swapchainFormat,
                          uint32_t framesInFlight)
{
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 },
        { VK_DESCRIPTOR_TYPE_SAMPLER, 50 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 50 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 50 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 },
    };
    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCI.maxSets       = 100;
    poolCI.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
    poolCI.pPoolSizes    = poolSizes;
    vkCreateDescriptorPool(device, &poolCI, nullptr, &mDescriptorPool);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.IniFilename = "imgui_layout.ini";

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding   = 4.0f;
    style.FrameRounding    = 2.0f;
    style.GrabRounding     = 2.0f;
    style.ScrollbarRounding = 3.0f;
    style.Colors[ImGuiCol_WindowBg].w = 0.92f;

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_LoadFunctions(VK_API_VERSION_1_3,
        [](const char* name, void* ud) -> PFN_vkVoidFunction {
            return vkGetInstanceProcAddr(static_cast<VkInstance>(ud), name);
        }, instance);

    VkPipelineRenderingCreateInfoKHR renderingCI{};
    renderingCI.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    renderingCI.colorAttachmentCount    = 1;
    renderingCI.pColorAttachmentFormats = &swapchainFormat;

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion      = VK_API_VERSION_1_3;
    initInfo.Instance        = instance;
    initInfo.PhysicalDevice  = physicalDevice;
    initInfo.Device          = device;
    initInfo.QueueFamily     = graphicsFamily;
    initInfo.Queue           = graphicsQueue;
    initInfo.DescriptorPool  = mDescriptorPool;
    initInfo.MinImageCount   = framesInFlight;
    initInfo.ImageCount      = framesInFlight;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = renderingCI;
    initInfo.CheckVkResultFn = CheckVkResult;
    initInfo.UseDynamicRendering = true;

    ImGui_ImplVulkan_Init(&initInfo);

    std::memset(mFpsHistory, 0, sizeof(mFpsHistory));

    LOG_INFO("DebugUI initialized (ImGui {} with docking)", IMGUI_VERSION);
}

void DebugUI::Shutdown(VkDevice device) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (mDescriptorPool) {
        vkDestroyDescriptorPool(device, mDescriptorPool, nullptr);
        mDescriptorPool = VK_NULL_HANDLE;
    }
}

void DebugUI::BeginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void DebugUI::BuildUI(float deltaTime, const GPUProfiler* profiler, const PipelineStatistics* pipeStats,
                      Registry* registry, std::vector<GPUMaterialData>* materials) {
    ImGuiID dockID = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                                    ImGuiDockNodeFlags_PassthruCentralNode);
    (void)dockID;

    DrawMainMenuBar();
    DrawRenderSettingsPanel(deltaTime);
    DrawSceneHierarchyPanel(registry);
    DrawMaterialEditorPanel(materials);
    DrawProfilerPanel(profiler);
    DrawPipelineStatsPanel(pipeStats);

    if (mState.showDemoWindow)
        ImGui::ShowDemoWindow(&mState.showDemoWindow);
}

void DebugUI::EndFrame() {
    ImGui::Render();
}

void DebugUI::Render(VkCommandBuffer cmd, VkImageView colorView, VkExtent2D extent) {
    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView   = colorView;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea           = {{0, 0}, extent};
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &colorAtt;

    vkCmdBeginRendering(cmd, &ri);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

bool DebugUI::WantCaptureMouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

bool DebugUI::WantCaptureKeyboard() const {
    return ImGui::GetIO().WantCaptureKeyboard;
}

void DebugUI::DrawMainMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Demo Window", nullptr, &mState.showDemoWindow);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void DebugUI::DrawRenderSettingsPanel(float deltaTime) {
    float fps = (deltaTime > 0.0f) ? 1.0f / deltaTime : 0.0f;

    constexpr float smoothAlpha = 0.05f;
    mFpsSmoothed = mFpsSmoothed * (1.0f - smoothAlpha) + fps * smoothAlpha;
    if (mFpsSmoothed == 0.0f) mFpsSmoothed = fps;

    mFpsHistory[mFpsHistoryOffset] = mFpsSmoothed;
    mFpsHistoryOffset = (mFpsHistoryOffset + 1) % 120;

    ImGui::Begin("Render Settings");

    ImGui::Text("FPS: %.1f  (%.2f ms)", mFpsSmoothed, 1000.0f / std::max(mFpsSmoothed, 0.001f));
    ImGui::PlotLines("##fps", mFpsHistory, 120, mFpsHistoryOffset, nullptr, 0.0f, 240.0f, ImVec2(0, 50));

    ImGui::Separator();
    ImGui::Text("GPU-Driven Rendering");
    ImGui::Checkbox("GPU Driven", &mState.gpuDriven);
    ImGui::Checkbox("Occlusion Culling", &mState.occlusionCulling);
    ImGui::SliderFloat("Occluder Ratio", &mState.occluderRatio, 0.05f, 1.0f, "%.2f");

    ImGui::Separator();
    ImGui::Checkbox("Pipeline Statistics", &mState.pipelineStatsEnabled);

    ImGui::Separator();
    ImGui::Text("Debug Visualization");
    const char* visModes[] = {
        "None", "Wireframe", "World Normals", "Tangent Normals",
        "UVs", "Mip Level", "Overdraw", "Cascade Coloring", "Depth Buffer"
    };
    int current = static_cast<int>(mState.visMode);
    if (ImGui::Combo("Mode", &current, visModes, static_cast<int>(DebugUIState::VisMode::Count)))
        mState.visMode = static_cast<DebugUIState::VisMode>(current);

    ImGui::End();
}

void DebugUI::DrawProfilerPanel(const GPUProfiler* profiler) {
    if (!profiler) return;

    ImGui::Begin("GPU Profiler");
    ImGui::Text("Total GPU: %.2f ms", profiler->GetTotalMs());
    ImGui::Separator();

    const auto& results = profiler->GetResults();
    for (const auto& r : results) {
        float frac = (profiler->GetTotalMs() > 0.001f)
                     ? r.durationMs / profiler->GetTotalMs()
                     : 0.0f;
        ImGui::ProgressBar(frac, ImVec2(-1, 0),
            (r.name + ": " + std::to_string(r.durationMs).substr(0,5) + " ms").c_str());
    }

    if (ImGui::Button("Export CSV"))
        profiler->ExportCSV("gpu_profile.csv");
    ImGui::SameLine();
    if (ImGui::Button("Export Chrome Tracing"))
        profiler->ExportChromeTracing("gpu_profile.json");

    ImGui::End();
}

void DebugUI::DrawPipelineStatsPanel(const PipelineStatistics* pipeStats) {
    if (!pipeStats || !pipeStats->IsEnabled()) return;

    ImGui::Begin("Pipeline Statistics");
    const auto& s = pipeStats->GetStats();
    ImGui::Text("Vertex Shader Invocations:   %llu", static_cast<unsigned long long>(s.vertexShaderInvocations));
    ImGui::Text("Fragment Shader Invocations: %llu", static_cast<unsigned long long>(s.fragmentShaderInvocations));
    ImGui::Text("Compute Shader Invocations:  %llu", static_cast<unsigned long long>(s.computeShaderInvocations));
    ImGui::Text("Clipping Primitives:         %llu", static_cast<unsigned long long>(s.clippingPrimitives));
    ImGui::End();
}

void DebugUI::DrawSceneHierarchyPanel(Registry* registry) {
    if (!registry) return;

    ImGui::Begin("Scene Hierarchy");
    ImGui::Text("Entities: %u", registry->EntityCount());
    ImGui::Separator();

    registry->ForEachRenderable([&](Entity e, const TransformComponent& tc,
                                    const MeshComponent& mc, const MaterialComponent& matc) {
        char label[64];
        snprintf(label, sizeof(label), "Entity %u [mesh=%d mat=%d]", e, mc.meshIndex, matc.materialIndex);
        if (ImGui::Selectable(label, mSelectedEntity == e)) {
            mSelectedEntity = e;
            mSelectedMaterialIndex = matc.materialIndex;
        }
    });

    if (mSelectedEntity != UINT32_MAX) {
        ImGui::Separator();
        ImGui::Text("Selected: Entity %u", mSelectedEntity);
        auto* tc = registry->GetTransform(mSelectedEntity);
        if (tc) {
            ImGui::DragFloat3("Position", &tc->localPosition.x, 0.1f);
            ImGui::DragFloat3("Scale", &tc->localScale.x, 0.01f, 0.01f, 100.0f);
        }
    }

    ImGui::End();
}

void DebugUI::DrawMaterialEditorPanel(std::vector<GPUMaterialData>* materials) {
    if (!materials || materials->empty()) return;

    ImGui::Begin("Material Editor");

    if (mSelectedMaterialIndex >= 0 && mSelectedMaterialIndex < static_cast<int>(materials->size())) {
        ImGui::Text("Material %d", mSelectedMaterialIndex);
        ImGui::Separator();

        auto& mat = (*materials)[mSelectedMaterialIndex];
        ImGui::ColorEdit4("Base Color", &mat.baseColorFactor.x);
        ImGui::SliderFloat("Metallic", &mat.metallicFactor, 0.0f, 1.0f);
        ImGui::SliderFloat("Roughness", &mat.roughnessFactor, 0.0f, 1.0f);
    } else {
        ImGui::Text("Select an entity in the Scene Hierarchy");
    }

    ImGui::End();
}
