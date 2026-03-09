#pragma once

#include <volk.h>
#include <cstdint>
#include <vector>

struct GLFWwindow;
class GPUProfiler;
class PipelineStatistics;
class Registry;
struct GPUMaterialData;
struct PostProcessSettings;

struct DebugUIState {
    bool gpuDriven        = true;
    bool occlusionCulling = true;
    float occluderRatio   = 0.2f;

    enum class VisMode : int {
        None = 0, Wireframe, WorldNormals, TangentNormals,
        UVs, MipLevel, Overdraw, CascadeColoring, DepthBuffer,
        Count
    };
    VisMode visMode = VisMode::None;

    bool showDemoWindow    = false;
    bool pipelineStatsEnabled = false;

    int  msaaIndex = 0;
    bool msaaChanged = false;
};

class DebugUI {
public:
    void Initialize(VkInstance instance, VkDevice device, VkPhysicalDevice physicalDevice,
                    uint32_t graphicsFamily, VkQueue graphicsQueue,
                    GLFWwindow* window, VkFormat swapchainFormat,
                    uint32_t framesInFlight);
    void Shutdown(VkDevice device);

    void BeginFrame();
    void BuildUI(float deltaTime, const GPUProfiler* profiler, const PipelineStatistics* pipeStats,
                 Registry* registry = nullptr, std::vector<GPUMaterialData>* materials = nullptr,
                 PostProcessSettings* ppSettings = nullptr,
                 const std::vector<VkSampleCountFlagBits>* supportedMSAA = nullptr);
    void EndFrame();

    void Render(VkCommandBuffer cmd, VkImageView colorView, VkExtent2D extent);

    DebugUIState& GetState() { return mState; }
    const DebugUIState& GetState() const { return mState; }

    bool WantCaptureMouse() const;
    bool WantCaptureKeyboard() const;

private:
    void DrawMainMenuBar();
    void DrawRenderSettingsPanel(float deltaTime, const std::vector<VkSampleCountFlagBits>* supportedMSAA);
    void DrawProfilerPanel(const GPUProfiler* profiler);
    void DrawPipelineStatsPanel(const PipelineStatistics* pipeStats);
    void DrawSceneHierarchyPanel(Registry* registry);
    void DrawMaterialEditorPanel(std::vector<GPUMaterialData>* materials);
    void DrawPostProcessPanel(PostProcessSettings* settings);
    void DrawToneMappingPanel(PostProcessSettings* settings);

    VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
    DebugUIState     mState;

    float    mFpsHistory[120]{};
    int      mFpsHistoryOffset      = 0;
    float    mFpsSmoothed           = 0.0f;
    uint32_t mSelectedEntity        = UINT32_MAX;
    int      mSelectedMaterialIndex = -1;
};
