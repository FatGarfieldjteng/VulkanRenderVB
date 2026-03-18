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

enum class SceneType : int { Sponza = 0, TestScene = 1 };

struct DebugUIState {
    bool gpuDriven        = true;
    bool occlusionCulling = true;
    float occluderRatio   = 0.2f;

    SceneType sceneType    = SceneType::TestScene;
    bool      sceneChanged = false;

    // Lighting
    float lightAzimuth   = 53.0f;   // horizontal angle in degrees
    float lightElevation = 58.0f;   // angle above horizon in degrees
    bool  csmEnabled     = true;

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

    // Ray Tracing (Phase 9)
    bool  rtShadowsEnabled  = true;
    bool  rtReflEnabled      = true;
    float rtShadowStrength   = 1.0f;
    float rtReflStrength     = 0.5f;
    float rtReflRoughness    = 0.15f;
    float rtLightRadius      = 0.02f;
    bool  rtAvailable        = false;
    bool  rtDebugShadowVis   = false;

    // Render Mode (Phase 10)
    enum class RenderMode : int {
        Rasterization = 0,
        Hybrid        = 1,
        FullPathTracing = 2
    };
    RenderMode renderMode        = RenderMode::Rasterization;
    bool       renderModeChanged = false;
    bool       rtPipelineAvailable = false;

    bool  splitScreenEnabled = false;
    float splitScreenPos     = 0.5f;   // 0..1, left = mode A, right = mode B
    RenderMode splitModeA    = RenderMode::Rasterization;
    RenderMode splitModeB    = RenderMode::FullPathTracing;

    // Path tracer settings
    int   ptMaxBounces       = 8;
    bool  ptEnableMIS        = true;
    bool  ptEnableDenoiser   = true;
    bool  ptProgressive      = true;
    bool  ptBypassNRDOutput  = false;  // When denoiser on: show accum instead of NRD output (debug)
    bool  ptDenoiserComparison = false;  // Split-screen: left=denoised, right=raw (for debugging)
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
    void DrawRenderModePanel();

    VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
    DebugUIState     mState;

    float    mFpsHistory[120]{};
    int      mFpsHistoryOffset      = 0;
    float    mFpsSmoothed           = 0.0f;
    uint32_t mSelectedEntity        = UINT32_MAX;
    int      mSelectedMaterialIndex = -1;
};
