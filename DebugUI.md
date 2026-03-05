# DebugUI

## Design

`DebugUI` follows a **separation of concerns** pattern. It is purely a UI layer — it doesn't own or directly modify any rendering state. Instead, it uses an intermediary data structure, `DebugUIState`, as a **two-way bridge** between the UI and the application:

- The UI writes user selections into `mState` (checkboxes, sliders, combos)
- The application reads `mState` via `GetState()` in `SyncUIState()` and applies the changes to its own members (`mGPUDriven`, `mOcclusionCulling`, etc.)

This keeps ImGui completely decoupled from the rendering backend.

## Initialization

`Initialize` sets up three things:

1. **Descriptor pool** — ImGui's Vulkan backend needs its own pool for internal textures (font atlas, etc.). A generous pool is created with mixed descriptor types.

2. **ImGui context** — Creates the ImGui context, enables docking, sets a dark theme with rounded corners, and configures layout persistence to `imgui_layout.ini`.

3. **Vulkan backend** — Initializes `imgui_impl_glfw` (for input) and `imgui_impl_vulkan` (for rendering). Notably, it uses **dynamic rendering** (`UseDynamicRendering = true`) instead of render passes, matching the rest of the engine.

## Per-frame Flow

Driven from `Application.cpp`, the frame lifecycle is:

```
1. SyncUIState()              ← read mState, apply to application
2. mDebugUI.BeginFrame()      ← ImGui_ImplVulkan_NewFrame + ImGui::NewFrame
3. mDebugUI.BuildUI(...)      ← construct all panels (immediate-mode)
4. mDebugUI.EndFrame()        ← ImGui::Render() — finalizes draw lists
5. (later, in render graph)
   mDebugUI.Render(cmd, ...)  ← records actual Vulkan draw commands
```

Steps 2-4 happen on the CPU and only build ImGui's internal draw lists. Step 5 happens during command buffer recording and actually emits the GPU draw calls via `ImGui_ImplVulkan_RenderDrawData`.

## The Panels

`BuildUI` sets up a **dockspace** over the entire viewport (with `PassthruCentralNode` so the 3D scene shows through), then draws six panels:

| Panel | Method | What it does |
|-------|--------|-------------|
| Menu bar | `DrawMainMenuBar` | Top bar with "View > Demo Window" toggle |
| Render Settings | `DrawRenderSettingsPanel` | FPS graph (120-frame rolling history with EMA smoothing), GPU-driven/occlusion checkboxes, occluder ratio slider, pipeline stats toggle, debug visualization mode combo |
| GPU Profiler | `DrawProfilerPanel` | Per-pass GPU timings as progress bars (proportional to total), CSV and Chrome Tracing export buttons |
| Pipeline Statistics | `DrawPipelineStatsPanel` | Vertex/fragment/compute invocation counts and clipping primitive count (only shown when enabled) |
| Scene Hierarchy | `DrawSceneHierarchyPanel` | Scrollable list of all entities from the ECS registry. Clicking one selects it and shows position/scale drag controls |
| Material Editor | `DrawMaterialEditorPanel` | Edits the selected entity's material: base color (color picker), metallic and roughness sliders |

## Rendering

`Render` uses Vulkan 1.3 dynamic rendering (no VkRenderPass needed):

```cpp
void DebugUI::Render(VkCommandBuffer cmd, VkImageView colorView, VkExtent2D extent) {
    // ... sets up VkRenderingAttachmentInfo with LOAD_OP_LOAD (preserves scene)
    vkCmdBeginRendering(cmd, &ri);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}
```

`LOAD_OP_LOAD` is key — it blends the UI on top of the already-rendered 3D scene rather than clearing it.

## Input Gating

`WantCaptureMouse()` and `WantCaptureKeyboard()` delegate to ImGui's IO state. The application uses these to prevent camera movement when the user is interacting with a UI widget:

```cpp
bool uiWantsMouse = mDebugUI.WantCaptureMouse();
if (!uiWantsMouse)
    mCamera.Update(mInput, dt);
```

## State Flow Summary

```
User clicks checkbox  →  ImGui writes to mState.gpuDriven
                      →  SyncUIState() reads mState, sets mGPUDriven
                      →  Next frame picks up mGPUDriven in render path
```

This is the **immediate-mode GUI** pattern: every frame the UI is rebuilt from scratch based on current state, and user interactions mutate the state variables directly. No event system, no callbacks, no retained widget tree.
