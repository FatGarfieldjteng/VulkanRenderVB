#include "Core/Application.h"
#include "Core/Logger.h"

#include <exception>
#include <cstring>
#include <string>

int main(int argc, char* argv[]) {
    try {
        bool benchmark = false;
        uint32_t frames = 200;
        bool gpuDriven = true;
        bool occlusion = true;
        std::string scenePath;
        bool pathTracing = false;
        bool denoiserOn = true;  // default on when path tracing

        for (int i = 1; i < argc; i++) {
            if (std::strcmp(argv[i], "--benchmark") == 0) benchmark = true;
            else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) frames = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--no-gpu") == 0) gpuDriven = false;
            else if (std::strcmp(argv[i], "--no-occlusion") == 0) occlusion = false;
            else if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc) scenePath = argv[++i];
            else if (std::strcmp(argv[i], "--path-tracing") == 0) pathTracing = true;
            else if (std::strcmp(argv[i], "--no-denoiser") == 0) { denoiserOn = false; pathTracing = true; }
        }

        Application app;
        if (!scenePath.empty())
            app.SetScenePath(scenePath);
        if (pathTracing)
            app.SetInitialRenderMode(DebugUIState::RenderMode::FullPathTracing);
        if (!denoiserOn)
            app.SetInitialDenoiser(false);
        if (benchmark)
            app.RunBenchmark(frames, gpuDriven, occlusion);
        else
            app.Run();
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal exception: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
