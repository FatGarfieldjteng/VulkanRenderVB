#include "Core/Application.h"
#include "Core/Logger.h"

#include <exception>

int main() {
    try {
        Application app;
        app.Run();
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal exception: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
