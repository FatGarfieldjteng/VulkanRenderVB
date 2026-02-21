#pragma once

#include <functional>
#include <string>
#include <cstdint>

struct GLFWwindow;

class Window {
public:
    using ResizeCallback = std::function<void(uint32_t width, uint32_t height)>;

    void Initialize(uint32_t width, uint32_t height, const std::string& title);
    void Shutdown();

    bool ShouldClose() const;
    void PollEvents();
    void WaitEvents();

    GLFWwindow* GetHandle() const { return mHandle; }
    uint32_t    GetWidth()  const { return mWidth; }
    uint32_t    GetHeight() const { return mHeight; }

    void SetResizeCallback(ResizeCallback callback) { mResizeCallback = std::move(callback); }

private:
    static void FramebufferResizeCallback(GLFWwindow* window, int width, int height);

    GLFWwindow*    mHandle = nullptr;
    uint32_t       mWidth  = 0;
    uint32_t       mHeight = 0;
    ResizeCallback mResizeCallback;
};
