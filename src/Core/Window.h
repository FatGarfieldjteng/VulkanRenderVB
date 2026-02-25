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

    // Input state
    bool  IsKeyDown(int key) const;
    bool  IsMouseButtonDown(int button) const;
    float GetMouseDX() const { return mMouseDX; }
    float GetMouseDY() const { return mMouseDY; }
    float GetScrollDY() const { return mScrollDY; }

    void SetCursorCaptured(bool captured);
    bool IsCursorCaptured() const { return mCursorCaptured; }

    /// Must be called once per frame to reset deltas.
    void ResetInputDeltas();

private:
    static void FramebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void ScrollCallback(GLFWwindow* window, double xoff, double yoff);
    static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    GLFWwindow*    mHandle = nullptr;
    uint32_t       mWidth  = 0;
    uint32_t       mHeight = 0;
    ResizeCallback mResizeCallback;

    float mMouseDX = 0.0f, mMouseDY = 0.0f;
    float mLastMouseX = 0.0f, mLastMouseY = 0.0f;
    float mScrollDY = 0.0f;
    bool  mFirstMouse = true;
    bool  mCursorCaptured = false;
};
