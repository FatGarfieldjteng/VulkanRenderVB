#include "Core/Window.h"
#include "Core/Logger.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

void Window::Initialize(uint32_t width, uint32_t height, const std::string& title) {
    if (!glfwInit()) {
        LOG_ERROR("Failed to initialize GLFW");
        return;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    mWidth  = width;
    mHeight = height;
    mHandle = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height),
                               title.c_str(), nullptr, nullptr);
    if (!mHandle) {
        LOG_ERROR("Failed to create GLFW window");
        return;
    }

    glfwSetWindowUserPointer(mHandle, this);
    glfwSetFramebufferSizeCallback(mHandle, FramebufferResizeCallback);
    glfwSetCursorPosCallback(mHandle, CursorPosCallback);
    glfwSetScrollCallback(mHandle, ScrollCallback);
    glfwSetKeyCallback(mHandle, KeyCallback);

    LOG_INFO("Window created: {}x{} \"{}\"", width, height, title);
}

void Window::Shutdown() {
    if (mHandle) {
        glfwDestroyWindow(mHandle);
        mHandle = nullptr;
    }
    glfwTerminate();
    LOG_INFO("Window destroyed");
}

bool Window::ShouldClose() const {
    return mHandle && glfwWindowShouldClose(mHandle);
}

void Window::PollEvents() {
    glfwPollEvents();
}

void Window::WaitEvents() {
    glfwWaitEvents();
}

bool Window::IsKeyDown(int key) const {
    return mHandle && glfwGetKey(mHandle, key) == GLFW_PRESS;
}

bool Window::IsMouseButtonDown(int button) const {
    return mHandle && glfwGetMouseButton(mHandle, button) == GLFW_PRESS;
}

void Window::SetCursorCaptured(bool captured) {
    mCursorCaptured = captured;
    if (mHandle) {
        glfwSetInputMode(mHandle, GLFW_CURSOR,
                         captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        if (captured) mFirstMouse = true;
    }
}

void Window::ResetInputDeltas() {
    mMouseDX = 0.0f;
    mMouseDY = 0.0f;
    mScrollDY = 0.0f;
}

void Window::FramebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    self->mWidth  = static_cast<uint32_t>(width);
    self->mHeight = static_cast<uint32_t>(height);
    if (self->mResizeCallback) {
        self->mResizeCallback(self->mWidth, self->mHeight);
    }
}

void Window::CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    auto fx = static_cast<float>(xpos);
    auto fy = static_cast<float>(ypos);

    if (self->mFirstMouse) {
        self->mLastMouseX = fx;
        self->mLastMouseY = fy;
        self->mFirstMouse = false;
    }

    self->mMouseDX += fx - self->mLastMouseX;
    self->mMouseDY += fy - self->mLastMouseY;
    self->mLastMouseX = fx;
    self->mLastMouseY = fy;
}

void Window::ScrollCallback(GLFWwindow* window, double /*xoff*/, double yoff) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    self->mScrollDY += static_cast<float>(yoff);
}

void Window::KeyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        if (self->mCursorCaptured)
            self->SetCursorCaptured(false);
        else
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {
        self->SetCursorCaptured(!self->mCursorCaptured);
    }
}
