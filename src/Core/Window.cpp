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

void Window::FramebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    self->mWidth  = static_cast<uint32_t>(width);
    self->mHeight = static_cast<uint32_t>(height);
    if (self->mResizeCallback) {
        self->mResizeCallback(self->mWidth, self->mHeight);
    }
}
