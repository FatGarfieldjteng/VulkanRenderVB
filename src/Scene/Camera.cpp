#include "Scene/Camera.h"
#include "Core/Window.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

void Camera::Init(glm::vec3 position, glm::vec3 target, float fovDeg,
                  float nearPlane, float farPlane) {
    mPosition = position;
    mTarget   = target;
    mFovDeg   = fovDeg;
    mNear     = nearPlane;
    mFar      = farPlane;

    glm::vec3 dir = glm::normalize(target - position);
    mYaw   = glm::degrees(std::atan2(dir.z, dir.x));
    mPitch = glm::degrees(std::asin(dir.y));

    mOrbitDistance = glm::length(target - position);
    mOrbitYaw     = -mYaw;
    mOrbitPitch   = -mPitch;
}

void Camera::Update(const Window& window, float dt) {
    if (window.IsKeyDown(GLFW_KEY_TAB)) {
        // handled by key callback debounce; but for simplicity toggle check each frame
    }

    if (mMode == CameraMode::FPS)
        UpdateFPS(window, dt);
    else
        UpdateOrbit(window, dt);
}

void Camera::UpdateFPS(const Window& window, float dt) {
    if (!window.IsCursorCaptured()) return;

    float dx = window.GetMouseDX() * mMouseSensitivity;
    float dy = window.GetMouseDY() * mMouseSensitivity;

    mYaw   += dx;
    mPitch -= dy;
    mPitch = std::clamp(mPitch, -89.0f, 89.0f);

    float yawRad   = glm::radians(mYaw);
    float pitchRad = glm::radians(mPitch);

    glm::vec3 front;
    front.x = std::cos(pitchRad) * std::cos(yawRad);
    front.y = std::sin(pitchRad);
    front.z = std::cos(pitchRad) * std::sin(yawRad);
    front = glm::normalize(front);

    glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0, 1, 0)));
    glm::vec3 up    = glm::normalize(glm::cross(right, front));

    float speed = mMoveSpeed * dt;
    if (window.IsKeyDown(GLFW_KEY_LEFT_SHIFT)) speed *= 3.0f;

    if (window.IsKeyDown(GLFW_KEY_W)) mPosition += front * speed;
    if (window.IsKeyDown(GLFW_KEY_S)) mPosition -= front * speed;
    if (window.IsKeyDown(GLFW_KEY_A)) mPosition -= right * speed;
    if (window.IsKeyDown(GLFW_KEY_D)) mPosition += right * speed;
    if (window.IsKeyDown(GLFW_KEY_E)) mPosition += up * speed;
    if (window.IsKeyDown(GLFW_KEY_Q)) mPosition -= up * speed;
}

void Camera::UpdateOrbit(const Window& window, float dt) {
    (void)dt;

    if (window.IsMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT)) {
        mOrbitYaw   += window.GetMouseDX() * mMouseSensitivity;
        mOrbitPitch += window.GetMouseDY() * mMouseSensitivity;
        mOrbitPitch = std::clamp(mOrbitPitch, -89.0f, 89.0f);
    }

    if (window.IsMouseButtonDown(GLFW_MOUSE_BUTTON_MIDDLE)) {
        float panSpeed = mOrbitDistance * 0.002f;
        float yawRad   = glm::radians(mOrbitYaw);
        float pitchRad = glm::radians(mOrbitPitch);

        glm::vec3 front;
        front.x = std::cos(pitchRad) * std::sin(yawRad);
        front.y = -std::sin(pitchRad);
        front.z = std::cos(pitchRad) * std::cos(yawRad);

        glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0, 1, 0)));
        glm::vec3 up    = glm::normalize(glm::cross(right, front));

        mTarget -= right * window.GetMouseDX() * panSpeed;
        mTarget += up    * window.GetMouseDY() * panSpeed;
    }

    mOrbitDistance -= window.GetScrollDY() * mOrbitDistance * 0.1f;
    mOrbitDistance = std::clamp(mOrbitDistance, 0.5f, 200.0f);

    float yawRad   = glm::radians(mOrbitYaw);
    float pitchRad = glm::radians(mOrbitPitch);

    mPosition.x = mTarget.x + mOrbitDistance * std::cos(pitchRad) * std::sin(yawRad);
    mPosition.y = mTarget.y + mOrbitDistance * std::sin(pitchRad);
    mPosition.z = mTarget.z + mOrbitDistance * std::cos(pitchRad) * std::cos(yawRad);
}

glm::mat4 Camera::GetViewMatrix() const {
    if (mMode == CameraMode::Orbit) {
        return glm::lookAt(mPosition, mTarget, glm::vec3(0, 1, 0));
    }

    float yawRad   = glm::radians(mYaw);
    float pitchRad = glm::radians(mPitch);
    glm::vec3 front;
    front.x = std::cos(pitchRad) * std::cos(yawRad);
    front.y = std::sin(pitchRad);
    front.z = std::cos(pitchRad) * std::sin(yawRad);
    return glm::lookAt(mPosition, mPosition + glm::normalize(front), glm::vec3(0, 1, 0));
}

glm::mat4 Camera::GetProjectionMatrix(float aspect) const {
    glm::mat4 proj = glm::perspective(glm::radians(mFovDeg), aspect, mNear, mFar);
    proj[1][1] *= -1.0f;
    return proj;
}

float Camera::GetFovRad() const {
    return glm::radians(mFovDeg);
}
