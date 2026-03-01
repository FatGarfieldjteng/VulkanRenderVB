#include "Scene/Camera.h"
#include "Core/InputManager.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

void Camera::Init(glm::vec3 position, glm::vec3 focusPoint, float fovDeg,
                  float nearPlane, float farPlane) {
    mPosition   = position;
    mFocusPoint = focusPoint;
    mFovDeg     = fovDeg;
    mNear       = nearPlane;
    mFar        = farPlane;

    mFocusDistance = glm::length(focusPoint - position);
    if (mFocusDistance < 0.1f) mFocusDistance = 5.0f;

    glm::vec3 dir = glm::normalize(focusPoint - position);
    mYaw   = glm::degrees(std::atan2(dir.z, dir.x));
    mPitch = glm::degrees(std::asin(std::clamp(dir.y, -1.0f, 1.0f)));
}

void Camera::Update(const InputManager& input, float dt) {
    float dx = input.GetMouseDX();
    float dy = input.GetMouseDY();

    bool rightBtn  = input.IsMouseButtonDown(GLFW_MOUSE_BUTTON_RIGHT);
    bool leftBtn   = input.IsMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT);
    bool middleBtn = input.IsMouseButtonDown(GLFW_MOUSE_BUTTON_MIDDLE);

    // --- Right-click: FPS fly ---
    if (rightBtn) {
        mYaw   += dx * mMouseSensitivity;
        mPitch -= dy * mMouseSensitivity;
        mPitch  = std::clamp(mPitch, -89.0f, 89.0f);

        float speed = mMoveSpeed * dt;
        if (input.IsActive(InputManager::Action::SpeedBoost))
            speed *= 3.0f;

        glm::vec3 front = GetFront();
        glm::vec3 right = GetRight();
        glm::vec3 worldUp(0.0f, 1.0f, 0.0f);

        if (input.IsActive(InputManager::Action::MoveForward))  mPosition += front   * speed;
        if (input.IsActive(InputManager::Action::MoveBackward)) mPosition -= front   * speed;
        if (input.IsActive(InputManager::Action::MoveLeft))     mPosition -= right   * speed;
        if (input.IsActive(InputManager::Action::MoveRight))    mPosition += right   * speed;
        if (input.IsActive(InputManager::Action::MoveUp))       mPosition += worldUp * speed;
        if (input.IsActive(InputManager::Action::MoveDown))     mPosition -= worldUp * speed;

        RecalcFocusFromCamera();
    }
    // --- Left-click: Orbit ---
    else if (leftBtn) {
        mYaw   += dx * mMouseSensitivity;
        mPitch -= dy * mMouseSensitivity;
        mPitch  = std::clamp(mPitch, -89.0f, 89.0f);

        RecalcCameraFromFocus();
    }
    // --- Middle-click: Pan ---
    else if (middleBtn) {
        float panSpeed = mFocusDistance * 0.002f;

        glm::vec3 right = GetRight();
        glm::vec3 up    = GetUp();

        glm::vec3 offset = -right * dx * panSpeed + up * dy * panSpeed;
        mFocusPoint += offset;
        mPosition   += offset;
    }

    // --- Scroll: Dolly ---
    float scroll = input.GetScrollDY();
    if (std::abs(scroll) > 0.001f) {
        glm::vec3 front = GetFront();
        float dolly = scroll * mFocusDistance * 0.1f;
        mPosition      += front * dolly;
        mFocusDistance  -= dolly;
        mFocusDistance   = std::max(mFocusDistance, 0.1f);
        mFocusPoint      = mPosition + front * mFocusDistance;
    }
}

glm::vec3 Camera::GetFront() const {
    float yawRad   = glm::radians(mYaw);
    float pitchRad = glm::radians(mPitch);
    return glm::normalize(glm::vec3(
        std::cos(pitchRad) * std::cos(yawRad),
        std::sin(pitchRad),
        std::cos(pitchRad) * std::sin(yawRad)
    ));
}

glm::vec3 Camera::GetRight() const {
    return glm::normalize(glm::cross(GetFront(), glm::vec3(0.0f, 1.0f, 0.0f)));
}

glm::vec3 Camera::GetUp() const {
    return glm::normalize(glm::cross(GetRight(), GetFront()));
}

void Camera::RecalcFocusFromCamera() {
    mFocusPoint = mPosition + GetFront() * mFocusDistance;
}

void Camera::RecalcCameraFromFocus() {
    mPosition = mFocusPoint - GetFront() * mFocusDistance;
}

glm::mat4 Camera::GetViewMatrix() const {
    return glm::lookAt(mPosition, mPosition + GetFront(), glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::GetProjectionMatrix(float aspect) const {
    glm::mat4 proj = glm::perspective(glm::radians(mFovDeg), aspect, mNear, mFar);
    proj[1][1] *= -1.0f;
    return proj;
}

float Camera::GetFovRad() const {
    return glm::radians(mFovDeg);
}
