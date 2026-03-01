#pragma once

#include <glm/glm.hpp>

class InputManager;

class Camera {
public:
    void Init(glm::vec3 position, glm::vec3 focusPoint, float fovDeg = 45.0f,
              float nearPlane = 0.1f, float farPlane = 150.0f);

    void Update(const InputManager& input, float dt);

    glm::mat4 GetViewMatrix() const;
    glm::mat4 GetProjectionMatrix(float aspect) const;

    glm::vec3 GetPosition() const { return mPosition; }
    float     GetNear()     const { return mNear; }
    float     GetFar()      const { return mFar; }
    float     GetFovRad()   const;

private:
    glm::vec3 GetFront() const;
    glm::vec3 GetRight() const;
    glm::vec3 GetUp()    const;
    void RecalcFocusFromCamera();
    void RecalcCameraFromFocus();

    glm::vec3 mPosition{0.0f, 2.0f, 5.0f};
    glm::vec3 mFocusPoint{0.0f, 0.0f, 0.0f};
    float     mFocusDistance = 5.0f;

    float mYaw   = -90.0f;
    float mPitch = -15.0f;

    float mFovDeg           = 45.0f;
    float mNear             = 0.1f;
    float mFar              = 150.0f;
    float mMoveSpeed        = 500.0f;
    float mMouseSensitivity = 0.15f;
};
