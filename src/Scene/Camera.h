#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class Window;

enum class CameraMode { FPS, Orbit };

class Camera {
public:
    void Init(glm::vec3 position, glm::vec3 target, float fovDeg = 45.0f,
              float nearPlane = 0.1f, float farPlane = 150.0f);

    void Update(const Window& window, float dt);

    glm::mat4 GetViewMatrix() const;
    glm::mat4 GetProjectionMatrix(float aspect) const;

    glm::vec3 GetPosition() const { return mPosition; }
    float     GetNear()     const { return mNear; }
    float     GetFar()      const { return mFar; }
    float     GetFovRad()   const;

    void SetMode(CameraMode mode) { mMode = mode; }
    CameraMode GetMode() const { return mMode; }

private:
    void UpdateFPS(const Window& window, float dt);
    void UpdateOrbit(const Window& window, float dt);

    CameraMode mMode = CameraMode::Orbit;

    // FPS
    glm::vec3 mPosition{0.0f, 2.0f, 5.0f};
    float     mYaw   = -90.0f;
    float     mPitch = -15.0f;

    // Orbit
    glm::vec3 mTarget{0.0f, 0.0f, 0.0f};
    float     mOrbitDistance = 5.0f;
    float     mOrbitYaw     = 0.0f;
    float     mOrbitPitch   = 25.0f;

    float mFovDeg    = 45.0f;
    float mNear      = 0.1f;
    float mFar       = 150.0f;
    float mMoveSpeed = 5.0f;
    float mMouseSensitivity = 0.1f;
};
