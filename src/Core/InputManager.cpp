#include "Core/InputManager.h"
#include "Core/Window.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

void InputManager::Initialize() {
    mKeyBinding[static_cast<uint32_t>(Action::MoveForward)]  = GLFW_KEY_W;
    mKeyBinding[static_cast<uint32_t>(Action::MoveBackward)] = GLFW_KEY_S;
    mKeyBinding[static_cast<uint32_t>(Action::MoveLeft)]     = GLFW_KEY_A;
    mKeyBinding[static_cast<uint32_t>(Action::MoveRight)]    = GLFW_KEY_D;
    mKeyBinding[static_cast<uint32_t>(Action::MoveUp)]       = GLFW_KEY_E;
    mKeyBinding[static_cast<uint32_t>(Action::MoveDown)]     = GLFW_KEY_Q;
    mKeyBinding[static_cast<uint32_t>(Action::SpeedBoost)]   = GLFW_KEY_LEFT_SHIFT;
    mKeyBinding[static_cast<uint32_t>(Action::Quit)]         = GLFW_KEY_ESCAPE;
}

void InputManager::Update(const Window& window) {
    for (uint32_t i = 0; i < kCount; i++) {
        mPrevious[i] = mCurrent[i];
        mCurrent[i]  = window.IsKeyDown(mKeyBinding[i]);
    }

    mMouseButtons[GLFW_MOUSE_BUTTON_LEFT]   = window.IsMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT);
    mMouseButtons[GLFW_MOUSE_BUTTON_RIGHT]  = window.IsMouseButtonDown(GLFW_MOUSE_BUTTON_RIGHT);
    mMouseButtons[GLFW_MOUSE_BUTTON_MIDDLE] = window.IsMouseButtonDown(GLFW_MOUSE_BUTTON_MIDDLE);

    mMouseDX  = window.GetMouseDX();
    mMouseDY  = window.GetMouseDY();
    mScrollDY = window.GetScrollDY();
}

bool InputManager::IsActive(Action action) const {
    return mCurrent[static_cast<uint32_t>(action)];
}

bool InputManager::WasPressed(Action action) const {
    uint32_t i = static_cast<uint32_t>(action);
    return mCurrent[i] && !mPrevious[i];
}

bool InputManager::IsMouseButtonDown(int button) const {
    if (button < 0 || button >= static_cast<int>(kMouseButtons)) return false;
    return mMouseButtons[button];
}
