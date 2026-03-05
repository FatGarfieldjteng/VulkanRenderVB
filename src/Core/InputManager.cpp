#include "Core/InputManager.h"
#include "Core/Window.h"
#include "Core/Logger.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>

void InputManager::Initialize() {
    mKeyBinding[static_cast<uint32_t>(Action::MoveForward)]  = GLFW_KEY_W;
    mKeyBinding[static_cast<uint32_t>(Action::MoveBackward)] = GLFW_KEY_S;
    mKeyBinding[static_cast<uint32_t>(Action::MoveLeft)]     = GLFW_KEY_A;
    mKeyBinding[static_cast<uint32_t>(Action::MoveRight)]    = GLFW_KEY_D;
    mKeyBinding[static_cast<uint32_t>(Action::MoveUp)]       = GLFW_KEY_E;
    mKeyBinding[static_cast<uint32_t>(Action::MoveDown)]     = GLFW_KEY_Q;
    mKeyBinding[static_cast<uint32_t>(Action::SpeedBoost)]   = GLFW_KEY_LEFT_SHIFT;
    mKeyBinding[static_cast<uint32_t>(Action::ToggleUI)]     = GLFW_KEY_F2;
    mKeyBinding[static_cast<uint32_t>(Action::Quit)]         = GLFW_KEY_ESCAPE;

    std::memset(mGamepadAxes, 0, sizeof(mGamepadAxes));
    std::memset(mGamepadButtons, 0, sizeof(mGamepadButtons));
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

    mGamepadConnected = false;
    GLFWgamepadstate state;
    if (glfwJoystickIsGamepad(GLFW_JOYSTICK_1) && glfwGetGamepadState(GLFW_JOYSTICK_1, &state)) {
        mGamepadConnected = true;
        for (int i = 0; i < 6; i++)
            mGamepadAxes[i] = state.axes[i];
        for (int i = 0; i < 15; i++)
            mGamepadButtons[i] = state.buttons[i] == GLFW_PRESS;
    }
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

float InputManager::GetGamepadAxis(int axis) const {
    if (axis < 0 || axis >= 6) return 0.0f;
    float val = mGamepadAxes[axis];
    return (std::abs(val) < 0.15f) ? 0.0f : val;
}

bool InputManager::IsGamepadButtonDown(int button) const {
    if (button < 0 || button >= 16) return false;
    return mGamepadButtons[button];
}

void InputManager::SetKeyBinding(Action action, int glfwKey) {
    mKeyBinding[static_cast<uint32_t>(action)] = glfwKey;
}

int InputManager::GetKeyBinding(Action action) const {
    return mKeyBinding[static_cast<uint32_t>(action)];
}

const char* InputManager::ActionName(Action action) {
    switch (action) {
        case Action::MoveForward:  return "MoveForward";
        case Action::MoveBackward: return "MoveBackward";
        case Action::MoveLeft:     return "MoveLeft";
        case Action::MoveRight:    return "MoveRight";
        case Action::MoveUp:       return "MoveUp";
        case Action::MoveDown:     return "MoveDown";
        case Action::SpeedBoost:   return "SpeedBoost";
        case Action::ToggleUI:     return "ToggleUI";
        case Action::Quit:         return "Quit";
        default: return "Unknown";
    }
}

void InputManager::SaveBindings(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) return;

    for (uint32_t i = 0; i < kCount; i++) {
        out << ActionName(static_cast<Action>(i)) << "=" << mKeyBinding[i] << "\n";
    }
    LOG_INFO("Input bindings saved to {}", path);
}

void InputManager::LoadBindings(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) return;

    std::unordered_map<std::string, int> bindings;
    std::string line;
    while (std::getline(in, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string name = line.substr(0, eq);
        int key = std::stoi(line.substr(eq + 1));
        bindings[name] = key;
    }

    for (uint32_t i = 0; i < kCount; i++) {
        auto it = bindings.find(ActionName(static_cast<Action>(i)));
        if (it != bindings.end())
            mKeyBinding[i] = it->second;
    }

    LOG_INFO("Input bindings loaded from {} ({} entries)", path, bindings.size());
}
