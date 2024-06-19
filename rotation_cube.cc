#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <cstring>
#include <thread>
#include <chrono>

constexpr int width = 160;
constexpr int height = 44;
constexpr int backgroundASCIICode = ' ';
constexpr float distanceFromCam = 100.0f;
constexpr float K1 = 40.0f;
constexpr float incrementSpeed = 0.6f;

float A = 0.0f, B = 0.0f, C = 0.0f;
float cubeWidth = 20.0f;
float horizontalOffset = -2 * cubeWidth;

std::array<float, width * height> zBuffer{};
std::array<char, width * height> buffer{};

float calculateX(int i, int j, int k) {
    return j * sin(A) * sin(B) * cos(C) - k * cos(A) * sin(B) * cos(C) +
           j * cos(A) * sin(C) + k * sin(A) * sin(C) + i * cos(B) * cos(C);
}

float calculateY(int i, int j, int k) {
    return j * cos(A) * cos(C) + k * sin(A) * cos(C) -
           j * sin(A) * sin(B) * sin(C) + k * cos(A) * sin(B) * sin(C) -
           i * cos(B) * sin(C);
}

float calculateZ(int i, int j, int k) {
    return k * cos(A) * cos(B) - j * sin(A) * cos(B) + i * sin(B);
}

void calculateForSurface(float cubeX, float cubeY, float cubeZ, char ch) {
    float x = calculateX(cubeX, cubeY, cubeZ);
    float y = calculateY(cubeX, cubeY, cubeZ);
    float z = calculateZ(cubeX, cubeY, cubeZ) + distanceFromCam;

    float ooz = 1.0f / z;

    int xp = static_cast<int>(width / 2 + horizontalOffset + K1 * ooz * x * 2);
    int yp = static_cast<int>(height / 2 + K1 * ooz * y);

    int idx = xp + yp * width;
    if (idx >= 0 && idx < width * height) {
        if (ooz > zBuffer[idx]) {
            zBuffer[idx] = ooz;
            buffer[idx] = ch;
        }
    }
}

int main() {
    std::cout << "\x1b[2J";

    while (true) {
        std::fill(buffer.begin(), buffer.end(), backgroundASCIICode);
        std::fill(zBuffer.begin(), zBuffer.end(), 0);

        for (float cubeX = -cubeWidth; cubeX < cubeWidth; cubeX += incrementSpeed) {
            for (float cubeY = -cubeWidth; cubeY < cubeWidth; cubeY += incrementSpeed) {
                calculateForSurface(cubeX, cubeY, -cubeWidth, '@');
                calculateForSurface(cubeWidth, cubeY, cubeX, '$');
                calculateForSurface(-cubeWidth, cubeY, -cubeX, '~');
                calculateForSurface(-cubeX, cubeY, cubeWidth, '#');
                calculateForSurface(cubeX, -cubeWidth, -cubeY, ';');
                calculateForSurface(cubeX, cubeWidth, cubeY, '+');
            }
        }

        std::cout << "\x1b[H";
        for (int k = 0; k < width * height; ++k) {
            std::cout.put(k % width ? buffer[k] : '\n');
        }

        A += 0.05f;
        B += 0.05f;
        C += 0.01f;

        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }

    return 0;
}
