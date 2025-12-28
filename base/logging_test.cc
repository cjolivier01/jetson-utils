#include "logging.h"
#include <cstring>

int main() {
    // Basic sanity checks on logging API
    Log::SetLevel(Log::INFO);
    Log::SetFile(stdout);

    const char* s = Log::LevelToStr(Log::INFO);
    if (!s || std::strcmp(s, "info") != 0) return 1;

    LogInfo("logging test info message\n");
    LogSuccess("logging test success message\n");
    LogWarning("logging test warning message\n");
    LogError("logging test error message\n");

    return 0;
}

