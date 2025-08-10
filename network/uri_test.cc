#include "URI.h"
#include <string>

int main() {
    URI u;
    if (!u.Parse("rtsp://user:pass@127.0.0.1:8554/test")) return 1;
    if (u.protocol != std::string("rtsp")) return 1;
    if (u.location.find("127.0.0.1") == std::string::npos) return 1;
    return 0;
}
