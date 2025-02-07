#include "showImage.h"

#include <opencv4/opencv2/highgui.hpp>

#include <fcntl.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

namespace hm { namespace utils {
int kbhit() {
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar();

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF) {
    ungetc(ch, stdin);
    return 1;
  }

  return 0;
}

int wait_key() {
  int c;
  while (!(c = kbhit())) {
    usleep(100);
  }
  return c;
}

void show_image(const std::string& label, const cv::Mat& img, bool wait = true) {
  cv::imshow(label, img);
  cv::waitKey(wait ? 0 : 1);
}

void displayScaledImage(const std::string& label, cv::Mat image, float scale = 1.0, bool wait = true) {
  if (scale != 1.0f) {
    // Calculate new dimensions
    int newWidth = static_cast<int>(image.cols * scale);
    int newHeight = static_cast<int>(image.rows * scale);

    // Resize the image
    cv::resize(image, image, cv::Size(newWidth, newHeight));
  }

  // Display the image
  cv::imshow(label, image);
  cv::waitKey(wait ? 0 : 1); // Wait for a keystroke in the window
}

}}
