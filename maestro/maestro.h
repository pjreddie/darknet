
#pragma once

extern "C" {
int maestroGetPosition(int fd, unsigned char channel);
int maestroSetTarget(int fd, unsigned char channel, unsigned short target);
int test_maestro();
}


