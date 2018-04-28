
#pragma once

extern "C" {

int maestroIni();

int maestroGetPosition(int fd, unsigned char channel);
int maestroSetTarget(int fd, unsigned char channel, unsigned short target);
int maestroSetSpeed(int fd, unsigned char channel, unsigned short speed);
int maestroSetAccel(int fd, unsigned char channel, unsigned short accel);

int test_maestro();

}


