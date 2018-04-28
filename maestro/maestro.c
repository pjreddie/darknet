
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <termios.h>

int maestroIni()
{
  const char * device = "/dev/ttyACM0";  // Linux
  
  int fd = open(device, O_RDWR | O_NOCTTY);
  if (fd == -1)
  {
    perror(device);
    return -1;
  }
 
  struct termios options;
  tcgetattr(fd, &options);
  options.c_iflag &= ~(INLCR | IGNCR | ICRNL | IXON | IXOFF);
  options.c_oflag &= ~(ONLCR | OCRNL);
  options.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
  tcsetattr(fd, TCSANOW, &options);

  return fd;
}

int maestroGetPosition(int fd, unsigned char channel)
{
  unsigned char command[] = {0x90, channel};
  if(write(fd, command, sizeof(command)) == -1)
  {
    perror("error writing");
    return -1;
  }
 
  unsigned char response[2];
  if(read(fd,response,2) != 2)
  {
    perror("error reading");
    return -1;
  }
 
  return response[0] + 256*response[1];
}
 
// The units of 'target' are quarter-microseconds.
int maestroSetTarget(int fd, unsigned char channel, unsigned short target)
{
  unsigned char command[] = {0x84, channel, target & 0x7F, target >> 7 & 0x7F};
  if (write(fd, command, sizeof(command)) == -1)
  {
    perror("error writing");
    return -1;
  }
  return 0;
}

// Speed of 0 is unrestricted.
// A speed of 1 will take 1 minute, and a speed of 60 would take 1 second.
int maestroSetSpeed(int fd, unsigned char channel, unsigned short speed)
{
  unsigned char command[] = {0x07, channel, speed & 0x7F, speed >> 7 & 0x7F};
  if (write(fd, command, sizeof(command)) == -1)
  {
    perror("error writing");
    return -1;
  }
  return 0;
}
// Valid values are from 0 to 255. 0=unrestricted, 1 is slowest start.
// A value of 1 will take the servo about 3s to move between 1ms to 2ms range.
int maestroSetAccel(int fd, unsigned char channel, unsigned short accel)
{
  unsigned char command[] = {0x09, channel, accel & 0x7F, accel >> 7 & 0x7F};
  if (write(fd, command, sizeof(command)) == -1)
  {
    perror("error writing");
    return -1;
  }
  return 0;
}

//int main()
int test_maestro()
{
  int fd = maestroIni();
  //get the position of maestro
  int position = maestroGetPosition(fd, 0);
  printf("Current position is %d.\n", position);
 
  //send control command
  int speed = 30;
  maestroSetSpeed(fd, 0, speed);
  //maestroSetAccel(fd, 0, 30);

  int ud = 6000;
  maestroSetTarget(fd, 0, ud); //control up down

  int lr = 6000; 
  maestroSetTarget(fd, 1, lr); //control left right

  close(fd);
  return 0;
}
