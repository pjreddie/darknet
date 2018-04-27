
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include <termios.h>
 
// Gets the position of a Maestro channel.
// See the "Serial Servo Commands" section of the user's guide.
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
 
// Sets the target of a Maestro channel.
// See the "Serial Servo Commands" section of the user's guide.
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
 
int test_maestro()
{
  const char * device = "/dev/ttyACM0";  // Linux
  //const char * device = "/dev/cu.usbmodem00034567";  // Mac OS X
  int fd = open(device, O_RDWR | O_NOCTTY);
  if (fd == -1)
  {
    perror(device);
    return 1;
  }
 
  struct termios options;
  tcgetattr(fd, &options);
  options.c_iflag &= ~(INLCR | IGNCR | ICRNL | IXON | IXOFF);
  options.c_oflag &= ~(ONLCR | OCRNL);
  options.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
  tcsetattr(fd, TCSANOW, &options);

 
  int position = maestroGetPosition(fd, 0);
  printf("Current position is %d.\n", position);
 
  int ud = 6000;
  printf("Setting target to %d (%d us).\n", ud, ud/4);
  maestroSetTarget(fd, 0, ud); //control up down


  int lr = 6000; 
  maestroSetTarget(fd, 1, lr); //control left right

  close(fd);
  return 0;
}
