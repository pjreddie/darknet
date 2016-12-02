#include "gettimeofday.h"
 
int gettimeofday(struct timeval *tv, struct timezone *tz)
{
  FILETIME ft;
  unsigned __int64 tmpres = 0;
  static int tzflag;
 
  if (NULL != tv)
  {
    GetSystemTimeAsFileTime(&ft);
 
    tmpres |= ft.dwHighDateTime;
    tmpres <<= 32;
    tmpres |= ft.dwLowDateTime;
 
    /*converting file time to unix epoch*/
    tmpres -= DELTA_EPOCH_IN_MICROSECS; 
    tmpres /= 10;  /*convert into microseconds*/
    tv->tv_sec = (long)(tmpres / 1000000UL);
    tv->tv_usec = (long)(tmpres % 1000000UL);
  }
 
  if (NULL != tz)
  {
    if (!tzflag)
    {
      _tzset();
      tzflag++;
    }
    tz->tz_minuteswest = _timezone / 60;
    tz->tz_dsttime = _daylight;
  }
 
  return 0;
}

/* never worry about timersub type activies again -- from GLIBC and upcased. */
int timersub(struct timeval *a, struct timeval *b, struct timeval *result)
{                                                                
         (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;                        
         (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;                     
         if ((result)->tv_usec < 0) {                                         
           --(result)->tv_sec;                                                
           (result)->tv_usec += 1000000;                                      
         }                                                                         

    return 0;
}