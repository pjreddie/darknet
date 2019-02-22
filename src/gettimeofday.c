#ifdef _WIN32
#include "gettimeofday.h"

int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
  static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);
  SYSTEMTIME system_time;
  FILETIME file_time;
  uint64_t time;


  GetSystemTime(&system_time);
  SystemTimeToFileTime(&system_time, &file_time);
  time = ((uint64_t)file_time.dwLowDateTime);
  time += ((uint64_t)file_time.dwHighDateTime) << 32;
    /*converting file time to unix epoch*/
  tp->tv_sec = (long)((time - EPOCH) / 10000000L);
  tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
  return 0;
  }

int clock_gettime(int dummy, struct timespec* ct)
  {
  LARGE_INTEGER count;

  if (g_first_time) {
    g_first_time = 0;

    if (0 == QueryPerformanceFrequency(&g_counts_per_sec)) {
      g_counts_per_sec.QuadPart = 0;
    }
  }

  if ((NULL == ct) || (g_counts_per_sec.QuadPart <= 0) || (0 == QueryPerformanceCounter(&count))) {
    return -1;
}

  ct->tv_sec = count.QuadPart / g_counts_per_sec.QuadPart;
  ct->tv_nsec = ((count.QuadPart % g_counts_per_sec.QuadPart) * BILLION) / g_counts_per_sec.QuadPart;

    return 0;
}
#endif
