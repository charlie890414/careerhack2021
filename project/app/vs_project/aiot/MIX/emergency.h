#ifndef __EMERGENCY_H__
#define __EMERGENCY_H__

extern void emergency_detect_setup();
extern int emergency_detect_loop(uint8_t* input_buf);

#endif