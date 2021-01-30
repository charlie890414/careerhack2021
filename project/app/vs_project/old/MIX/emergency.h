#ifndef __EMERGENCY_H__
#define __EMERGENCY_H__

void emergency_detect_setup(void);
int emergency_detect_loop(uint8_t* input_buf);

#endif