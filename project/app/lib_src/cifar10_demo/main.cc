#include "main_functions.h"

extern "C" int cifar10_demo(void) {
       int i = 3;

        cifar10_setup();
        while(i-- > 0) {
                cifar10_loop();
        }
        return 0;
}