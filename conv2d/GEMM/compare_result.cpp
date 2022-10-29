#include "parameters.h"

#define OUTPUTC(n ,h, w) outputc[((n)*output_size.oh+(h))*output_size.ow+(w)]
#define OUTPUTM_T(n ,h, w) outputm_t[((n)*output_size.oh+(h))*output_size.ow+(w)]

int compare_result(float* outputc, float* outputm_t, struct OUTPUT_SIZE output_size) {

	int err = 0;
	for (int n = 0; n < output_size.oc; n++) {
		for (int h = 0; h < output_size.oh; h++) {
			for (int w = 0; w < output_size.ow; w++) {
				if (OUTPUTC(n, h, w) != OUTPUTM_T(n ,h ,w)) {
					err += 1;
				}
			}
		}
	}

	return err;
}