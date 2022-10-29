#include "parameters.h"

#define OUTPUTM(i, j) outputm[(i)*output_size.oc+(j)]
#define OUTPUTM_T(n ,h, w) outputm_t[((n)*output_size.oh+(h))*output_size.ow+(w)]

void transpose(float* outputm, float* outputm_t, struct OUTPUT_SIZE output_size) {

	for (int n = 0; n < output_size.oc; n++) {
		for (int h = 0; h < output_size.oh; h++) {
			for (int w = 0; w < output_size.ow; w++) {
				OUTPUTM_T(n, h, w) = OUTPUTM((h * output_size.ow + w), n);
			}
		}
	}

}