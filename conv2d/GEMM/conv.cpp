#include "parameters.h"

#define INPUTC(c, h, w) inputc[((c)*input_size.ih+(h))*input_size.iw+(w)]
#define KERNELC(n, c, h, w) kernelc[(((n)*w_size.ic+(c))*w_size.kh+(h))*w_size.kw+(w)]
#define OUTPUTC(n ,h ,w) outputc[((n)*output_size.oh+(h))*output_size.ow+(w)]

void conv(float* inputc, float* kernelc, float* outputc, 
	struct INPUT_SIZE input_size,
	struct KERNEL_SIZE w_size,
	struct OUTPUT_SIZE output_size,
	int stride, int padding) {

	for (int n = 0; n < output_size.oc; n++) {
		for (int h = 0; h < output_size.oh; h++) {
			for (int w = 0; w < output_size.ow; w++) {
				float tmp = 0.0;
				for (int c = 0; c < input_size.ic; c++) {
					float tmp1 = 0.0;
					for (int i = 0; i < w_size.kh; i++) {
						for (int j = 0; j < w_size.kw; j++) {
							tmp1 += INPUTC(c, (i + h), (j + w)) * KERNELC(n, c, i, j);
						}
					}
					tmp += tmp1;
				}
				OUTPUTC(n, h, w) = tmp;
			}
		}
	}
}