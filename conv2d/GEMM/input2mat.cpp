#include "parameters.h"

#define INPUTC(c, h, w) inputc[((c)*input_size.ih+(h))*input_size.iw+(w)]
#define INPUTM(i, j) inputm[(i)*(w_size.ic * w_size.kh * w_size.kw)+(j)]

void input2mat(float* inputc, float* inputm, 
	struct INPUT_SIZE input_size,
	struct KERNEL_SIZE w_size,
	struct OUTPUT_SIZE output_size) {
	for (int h = 0; h < output_size.oh; h++) {
		for (int w = 0; w < output_size.ow; w++) {
			for (int c = 0; c < input_size.ic; c++) {
				for (int i = 0; i < w_size.kh; i++) {
					for (int j = 0; j < w_size.kw; j++) {
						INPUTM((h * output_size.ow + w), ((c * w_size.kh + i) * w_size.kw + j)) = INPUTC(c, (h + i), (w + j));
					}
				}
			}
		}
	}
}
