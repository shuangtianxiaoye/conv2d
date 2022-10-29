#include "parameters.h"

#define KERNELC(n, c, h, w) kernelc[(((n)*w_size.ic+(c))*w_size.kh+(h))*w_size.kw+(w)]
#define WEIGHTM(i, j) weightm[(i)*w_size.oc+(j)]

void kernel2mat(float* kernelc, float* weightm, struct KERNEL_SIZE w_size) {
	for (int n = 0; n < w_size.oc; n++) {
		for (int c = 0; c < w_size.ic; c++) {
			for (int h = 0; h < w_size.kh; h++) {
				for (int w = 0; w < w_size.kw; w++) {
					WEIGHTM((c * w_size.kh + h) * w_size.kw + w, n) = KERNELC(n, c, h, w);
				}
			}
		}
	}
}