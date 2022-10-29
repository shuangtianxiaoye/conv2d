#include <cstdlib>
#include "parameters.h"

#define INPUTC(c, h, w) inputc[((c)*input_size.ih+(h))*input_size.iw+(w)]

void random_input(float* inputc, struct INPUT_SIZE input_size) {

	for (int c = 0; c < input_size.ic; c++) {
		for (int h = 0; h < input_size.ih; h++) {
			for (int w = 0; w < input_size.iw; w++) {
				INPUTC(c, h, w) = (float)(rand() % 256);
			}
		}
	}

}
