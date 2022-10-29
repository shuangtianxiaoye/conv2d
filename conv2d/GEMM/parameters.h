/* Input img size*/
#define PFIRST 40
#define PLAST 400
#define PINC 40

#define IC 64
#define IH -1
#define IW -1

#define OC 128
#define KH -1
#define KW -1

#define STRIDE 1
#define PADDING 0

struct INPUT_SIZE
{
	int ic;
	int ih;
	int iw;
};

struct KERNEL_SIZE
{
	int oc;
	int ic;
	int kh;
	int kw;
};

struct OUTPUT_SIZE
{
	int oc;
	int oh;
	int ow;
};

struct GEMM_SIZE
{
	int m;
	int k;
	int n;
};
