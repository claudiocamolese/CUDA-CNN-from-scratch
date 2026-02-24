
#include <cuda_runtime.h>

__global__ void SoftmaxCrossBackward(float* gradLogits, const float* prob, const int* labels, int batchSize, int numClasses);

__global__ void FCParamBackward(const float* gradOut, const float* in, float* gradW, float* gradB, int batchSize, int inFeatures, int outFeatures);

__global__ void FCGradBackward(const float* gradOut, const float* w, float* gradIn, int batchSize, int inFeatures, int outFeatures);

__global__ void ReLUBackward(const float* gradOut, const float* x, float* gradIn, int n);

__global__ void MaxPollBackward(const float* convOut, const float* gradFlat, float* gradConvOut, int batchSize, int inChannels, int inH, int inW, int poolSize, int outH, int outW);

__global__ void ConvWeightBackward(const float* in, const float* gradConvOut, float* gradW, float* gradB, int batchSize, int inChannels, int inH, int inW, int outChannels, int kH, int kW, int outH, int outW);

__global__ void ConvLayerBackward(const float* gradConvOut, const float* w, float* gradIn, int batchSize, int inChannels, int inH, int inW, int outChannels, int kH, int kW, int outH, int outW);

__global__ void SGDBackward(float* param, const float* grad, float lr, int n);