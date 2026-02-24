#pragma once
#include <cuda_runtime.h>

__global__ void ConvReluForward(const float* in, const float* w, const float* b, float* out, int batchSize, int inChannels, int inH, int inW, int outChannels, int kH, int kW, int outH, int outW, int stride=1, int padding=0);

__global__ void MaxPoolFlattenForward(const float* in, float* out, int batchSize, int inChannels, int inH, int inW, int poolSize, int outH, int outW);

__global__ void FCForward(const float* in, const float* w, const float* b, float* out, int batchSize, int inFeatures, int outFeatures);

__global__ void SoftmaxCrossForward(const float* logits, const int* labels, float* outLoss, float* outProb, int batchSize, int numClasses);
