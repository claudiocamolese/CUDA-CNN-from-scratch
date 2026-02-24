#pragma once

// ---------------------------------------------------------------------------------
// Dataset parameters
// ---------------------------------------------------------------------------------
#define TRAIN_IMAGES    60000
#define TEST_IMAGES     10000
#define IMAGE_ROWS      28
#define IMAGE_COLS      28
#define NUM_CLASSES     10
#define NUM_CHANNELS    1   // input channels

// ---------------------------------------------------------------------------------
// Convolution layers
// ---------------------------------------------------------------------------------
#define FILTER_SIZE           3

// First convolution
#define FIRST_INPUT_CHANNELS   NUM_CHANNELS
#define FIRST_OUTPUT_CHANNELS  16
#define CONV1_OUT_ROWS         (IMAGE_ROWS - FILTER_SIZE + 1)
#define CONV1_OUT_COLS         (IMAGE_COLS - FILTER_SIZE + 1)

// Second convolution
#define SECOND_INPUT_CHANNELS  FIRST_OUTPUT_CHANNELS
#define SECOND_OUTPUT_CHANNELS 32
#define CONV2_OUT_ROWS         (CONV1_OUT_ROWS - FILTER_SIZE + 1)
#define CONV2_OUT_COLS         (CONV1_OUT_COLS - FILTER_SIZE + 1)

// ---------------------------------------------------------------------------------
// Max Pooling layer (
// ---------------------------------------------------------------------------------
#define POOL_SIZE              2
#define POOL_OUT_ROWS          (CONV2_OUT_ROWS / POOL_SIZE)
#define POOL_OUT_COLS          (CONV2_OUT_COLS / POOL_SIZE)
#define FLATTEN_SIZE           (SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS)

// ---------------------------------------------------------------------------------
// Training hyperparameters
// ---------------------------------------------------------------------------------
#define EPOCHS                 5
#define BATCH_SIZE             64
#define LEARNING_RATE          0.01f
#define NUM_BATCHES            TRAIN_IMAGES / BATCH_SIZE
#define NUM_BUFFERS            3

// ---------------------------------------------------------------------------------
// CUDA configuration
// ---------------------------------------------------------------------------------
#define BLOCK_SIZE             256
