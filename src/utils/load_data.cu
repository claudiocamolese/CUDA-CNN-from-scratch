#include <cstdio>
#include "../../config.h"

/**
 * @brief Loads images from a binary file into a float array.
 * 
 * This function reads `num_images` images from the dataset file 
 * specified by `filename`. Each image is stored as `IMAGE_ROWS * IMAGE_COLS` 
 * floats normalized to [0,1]. The first 16 bytes of the file (header) are skipped.
 * 
 * @param filename Path to the image file (e.g., "train-images.idx3-ubyte").
 * @param data Pointer to a pre-allocated float array where image data will be stored.
 *             The array must have size at least `num_images * IMAGE_ROWS * IMAGE_COLS`.
 * @param num_images Number of images to read from the file.
 */
void load_images(const char* filename, float* data, int num_images){
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Failed to open file: %s\n", filename);
        exit(1);
    }
    // Skip first 16 lines since it's the header
    fseek(f, 16, SEEK_SET);

    /*
        For each image, for each pixel in the image, save the normalized element in the data array
    */
    for (int img = 0; img < num_images; img++){
        for (int pixel= 0; pixel < IMAGE_ROWS * IMAGE_COLS; pixel++){
            unsigned char p = 0;
            fread(&p, 1, 1, f); // adress of the adress in memory to fill, size in byte to read, number of elements to read, pointer of the file
            data[img * IMAGE_ROWS * IMAGE_COLS + pixel] = p / 255.0f;
        }
    }

    fclose(f);
}
/**
 * @brief Loads labels from a binary file into an integer array.
 * 
 * This function reads `num_labels` labels from the dataset file 
 * specified by `filename`. Each label is stored as an integer in the 
 * `labels` array. The first 8 bytes of the file (header) are skipped.
 * 
 * @param filename Path to the label file (e.g., "train-labels.idx1-ubyte").
 * @param labels Pointer to a pre-allocated integer array where labels will be stored.
 *               The array must have size at least `num_labels`.
 * @param num_labels Number of labels to read from the file.
 */
void load_labels(const char* filename, int* labels, int num_labels){
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Failed to open file: %s\n", filename);
        exit(1);
    }

    fseek(f, 8, SEEK_SET);

    for(int label= 0; label < num_labels; label++){
        unsigned char lb = 0;
        fread(&lb, 1, 1, f);
        labels[label] = (int)lb;
    }
}