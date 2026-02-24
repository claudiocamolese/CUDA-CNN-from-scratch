#include <iostream>
#include <iomanip> // library used for printing in terminal 


/**
 * @brief Print the progress bar during the training
 * 
 * @param current current batch idx
 * @param total total number of batches
 * @param epoch current epoch
 * @param epochs total number of epochs
 */
void print_progress(int current, int total, int epoch, int epochs) {
    const int bar_width = 40;
    float progress = (float)current / total;
    int pos = bar_width * progress;

    std::cout << "\rEpoch [" << epoch + 1 << "/" << epochs << "] [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0f) << "%";
    std::cout.flush();
}

/**
 * @brief Print in terminal the loss of the current epoch
 * 
 * @param avg_loss loss of the epoch
 */
void print_epoch_end(float avg_loss) {
    std::cout << " | avg loss = "
              << std::fixed << std::setprecision(6)
              << avg_loss << std::endl;
}
