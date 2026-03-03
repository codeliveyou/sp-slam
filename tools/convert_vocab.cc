#include <iostream>
#include <string>
#include "Thirdparty/DBoW3/src/DBoW3.h"

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "Usage: ./convert_vocab input.yml output.dbow3" << std::endl;
        std::cerr << "  Converts a DBoW3 YAML vocabulary to fast binary format." << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    std::cout << "Loading vocabulary from YAML: " << input_file << std::endl;
    std::cout << "This will take a long time for large YAML files. Please wait..." << std::endl;

    DBoW3::Vocabulary voc;

    cv::FileStorage fs(input_file.c_str(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Could not open file: " << input_file << std::endl;
        return 1;
    }
    voc.load(fs);
    fs.release();

    std::cout << "Vocabulary loaded: " << voc.size() << " words" << std::endl;
    std::cout << "Saving as binary: " << output_file << std::endl;

    voc.save(output_file, true);

    std::cout << "Done! Use '" << output_file << "' instead of the YAML file." << std::endl;
    return 0;
}
