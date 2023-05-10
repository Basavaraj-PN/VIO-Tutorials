#include "image_dataset_handler.hpp"

Dataset_Handler::Dataset_Handler(const std::string &dataset_path, const std::string &sequence)
{
    dataset_path_ = dataset_path;
    sequence_ = sequence;
    if (!initDataSet())
    {
        std::cerr << "Bad Path: " + dataset_path + "\n", exit(EXIT_FAILURE);
    }
    sequence_ = dataset_path_ + sequence_;
    calib_ = sequence_ + "/calib.txt";
    getProjectionMatrix("P0", P0);
    getProjectionMatrix("P1", P1);
    getTranslationMatrix("Tr", Tr);
    first_left_image_ = cv::imread(sequence_ + "/image_0/000000.png", cv::IMREAD_GRAYSCALE);
    first_right_image_ = cv::imread(sequence_ + "/image_1/000000.png", cv::IMREAD_GRAYSCALE);
    img_width_ = first_left_image_.rows;
    img_height_ = first_left_image_.cols;
}

bool Dataset_Handler::initDataSet()
{
    return pathExists(dataset_path_);
}

bool Dataset_Handler::fileExists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}

bool Dataset_Handler::pathExists(const std::string &path)
{
    std::filesystem::path fs_path(path);
    return std::filesystem::exists(fs_path);
}

void Dataset_Handler::getProjectionMatrix(const std::string projection, cv::Matx34d &matrix)
{
    std::ifstream file(calib_);
    if (!file.is_open())
    {
        std::cerr << "Error: cannot open file " << calib_ << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line))
    {

        std::istringstream iss(line);
        std::string label;
        iss >> label;
        if (label.substr(0, projection.size()) == projection)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    double value;
                    iss >> value;
                    matrix(i, j) = value;
                }
            }
        }
    }
}

void Dataset_Handler::getTranslationMatrix(const std::string projection, cv::Matx34d &matrix)
{
    getProjectionMatrix(projection, matrix);
}