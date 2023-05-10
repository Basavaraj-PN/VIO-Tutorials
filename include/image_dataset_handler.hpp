#ifndef IMAGE_HANDLER_H
#define IMAGE_HANDLER_H
#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>
class Dataset_Handler
{
public:
    // Constructor
    Dataset_Handler(const std::string &dataset_path, const std::string &sequence);
    bool initDataSet();
    bool fileExists(const std::string &filename);
    bool pathExists(const std::string &path);
    void getProjectionMatrix(const std::string projection, cv::Matx34d &matrix);
    void getTranslationMatrix(const std::string projection, cv::Matx34d &matrix);

public:
    cv::Matx34d P0, P1;
    cv::Matx34d Tr;
    cv::Mat first_left_image_;
    cv::Mat first_right_image_;
    uint32_t num_frames_;
    int img_height_;
    int img_width_;

private:
    std::string sequence_;
    std::string dataset_path_;
    std::string seq_dir_;
    std::string pose_dir_;
    std::string calib_;
};

#endif // MY_HEADER_FILE_H