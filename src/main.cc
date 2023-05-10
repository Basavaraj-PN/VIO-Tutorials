#include "../include/image_dataset_handler.hpp"
#include "../include/tick_tock.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/calib3d.hpp>
enum Matcher
{
    bm,
    sgbm
};

cv::Mat compute_disparity_map(cv::Mat &left_image, cv::Mat &right_image, Matcher matcher, bool verbose)
{
    int sad_window = 6;
    int num_disparities = sad_window * 16;
    int block_size = 11;

    cv::Mat disp_left;
    cv::Ptr<cv::StereoBM> bm_matcher{nullptr};
    cv::Ptr<cv::StereoSGBM> sgbm_matcher{nullptr};

    if (matcher == Matcher::bm)
    {
        bm_matcher = cv::StereoBM::create(num_disparities, block_size);
        Timer timer;
        bm_matcher->compute(left_image, right_image, disp_left);
    }
    else if (matcher == Matcher::sgbm)
    {

        int P1 = 8 * 3 * sad_window * sad_window;
        int P2 = 32 * 3 * sad_window * sad_window;

        sgbm_matcher = cv::StereoSGBM::create(0, num_disparities, block_size);
        sgbm_matcher->setP1(8 * 3 * sad_window * sad_window);
        sgbm_matcher->setP2(32 * 3 * sad_window * sad_window);
        sgbm_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

        Timer timer;
        sgbm_matcher->compute(left_image, right_image, disp_left);
    }
    else
    {
        std::cerr << "Error: Invalid matcher specified." << std::endl;
        return cv::Mat();
    }
    // Normalize the disparity map and convert to float
    disp_left.convertTo(disp_left, CV_32FC1);
    disp_left = disp_left / 16.0;

    return disp_left;
}

cv::Mat calculate_depth_map(cv::Mat disparity_left, cv::Mat k_left, cv::Mat k_right, cv::Vec4d t_left, cv::Vec4d t_right, bool rectified)
{
    double base_line;
    double fx = k_left.at<double>(0, 0);
    std::cout << fx << std::endl;

    if (rectified)
    {
        base_line = t_right[0] - t_left[0];
    }
    else
    {
        base_line = t_left[0] - t_right[0];
    }

    cv::Mat mask_zeros, mask_neg_ones;
    cv::compare(disparity_left, 0.0, mask_zeros, cv::CMP_EQ);
    cv::compare(disparity_left, -1.0, mask_neg_ones, cv::CMP_EQ);

    // replace zeros and -1s with 0.1
    disparity_left.setTo(0.1, mask_zeros);
    disparity_left.setTo(0.1, mask_neg_ones);
    cv::Mat depth_map = cv::Mat::ones(disparity_left.size(), CV_32FC1);

    depth_map = (fx * base_line) / disparity_left;

    return depth_map;
}

void decompose_projection_matrix(cv::Matx34d p, cv::Mat &k, cv::Mat &r, cv::Vec4d &t)
{
    cv::decomposeProjectionMatrix(p, k, r, t);

    t = t / t[3];
}
cv::Mat stereo_to_depth(cv::Mat &left_image, cv::Mat &right_image,
                        cv::Matx34d P0, cv::Matx34d P1, Matcher matcher, bool rectified, bool verbose)
{

    cv::Mat disparity_map = compute_disparity_map(left_image, right_image, matcher, verbose);

    cv::Mat k_left, r_left;
    cv::Vec4d t_left;
    decompose_projection_matrix(P0, k_left, r_left, t_left);

    cv::Mat k_right, r_right;
    cv::Vec4d t_right;
    decompose_projection_matrix(P1, k_right, r_right, t_right);

    cv::Mat stereo_depth = calculate_depth_map(disparity_map, k_left, k_right, t_left, t_right, rectified);
    return stereo_depth;
}
int main()
{
    const std::string dataset_path = "/media/omnipotent/HDD/Dataset/data_odometry_gray/dataset";
    Dataset_Handler dataset_handler(dataset_path, "/sequences/00");

    cv::Mat stereo_depth = stereo_to_depth(dataset_handler.first_left_image_, dataset_handler.first_right_image_,
                                           dataset_handler.P0, dataset_handler.P1, Matcher::sgbm, true, true);

    double max_val;
    cv::minMaxLoc(stereo_depth, nullptr, &max_val);
    cv::imshow("stereo_depth", stereo_depth / max_val);
    cv::waitKey(0);
    return EXIT_SUCCESS;
}
