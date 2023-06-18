#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vector>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

class PointCloudProcessor {
public:
    PointCloudProcessor(const std::string& pointCloudPath, const std::string& imagePath)
        : point_cloud(pointCloudPath), image_path(imagePath)
    {
        Tr << 0.0004276802385584, -0.9999672484945999, -0.008084491683470999, -0.01198459927713,
            -0.007210626507497, 0.008081198471645, -0.9999413164504, -0.05403984729748,
            0.9999738645903, 0.000485948581039, -0.007206933692422, -0.2921968648686;

        P0 << 718.856, 0.0, 607.1928, 0.0,
            0.0, 718.856, 185.2157, 0.0,
            0.0, 0.0, 1.0, 0.0;
    }

    void processPointCloud()
    {
        cv::Mat image = cv::imread(image_path);
        int imwidth = image.cols;
        int imheight = image.rows;

        PointCloudT::Ptr cloud(new PointCloudT);
        if (pcl::io::loadPCDFile<PointT>(point_cloud, *cloud) == -1)
        {
            std::cerr << "Failed to load PCD file." << std::endl;
            return;
        }

        filterPointCloud(cloud, Axis::X, 0.0);
        Eigen::MatrixXf cloudMatrix = pointCloudToMatrix(cloud);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cam_xyz = Tr * (cloudMatrix.transpose());
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cam_xyz_filter = filterPointsBehindCamera(cam_xyz);
        Eigen::VectorXf depth = cam_xyz.row(2).transpose();

        for (int i = 0; i < cam_xyz_filter.cols(); ++i)
        {
            cam_xyz_filter.col(i) /= cam_xyz_filter(2, i);
        }

        Eigen::MatrixXf cam_xyz_homogeneous(cam_xyz_filter.rows() + 1, cam_xyz_filter.cols());

        cam_xyz_homogeneous.block(0, 0, cam_xyz_filter.rows(), cam_xyz_filter.cols()) = cam_xyz_filter;
        cam_xyz_homogeneous.row(cam_xyz_filter.rows()).setOnes();

        auto projection = P0 * cam_xyz_homogeneous;
        Eigen::MatrixXi pixel_coordinates = projection.cast<int>().topLeftCorner(2, projection.cols()).array().round();

        Eigen::Array<bool, Eigen::Dynamic, 1> indices = ((pixel_coordinates.row(0).array() < imwidth) &&
                                                         (pixel_coordinates.row(0).array() >= 0) &&
                                                         (pixel_coordinates.row(1).array() < imheight) &&
                                                         (pixel_coordinates.row(1).array() >= 0));

        for (int i = 0; i < indices.size(); ++i)
        {
            if (indices(i))
            {
                int x = pixel_coordinates(0, i);
                int y = pixel_coordinates(1, i);
                cv::circle(image, cv::Point(x, y), 1, cv::Scalar(227, 97, 255), -1);
            }
        }

        cv::imshow("Image with Pixel Coordinates", image);
        cv::waitKey(0);
    }

private:
    std::string point_cloud;
    std::string image_path;

    Eigen::Matrix<float, 3, 4> Tr;
    Eigen::Matrix<float, 3, 4> P0;

    enum class Axis
    {
        X,
        Y,
        Z
    };

    void filterPointCloud(PointCloudT::Ptr cloud, Axis axis, float threshold)
    {
        PointCloudT::Ptr filteredCloud(new PointCloudT);
        for (const auto& point : cloud->points)
        {
            float coordinate;
            switch (axis)
            {
                case Axis::X:
                    coordinate = point.x;
                    break;
                case Axis::Y:
                    coordinate = point.y;
                    break;
                case Axis::Z:
                    coordinate = point.z;
                    break;
                default:
                    return; // Invalid axis
            }

            if (coordinate > threshold)
            {
                filteredCloud->points.push_back(point);
            }
        }

        filteredCloud->width = filteredCloud->points.size();
        filteredCloud->height = 1;
        filteredCloud->is_dense = true;

        pcl::copyPointCloud(*filteredCloud, *cloud);
    }

    Eigen::MatrixXf pointCloudToMatrix(const PointCloudT::Ptr cloud)
    {
        Eigen::MatrixXf matrix(cloud->size(), 4);
        for (std::size_t i = 0; i < cloud->size(); ++i)
        {
            const auto& point = cloud->points[i];
            matrix(i, 0) = point.x;
            matrix(i, 1) = point.y;
            matrix(i, 2) = point.z;
            matrix(i, 3) = 1.0f;
        }
        return matrix;
    }

    Eigen::MatrixXf filterPointsBehindCamera(const Eigen::MatrixXf& cam_xyz)
    {
        Eigen::MatrixXf filtered_cam_xyz;

        std::vector<int> indices;
        for (int i = 0; i < cam_xyz.cols(); ++i)
        {
            if (cam_xyz(2, i) > 0)
            {
                indices.push_back(i);
            }
        }
        int num_points = indices.size();
        filtered_cam_xyz.resize(cam_xyz.rows(), num_points);
        for (int i = 0; i < num_points; i++)
        {
            filtered_cam_xyz.col(i) = cam_xyz.col(indices[i]);
        }

        return filtered_cam_xyz;
    }
};

int main()
{
    std::string pointCloudPath = "/home/omnipotent/Desktop/Desktop/VIO-Tutorials/resources/output.pcd";
    std::string imagePath = "/media/omnipotent/HDD/Dataset/data_odometry_gray/dataset/sequences/00/image_0/000000.png";

    PointCloudProcessor processor(pointCloudPath, imagePath);
    processor.processPointCloud();

    return 0;
}
