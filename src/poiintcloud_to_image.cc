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

std::string point_cloud = "/home/omnipotent/Desktop/Desktop/vio_tutorial/resources/output.pcd";

Eigen::Matrix<float, 3, 4> Tr;
Eigen::Matrix<float, 3, 4> P0;

// Enum to specify the coordinate axis
enum class Axis
{
    X,
    Y,
    Z
};

// Function to filter points based on a coordinate axis and threshold
void filterPointCloud(PointCloudT::Ptr cloud, Axis axis, float threshold)
{
    PointCloudT::Ptr filteredCloud(new PointCloudT);
    for (const auto &point : cloud->points)
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

// Function to convert a PointCloud to a Nx4 matrix (with homogeneous coordinates)
Eigen::MatrixXf pointCloudToMatrix(const PointCloudT::Ptr cloud)
{
    Eigen::MatrixXf matrix(cloud->size(), 4);
    for (std::size_t i = 0; i < cloud->size(); ++i)
    {
        const auto &point = cloud->points[i];
        matrix(i, 0) = point.x;
        matrix(i, 1) = point.y;
        matrix(i, 2) = point.z;
        matrix(i, 3) = 1.0f;
    }
    return matrix;
}

Eigen::MatrixXf filterPointsBehindCamera(const Eigen::MatrixXf &cam_xyz)
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
    // filtered_cam_xyz = cam_xyz(Eigen::all, indices);

    return filtered_cam_xyz;
}

int main()
{

    Tr << 0.0004276802385584, -0.9999672484945999, -0.008084491683470999, -0.01198459927713,
        -0.007210626507497, 0.008081198471645, -0.9999413164504, -0.05403984729748,
        0.9999738645903, 0.000485948581039, -0.007206933692422, -0.2921968648686;

    P0 << 718.856, 0.0, 607.1928, 0.0,
        0.0, 718.856, 185.2157, 0.0,
        0.0, 0.0, 1.0, 0.0;

    // Load the image
    cv::Mat image = cv::imread("/media/omnipotent/HDD/Dataset/data_odometry_gray/dataset/sequences/00/image_0/000000.png");
    int imwidth = image.cols;
    int imheight = image.rows;

    PointCloudT::Ptr cloud(new PointCloudT);
    if (pcl::io::loadPCDFile<PointT>(point_cloud, *cloud) == -1)
    {
        std::cerr << "Failed to load PCD file." << std::endl;
        return -1;
    }

    // Set the threshold value for the condition
    float threshold = 0.0; // Adjust this value as needed

    // Filter the point cloud based on the condition
    filterPointCloud(cloud, Axis::X, threshold);

    // Convert the point cloud to a matrix with homogeneous coordinates
    Eigen::MatrixXf cloudMatrix = pointCloudToMatrix(cloud);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cam_xyz = Tr * (cloudMatrix.transpose());
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cam_xyz_filter = filterPointsBehindCamera(cam_xyz);

    Eigen::VectorXf depth = cam_xyz.row(2).transpose();

    // Divide each element of cam_xyz by the corresponding Z valuez
    for (int i = 0; i < cam_xyz_filter.cols(); ++i)
    {
        cam_xyz_filter.col(i) /= cam_xyz_filter(2, i);
    }

    Eigen::MatrixXf cam_xyz_homogeneous(cam_xyz_filter.rows() + 1, cam_xyz_filter.cols());

    cam_xyz_homogeneous.block(0, 0, cam_xyz_filter.rows(), cam_xyz_filter.cols()) = cam_xyz_filter;
    cam_xyz_homogeneous.row(cam_xyz_filter.rows()).setOnes();

    auto projection = P0 * cam_xyz_homogeneous;

    // Turn projection points into integers for indexing
    Eigen::MatrixXi pixel_coordinates = projection.cast<int>().topLeftCorner(2, projection.cols()).array().round();

    // Limit pixel coordinates considered to those that fit on the image plane
    Eigen::Array<bool, Eigen::Dynamic, 1> indices = ((pixel_coordinates.col(0).array() < imwidth) &&
                                                     (pixel_coordinates.col(0).array() >= 0) &&
                                                     (pixel_coordinates.col(1).array() < imheight) &&
                                                     (pixel_coordinates.col(1).array() >= 0));

    for (int i = 0; i < pixel_coordinates.cols(); ++i)
    {
        int x = pixel_coordinates(0, i);
        int y = pixel_coordinates(1, i);
        cv::circle(image, cv::Point(x, y), 2, cv::Scalar(223, 97, 255), -1);
    }

    // Display the image in an OpenCV window
    cv::imshow("Image with Pixel Coordinates", image);
    cv::waitKey(0);
    // Create a PCLVisualizer object
    // pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");

    // // Add the point cloud to the viewer
    // viewer.addPointCloud(cloud, "cloud");

    // // Set the background color to black
    // viewer.setBackgroundColor(0, 0, 0);

    // // Set the point cloud color to white
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, "cloud");

    // // Set the point size to 1
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
    // // Display the point cloud until the viewer is closed
    // while (!viewer.wasStopped())
    // {
    //     viewer.spinOnce();
    // }

    return 0;
}
