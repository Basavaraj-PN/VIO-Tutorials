#include <fstream>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vector>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

std::string point_cloud = "/home/omnipotent/Desktop/Desktop/vio_tutorial/resources/output.pcd";
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

int main()
{
    // Load the point cloud data from a PCD file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(point_cloud, *cloud) == -1)
    {
        std::cerr << "Failed to load PCD file." << std::endl;
        return -1;
    }

    // Create a PCLVisualizer object
    pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");

    // Add the point cloud to the viewer
    viewer.addPointCloud(cloud, "cloud");

    // Set the background color to black
    viewer.setBackgroundColor(0, 0, 0);

    // Set the point cloud color to white
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, "cloud");

    // Set the point size to 1
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

    // Display the point cloud until the viewer is closed
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }

    return 0;
}
