#include <iostream>
#include <chrono>
#include <thread>
#include <cassert>
#include "point_cloud_2.h"

bool test_point_cloud_2_async_loading() {
    PointCloud2 cloud;
    
    const std::string filename = "/home/linh/bunny.las";
    
    auto result = cloud.load_from_file_async(filename);

    assert(result.get());

    auto final_points = cloud.get_num_points();
    assert(final_points > 0);

    return true;
}


int main() {
    if (!test_point_cloud_2_async_loading()) {
        std::cout << "💥 Test FAILED!" << std::endl;
        return 1;
    }
    
    std::cout << "🎉 All tests PASSED!" << std::endl;
    return 0;
} 