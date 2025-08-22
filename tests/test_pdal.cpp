#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <pdal/pdal.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/Options.hpp>
#include <pdal/Stage.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/Dimension.hpp>
#include <pdal/Streamable.hpp>
#include <pdal/Filter.hpp>
#include <pdal/Reader.hpp>

using namespace pdal;

struct Point {
    Point() : x(0), y(0), z(0), r(0), g(0), b(0) {}
    Point(PointView& view, PointId idx){
        x = view.getFieldAs<double>(Dimension::Id::X, idx);
        y = view.getFieldAs<double>(Dimension::Id::Y, idx);
        z = view.getFieldAs<double>(Dimension::Id::Z, idx);
        r = view.getFieldAs<uint8_t>(Dimension::Id::Red, idx);
        g = view.getFieldAs<uint8_t>(Dimension::Id::Green, idx);
        b = view.getFieldAs<uint8_t>(Dimension::Id::Blue, idx);
    }
    float x, y, z;
    uint8_t r, g, b;
};

class TestFilter : public Filter, public Streamable
{
public:
    TestFilter()
    {}

    std::string getName() const { return "filters.test"; }


private:

    virtual bool processOne(PointRef&)
    {

        std::cout << "callbac" << std::endl;
        return true;
    }
};

bool test_pdal_read(const std::string& filename) {
    std::cout << "Testing PDAL read for: " << filename << std::endl;
    
    try {
        // Create PDAL pipeline
        pdal::StageFactory factory;
        std::string readerType = pdal::StageFactory::inferReaderDriver(filename);
        
        if (readerType.empty()) {
            std::cerr << "ERROR: Failed to infer PDAL reader driver for: " << filename << std::endl;
            return false;
        }
        
        std::cout << "Reader type: " << readerType << std::endl;
        
        // Create reader stage
        Stage& reader = *factory.createStage(readerType);
        
        // Set options
        pdal::Options options;
        options.add("filename", filename);
        reader.setOptions(options);

        #if 0
        int point_cnt = 0;
        auto point_read_cb = [&point_cnt](PointView& view, PointId idx) {
            // data.push_back(Point(view, idx));
            point_cnt++;
        };

        dynamic_cast<Reader&>(reader).setReadCb(point_read_cb);

        auto preview = reader.preview();
        if (preview.valid()) {
            std::cout << "Preview: " << preview.m_pointCount << std::endl;
        } else {
            std::cout << "No preview" << std::endl;
        }


        // Prepare and execute
        pdal::PointTable table;
        reader.prepare(table);
        reader.execute(table);
#else
        TestFilter filter;
        filter.setInput(reader);
        pdal::FixedPointTable table(100);
        filter.prepare(table);
        filter.execute(table);
#endif

        
        std::cout << "\n✅ PDAL test PASSED" << std::endl;
        return true;
        
    } catch (const pdal::pdal_error& e) {
        std::cerr << "PDAL error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    std::string filename = "/home/linh/bunny.las";
    std::cout << "PDAL Point Cloud Reader Test" << std::endl;
    std::cout << "============================" << std::endl;
    
    bool success = test_pdal_read(filename);
    
    if (success) {
        std::cout << "\n🎉 Test completed successfully!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Test failed!" << std::endl;
        return 1;
    }
}
