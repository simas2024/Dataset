//
//  DatasetGeneratorTest.m
//  DatasetGeneratorTest
//
//  Created by Markus Schmid on 27.03.25.
//

#include <opencv2/opencv.hpp>
#import <XCTest/XCTest.h>
#include "DatasetGenerator.hpp"

@interface DatasetGeneratorTest : XCTestCase

@end

using namespace cv;
using namespace std;
using namespace masc::annotations;
namespace fs = std::filesystem;


@implementation DatasetGeneratorTest

DatasetGenerator gen;
cv::Mat maske_bw;
cv::VideoCapture cap;

std::string set_path;
std::string clip_path;
std::string json_path;
std::string maske_bw_path;
std::string debug_path;
std::string yolo_path;
ErkennungKonfig config;

- (void)setUp {
    config.debug = 0;
    config.gauss_deviation = 17;
    config.thresh_thresh = 16.0;
    config.boundingbox_min_area = 700;
    config.kontrast_alpha = 0.9;
    config.kontrast_beta = 0.0;
    
    std::cout << "🏁 Starte Testcase: BoundingBox-Anzahl..." << std::endl;
    
    
    set_path = std::string(PROJECT_SOURCE_DIR) + "/Test/data/";
    clip_path = set_path + "clip.mov";
    maske_bw_path = set_path + "maske01bw640x424.png";
    json_path = set_path + "clip.json";
    debug_path = set_path + "debug/";
    yolo_path = set_path + "yolo/";
    cap.open(clip_path);
    XCTAssert(cap.isOpened());
    maske_bw = cv::imread(maske_bw_path, cv::IMREAD_GRAYSCALE);
    XCTAssert(!maske_bw.empty());
    gen.initReferenzFrames(json_path, clip_path, maske_bw, config);
    fs::create_directories(debug_path);
}

- (void)tearDown {
    cap.release();
}

- (void)testReferenzFrameLaden {
    cv::Mat ref_frame = gen.ladeReferenzFrame(clip_path, 0, maske_bw, config);
    XCTAssertEqual((bool)ref_frame.empty(),false, @"Referenz Frame %d wurde nicht geöffnet", 0);
}
    
- (void)testBoundingBoxFrame {
    std::string test_debug_path=debug_path+"testBoundingBoxFrame/";
    fs::create_directories(test_debug_path);
    std::vector<std::tuple<int, int, int>> frameDaten = {
        {26059, 2, 3}
    };
   
    for (const auto& [frame_index, erwarteteAnzahlMin, erwarteteAnzahlMax] : frameDaten) {
        int refIdx = gen.referenzFrameFuerIndex(json_path, frame_index);
        cv::Mat ref_frame = DatasetGenerator::getReferenzFrame(refIdx);
        XCTAssertEqual((bool)ref_frame.empty(),false, @"Referenz Frame %d für Frame %d wurde nicht geöffnet", refIdx, frame_index);
        auto [konturen, boxen, gefilterte_boxen, frame] = gen.boundingBoxes(cap, frame_index, maske_bw, ref_frame, config);
        int n = (int)gefilterte_boxen.size();
        XCTAssertTrue(n >= erwarteteAnzahlMin && n <= erwarteteAnzahlMax, "Frame:%d: Anzahl der Boxen ist %d und liegt nicht im erwarteten Bereich (%d – %d)",frame_index,n,erwarteteAnzahlMin,erwarteteAnzahlMax );
        gen.debugDataset(konturen, boxen, gefilterte_boxen, ref_frame, frame, frame_index, refIdx, test_debug_path, config);
    }
}

- (void)testBoundingBoxFrameSerie {
    std::string test_debug_path=debug_path+"testBoundingBoxFrameSerie/";
    fs::create_directories(test_debug_path);
    std::vector<std::tuple<int, int, int>> frameDaten = {
        {914, 0, 2}, {10423, 2, 2}, {10447, 2, 3}, {26059, 2, 3}, {35456, 1, 3}, {35683, 2, 3}, {35780, 2, 2}
    };
   
    for (const auto& [frame_index, erwarteteAnzahlMin, erwarteteAnzahlMax] : frameDaten) {
        int refIdx = gen.referenzFrameFuerIndex(json_path, frame_index);
        cv::Mat ref_frame = gen.getReferenzFrame(refIdx);
        XCTAssertEqual((bool)ref_frame.empty(),false, @"Referenz Frame %d für Frame %d wurde nicht geöffnet", refIdx, frame_index);
        auto [konturen, boxen, gefilterte_boxen, frame] = gen.boundingBoxes(cap, frame_index, maske_bw, ref_frame, config);
        int n = (int)gefilterte_boxen.size();
        XCTAssertTrue(n >= erwarteteAnzahlMin && n <= erwarteteAnzahlMax, "Frame:%d: Anzahl der Boxen ist %d und liegt nicht im erwarteten Bereich (%d – %d)",frame_index,n,erwarteteAnzahlMin,erwarteteAnzahlMax );
        gen.debugDataset(konturen, boxen, gefilterte_boxen, ref_frame, frame, frame_index, refIdx, test_debug_path, config);
    }
}

@end
