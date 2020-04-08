#include<iostream>
#include<string>
#include<vector>
#include<opencv2/opencv.hpp>

#include "Parser.h"
#include "ImageUtil.h"

using namespace std;
using namespace cv;

#define SOURCE_FOLDER "./source/"

int main() {
	string sourceFolder = SOURCE_FOLDER;
	string imageList = sourceFolder + "info.txt";
	ImageInfo info = ParseTool::readImages(sourceFolder, imageList);
	info.first = ParseTool::NormalizeWidthAndHeight(info.first);
	// bgr
	vector<vector<double>> gfunction = HDRTool::gFunction(info.first, info.second, 10);
	Mat hdrImage = HDRTool::hdrImage(info.first, info.second, gfunction);
	Mat toneImage = HDRTool::toneImage(hdrImage, 0.6);
	return 0;
}