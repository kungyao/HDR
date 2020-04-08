#pragma once
#include<string>
#include<vector>
#include<fstream>

#include<opencv2/opencv.hpp>
//#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define LINESIZE 256

std::vector<std::string> strSplit(const char line[LINESIZE], const char &delim) {
	std::vector<std::string> split(0);
	split.reserve(2);
	std::string tmp = "";
	for (int i = 0; i < LINESIZE; i++) {
		if (line[i] == '\0')
			break;
		if (line[i] == delim) {
			split.push_back(tmp);
			tmp.clear();
			tmp = "";
			continue;
		}
		tmp = tmp + line[i];
	}
	split.push_back(tmp);
	//for (std::string ttt : split)
	//	std::cout << ttt << std::endl;
	return split;
}

#define ImageInfo std::pair<std::vector<cv::Mat>, std::vector<double>>

class ParseTool {
public:
	static ImageInfo readImages(const std::string &sourcePath, const std::string &sourceListPath) {
		std::fstream in(sourceListPath, std::fstream::in);
		char line[LINESIZE];
		ImageInfo dataPair;
		dataPair.first.clear();
		dataPair.second.clear();
		while (in.getline(line, LINESIZE, '\n')) {
			std::vector<std::string> split = strSplit(line, ' ');
			std::string iPath = sourcePath + split[0];
			cv::Mat image = cv::imread(iPath);
			double exposeTime = log(std::stof(split[1]));
			dataPair.first.push_back(image);
			dataPair.second.push_back(exposeTime);
		}
		return dataPair;
	}

	static void writeImage(const std::vector<cv::Mat> &images, const std::string &name) {
		int i = 0;
		for (const cv::Mat &image : images) {
			std::string path = "./output/" + name + std::to_string(i++);
			cv::imwrite(path, image);
		}
	}

	static std::vector<cv::Mat> &NormalizeWidthAndHeight(std::vector<cv::Mat> &images) {
		if (images.empty())return images;
		int w = images[0].cols;
		int h = images[0].rows;
		for (int i = 1; i < images.size(); i++) {
			auto &image = images[i];
			if (image.cols == w && image.rows == h)
				continue;
			cv::resize(image, image, cv::Size(w, h), cv::InterpolationFlags::INTER_CUBIC);
		}
		return images;
	}
};

