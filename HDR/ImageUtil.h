#pragma once
#include<vector>
#include<direct.h>

#include<opencv2/opencv.hpp>
//#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class HDRTool {
private:
	static double zMin;
	static double zMax;
	
	static double weightFunction(const double &val) {
		double half = (zMax + zMin) / 2;
		return val <= half ? val - zMin : zMax - val;
	}

	static std::vector<cv::Vec3d> sampleImage(const cv::Mat &image, const int &xSample = 5, const int &ySample = 10) {
		int h = image.rows;
		int w = image.cols;
		int hStep = h / ySample;
		int wStep = w / xSample;
		std::vector<cv::Vec3d> samples(0);
		samples.reserve(xSample * ySample);
		for (int i = 0; i < ySample; i++) {
			for (int j = 0; j < xSample; j++) {
				samples.push_back(image.at<cv::Vec3b>(i * hStep, j * wStep));
			}
		}
		return samples;
	}

	static std::vector<double> gsolve(const std::vector<std::vector<double>> &imageSamples, const std::vector<double> &exposureTimes, const double &lamdba) {
		int n = 256;

		int imageSize = imageSamples.size();
		int sampleSize = imageSamples[0].size();
		
		cv::Mat A = cv::Mat::zeros(imageSize * sampleSize + 1 + n, n + sampleSize, CV_64F);
		cv::Mat b = cv::Mat::zeros(A.rows, 1, CV_64F);

		int k = 0;

		for (int i = 0; i < imageSize; i++) {
			for (int j = 0; j < sampleSize; j++) {
				double wij = weightFunction(imageSamples[i][j]);
				A.at<double>(k, (int)imageSamples[i][j]) = wij;
				A.at<double>(k, n + j) = -wij;
				b.at<double>(k, 0) = wij * exposureTimes[i];
				k++;
			}
		}

		A.at<double>(k, 129) = 1;
		k++;

		for (int i = 0; i < n - 2; i++) {
			double w = weightFunction(i + 1);
			A.at<double>(k, i) = lamdba * w;
			A.at<double>(k, i + 1) = -2 * lamdba * w;
			A.at<double>(k, i + 2) = lamdba * w;
			k++;
		}

		cv::Mat x = cv::Mat::zeros(A.cols, 1, CV_64F);

		//cv::Mat u = cv::Mat::zeros(A.rows, A.rows, CV_64F);
		//cv::Mat s = cv::Mat::zeros(A.rows, A.cols, CV_64F);
		//cv::Mat vh = cv::Mat::zeros(A.cols, A.cols, CV_64F);
		//cv::SVD::compute(A, s, u, vh, 0);
		//x = vh.t() * cv::Mat::diag(1 / s) * u.t() * b;

		cv::solve(A, b, x, cv::DECOMP_SVD);
		std::vector<double> xdouble(0);
		xdouble.reserve(n);
		for (int i = 0; i < n; i++) {
			xdouble.push_back(x.at<double>(i, 0));
			// std::cout << xdouble[i] << std::endl;
		}

		return xdouble;
	}
public:
	static std::vector<std::vector<double>> gFunction(const std::vector<cv::Mat> &images, const std::vector<double> &exposureTimes, const double &lamdba) {
		int xSample = 5;
		int ySample = 10;
		std::vector<std::vector<cv::Vec3d>> imageSamples(0);
		imageSamples.reserve(images.size());
		for (const auto &image : images) 
			imageSamples.push_back(sampleImage(image, xSample, ySample));

		std::vector<std::vector<double>> rSamples(0);
		std::vector<std::vector<double>> gSamples(0);
		std::vector<std::vector<double>> bSamples(0);
		rSamples.reserve(imageSamples.size());
		gSamples.reserve(imageSamples.size());
		bSamples.reserve(imageSamples.size());

		int sampleSize = xSample * ySample;
		for (const auto &imageSample : imageSamples) {
			std::vector<double> tmpr(0);
			std::vector<double> tmpg(0);
			std::vector<double> tmpb(0);
			for (const auto &pixel : imageSample) {
				tmpr.push_back(pixel[2]);
				tmpg.push_back(pixel[1]);
				tmpb.push_back(pixel[0]);
			}
			rSamples.push_back(tmpr);
			gSamples.push_back(tmpg);
			bSamples.push_back(tmpb);
		}

		std::vector<std::vector<double>> gfunction(0);
		gfunction.reserve(3);
		gfunction.push_back(gsolve(bSamples, exposureTimes, lamdba));
		gfunction.push_back(gsolve(gSamples, exposureTimes, lamdba));
		gfunction.push_back(gsolve(rSamples, exposureTimes, lamdba));

		return gfunction;
	}

	static cv::Mat hdrImage(const std::vector<cv::Mat> &images, const std::vector<double> &exposureTimes, const std::vector<std::vector<double>> &gfunction) {
		int h = images[0].rows;
		int w = images[0].cols;
		
		cv::Mat e = cv::Mat::zeros(h, w, CV_64FC3);
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				for (int c = 0; c < 3; c++) {
					double weight = 0;
					double tmp_le = 0;
					for (int k = 0; k < images.size(); k++) {
						const auto &image = images[k];
						const int &channelVal = image.at<cv::Vec3b>(i, j)[c];
						double tmpWeight = weightFunction(channelVal);
						weight += tmpWeight;
						tmp_le += tmpWeight * (gfunction[c][channelVal] - exposureTimes[k]);
					}
					tmp_le = exp(tmp_le / (1 + weight));
					e.at<cv::Vec3d>(i, j)[c] = tmp_le;
				}
			}
		}

		_mkdir("./output/");
		cv::imwrite("./output/hdrImage.hdr", e);
		return e;
	}

	static cv::Mat toneImage(const cv::Mat &hdrImage, const double &a = 0.5) {
		int h = hdrImage.rows;
		int w = hdrImage.cols;

		double lwhite = 0;
		double meanVal = 0;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				const auto &pixel = hdrImage.at<cv::Vec3d>(i, j);
				// printf("%lf %lf %lf\n", pixel[2], pixel[1], pixel[0]);
				double gray = 0.299*log(1 + pixel[2]) + 0.587*log(1 + pixel[1]) + 0.114*log(1 + pixel[0]);
				meanVal += gray;
				if (gray > lwhite)
					lwhite = gray;
			}
		}

		int imageSize = h * w;
		meanVal = exp(meanVal / imageSize);

		auto Clamp = [&](double val) {
			if (val < 0)return 0.0;
			else if (val > 255) return 255.0;
			return val;
		};

		double adm = a / meanVal;
		cv::Mat ldrImage = cv::Mat::zeros(h, w, CV_8UC3);
		lwhite *= lwhite;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				const auto &pixel = hdrImage.at<cv::Vec3d>(i, j);
				for (int k = 0; k < 3; k++) {
					double lm = adm * pixel[k];
					double ld = (lm * (1 + lm / lwhite)) / (1 + lm);
					ldrImage.at<cv::Vec3b>(i, j)[k] = Clamp(ld * 255);
				}
			}
		}

		_mkdir("./output/");
		cv::imwrite("./output/ldrImage.jpg", ldrImage);
		return ldrImage;
	}
};

double HDRTool::zMin = 0;
double HDRTool::zMax = 256;
