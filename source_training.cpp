/*This program is the test phase uses Dense SIFT algorithm
	with DPM in recognizing the person
*/
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "flandmark_detector.h"
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include<sstream>
#include<fstream>
#include<Windows.h>


using namespace std;
using namespace cv;

/*Function to read the images(obtained as a result of source_video_train.cpp)
  from the folder and store it as vector
*/
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
	Mat img;int i=0;
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
			img=imread(path);
			
			images.push_back(img);
			cout<<"image"<<i++<<endl;
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}


int main () {
	
	//declaring the variables and loading the models
	Mat gray1;
	CascadeClassifier front;
	FLANDMARK_Model *model;
	model = flandmark_init("flandmark_model.dat");
	Mat img;
	vector <Rect> faces;
	Mat gray;
	front.load("haarcascade_frontalface_alt.xml");
    string fn_csv = "training.csv";
    int deviceId = 0;
    vector<Mat> images;
    vector<int> labels;

	//Getting Images for training
	try 
	{
		read_csv(fn_csv, images, labels);
    } 
	catch (cv::Exception& e) 
	{
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }
	
		Mat data(images.size(),128*90,CV_32FC1);	 
		Mat data1(images.size(),128*90,CV_32FC1);	 
		Mat data2(images.size(),128*90,CV_32FC1);
		Mat data3(images.size(),128*400,CV_32FC1);
		cout<<images.size();
		double t,featurepts[20];
		for(size_t j=0;j<images.size();j++)
		{ 
			if(!images[j].empty())
				{


					img=images[j];
					t = (double)cvGetTickCount();
					cout<<"Got the image at"<<img.cols<<"x"<<img.rows<<" resolution..\n";
		
					//conversion to gray scale
					cvtColor(img,gray,CV_BGR2GRAY);

					//Face detection
					front.detectMultiScale(gray,faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(20,20) );

					 if (!faces.empty()){
						for (size_t i=0;i<faces.size();i++){
 							int bbox[4];
							bbox[0] = faces[i].x;
							bbox[1] = faces[i].y;
							bbox[2] = faces[i].x + faces[i].width;
							bbox[3] = faces[i].y + faces[i].height;
							Rect crop(Point (bbox[0],bbox[1]),Point (bbox[2],bbox[3]));
							Point a1(bbox[0],bbox[1]);
							Point a2(bbox[2],bbox[3]);
							rectangle(img,a1,a2,Scalar(255,0,255),4,8,0);	
								
							//Applying DPM model
					 		IplImage imgg=(IplImage) gray ;
							flandmark_detect(&imgg,bbox,model,featurepts);
							Mat img2=gray(crop);
							imshow("ss",img);waitKey(30);
 
							//Identifying the patches
							int EyeRatioL=(int(featurepts[2])-int(featurepts[10]))/2;
							int EyeRatioR=(int(featurepts[4])-int(featurepts[12]))/2;
							int MouthRatio=(int(featurepts[8])-int(featurepts[6]))/2;	
							int xL1=int(featurepts[10])-EyeRatioL,yL1=int(featurepts[11])-EyeRatioL;
							int xL2=int(featurepts[2])+EyeRatioL,yL2=int(featurepts[3])+EyeRatioL;
							int xR1=int(featurepts[12])-EyeRatioR,yR1=int(featurepts[13])-EyeRatioR;
							int xR2=int(featurepts[4])+EyeRatioR,yR2= int(featurepts[5])+EyeRatioR;
							int xLm1=int(featurepts[6])-MouthRatio,yLm1=int(featurepts[7])-MouthRatio;
							int xLm2=int(featurepts[8])+MouthRatio,yLm2=int(featurepts[9])+MouthRatio;
							
							Rect ROIl(Point(xL1,yL1),Point(xL2,yL2)); 
							Rect ROIR(Point(xR1,yR1),Point(xR2,yR2));
							Rect ROIM(Point(xLm1,yLm1),Point(xLm2,yLm2));

							Mat eyeL=gray(ROIl);
							Mat eyeR=gray(ROIR);
							Mat Mouth=gray(ROIM);

							resize(eyeL,eyeL,Size(150,60));
							resize(eyeR,eyeR,Size(150,60));
						    resize(Mouth,Mouth,Size(150,60));
							resize(img2,img2,Size(200,200));
							imshow("img2",img2);
							
							normalize(eyeL, eyeL, 0, 255, cv::NORM_MINMAX);
							normalize(eyeR, eyeR, 0, 255, cv::NORM_MINMAX);
							normalize(Mouth, Mouth, 0, 255, cv::NORM_MINMAX);
							normalize(img2, img2, 0, 255, cv::NORM_MINMAX);

							//Identifying the Dense SIFT keypoints
							DenseFeatureDetector detector(12.0f, 1, 0.1f, 10);
							vector<KeyPoint> keypoints,keypoints1,keypoints2,keypoints3;
							detector.detect(eyeL, keypoints);
							detector.detect(eyeR, keypoints1);
							detector.detect(Mouth, keypoints2);
							detector.detect(img2, keypoints3);
							
							//Extracting the descriptors	
							SiftDescriptorExtractor extractor;
							Mat descriptors(90,128,CV_32FC1);
							Mat descriptors1(90,128,CV_32FC1);
							Mat descriptors2(90,128,CV_32FC1);
							Mat descriptors3(400,128,CV_32FC1);;
							extractor.compute(eyeL,keypoints,descriptors);
							extractor.compute(eyeR,keypoints1,descriptors1);
							extractor.compute(Mouth,keypoints2,descriptors2);
							extractor.compute(img2,keypoints3,descriptors3);
							
							//conversion from Mat to Array
							int r=0;
							for(int p=0;p<90;p++)
							{
								for(int q=0;q<128;q++)
								{
									data.at<float>(j,r)=descriptors.at<float>(p,q);
									data1.at<float>(j,r)=descriptors1.at<float>(p,q);
									data2.at<float>(j,r)=descriptors2.at<float>(p,q);r++;
								}
							}
	 						r=0;
							for(int p=0;p<400;p++)
							{
								for(int q=0;q<128;q++)
								{
									data3.at<float>(j,r)=descriptors3.at<float>(p,q);
								}
							}
	 
							cout<<"updated"<<j<<endl;
   }
  }
 }
}
	
	//Converting the labels of the class to array
	Mat l1(labels.size(),1,CV_32FC1);
	for(int i=0;i<labels.size();i++) l1.at<float>(i,0)=labels[i];
	cout<<l1;
	
	//SVM parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	params.C=1000;
	//const CvMat data1=data;
	//const CvMat lab=l1;
	CvSVM SVM1,SVM2,SVM3,SVM4;
	cout<<"Training started"<<endl;

	//Training the SVM
	SVM1.train(data,l1,Mat(),Mat(),params);
	SVM1.save("dsift_new_eyeL1_caltech.yaml");
	SVM2.train(data1,l1,Mat(),Mat(),params);
	SVM2.save("dsift_new_eyeR1_caltech.yaml");
	SVM3.train(data2,l1,Mat(),Mat(),params);
	SVM3.save("dsift_new_mouth_caltech.yaml");
	SVM4.train(data3,l1,Mat(),Mat(),params);
	SVM4.save("dsift_new_face_caltech.yaml");
	cout<<"trained";
	waitKey(30);
	
	return 0;
}

  	