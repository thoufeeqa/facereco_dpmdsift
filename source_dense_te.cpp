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
#include <sstream>
#include <fstream>


using namespace std;
using namespace cv;

int main () {

	//declaring variables and loading models

	Mat gray1; //grayscale matrix to hold images converted from color to grayscale
	CascadeClassifier front; //load cascade classifier trained on frontal faces
	FLANDMARK_Model *model; // deformable part model for facial feature extraction
	model = flandmark_init("flandmark_model.dat");
	Mat img; //matrix object to hold source image frame
	vector <Rect> faces;
	Mat gray; //matrix object to hold images converted from color to grayscale
	front.load("haarcascade_frontalface_alt.xml");  //haar cascade model for face detection
    //string fn_csv = "new.csv";
    int deviceId = 0;
    
	vector<Mat> images;
    vector<int> labels;
	
	//support vector machines used to classify and recognize faces
	CvSVM SVM1,SVM2,SVM3,SVM4;        
	SVM1.load("dsift_new_eyeL1.yaml");
	SVM2.load("dsift_new_eyeR1.yaml");
	SVM3.load("dsift_new_mouth.yaml");
	SVM4.load("dsift_new_face.yaml");
    
	// Matrices to hold descriptor data
	Mat data(1,128*90,CV_32FC1);	 
	Mat data1(1,128*90,CV_32FC1);	
	Mat data2(1,128*90,CV_32FC1);
	Mat data3(1,128*400,CV_32FC1);
	
	double t,featurepts[20];

	//Video based recognition
	VideoCapture cap("VIDEO0015.mp4"); //test video capture used, can be substuted for realtime source (webcam)
	while(1)
	{
		
		cap>>img;

	//Uncomment for image-based recogition
		/*int k=-1;
	while(k++<0){
	stringstream ss;
	ss<<"C:/Users/thoufeeq/Desktop/test/";
	ss<<k;
	ss<<".jpg";
	img=imread(ss.str());*/

	if(img.empty())continue;
	t = (double)cvGetTickCount();
	cout<<"Got the image at"<<img.cols<<"x"<<img.rows<<" resolution..\n";
		
	//conversion to gray scale
	cvtColor(img,gray,CV_BGR2GRAY);

	//Face detection ~ detects if a face is present, and the number of faces present in a frame (uses haar cascade)
	front.detectMultiScale(gray,faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(20,20) );
	
	if (!faces.empty()){

		 for (size_t i=0;i<faces.size();i++){
	 
 			int bbox[4]; //bounding box for each face
			bbox[0] = faces[i].x;
			bbox[1] = faces[i].y;
			bbox[2] = faces[i].x + faces[i].width;
			bbox[3] = faces[i].y + faces[i].height;
			Rect crop(Point (bbox[0],bbox[1]),Point (bbox[2],bbox[3])); //crop faces from frame ~ recognition done on these pixels only
			Point a1(bbox[0],bbox[1]);
			Point a2(bbox[2],bbox[3]);
			//	stringstream kk;kk<<"C:/Users/thoufeeq/Documents/Visual Studio 2012/Projects/Project4/Project4/out/face/";kk<<k;kk<<".jpg";
			rectangle(img,a1,a2,Scalar(255,0,255),4,8,0);	
			//imshow("a",img);
			//imwrite(kk.str(),img);
			Mat img2=gray(crop);
			// if(!img2.empty()){ imshow("nn",img2);}
			
			//Applying Deformable Parts Model to get feature points
	 		IplImage imgg=(IplImage) gray ;
			flandmark_detect(&imgg,bbox,model,featurepts);
		
			//Identifying the patches
			int EyeRatioL=(int(featurepts[2])-int(featurepts[10]))/2;
			int EyeRatioR=(int(featurepts[4])-int(featurepts[12]))/2;
			int xL1=int(featurepts[10])-EyeRatioL,yL1=int(featurepts[11])-EyeRatioL;
			int xL2=int(featurepts[2])+EyeRatioL,yL2=int(featurepts[3])+EyeRatioL;
			int xR1=int(featurepts[12])-EyeRatioR,yR1=int(featurepts[13])-EyeRatioR;
			int xR2=int(featurepts[4])+EyeRatioR,yR2= int(featurepts[5])+EyeRatioR;
			int MouthRatio=(int(featurepts[8])-int(featurepts[6]))/2;
			int xLm1=int(featurepts[6])-MouthRatio,yLm1=int(featurepts[7])-MouthRatio;
			int xLm2=int(featurepts[8])+MouthRatio,yLm2=int(featurepts[9])+MouthRatio;
			
			//isolate regions of interest
			Rect ROIl(Point(xL1,yL1),Point(xL2,yL2)); 
			Rect ROIR(Point(xR1,yR1),Point(xR2,yR2));
			Rect ROIM(Point(xLm1,yLm1),Point(xLm2,yLm2));
			
			Mat eyeL=gray(ROIl);
			Mat eyeR=gray(ROIR);
			Mat Mouth=gray(ROIM); 
			//imshow("Sf",Mouth);
			
			//resizing and normalizing ~
			resize(eyeL,eyeL,Size(150,60));
			resize(eyeR,eyeR,Size(150,60));
			resize(Mouth,Mouth,Size(150,60));
			resize(img2,img2,Size(200,200));
			
			rectangle(img,ROIl,Scalar(0,0,255));
			rectangle(img,ROIR,Scalar(0,0,255));
			rectangle(img,ROIM,Scalar(0,0,255));
			
			normalize(eyeL, eyeL, 0, 255, cv::NORM_MINMAX);
			normalize(eyeR, eyeR, 0, 255, cv::NORM_MINMAX);
			normalize(Mouth, Mouth, 0, 255, cv::NORM_MINMAX);
			normalize(img2, img2, 0, 255, cv::NORM_MINMAX);
				//imshow("eye",eyeL);
			double t = (double)cvGetTickCount();				
			
			//Identifying the Dense SIFT keypoints
			DenseFeatureDetector detector(12.0f, 1, 0.1f, 10); //detector params
			vector<KeyPoint> keypoints,keypoints1,keypoints2,keypoints3;
			detector.detect(eyeL, keypoints);
			detector.detect(eyeR, keypoints1);
			detector.detect(Mouth, keypoints2);
			detector.detect(img2,keypoints3);
			t = (double)cvGetTickCount() - t;	
			
			//Extracting the descriptors	
			SiftDescriptorExtractor extractor;
			Mat descriptors(90,128,CV_32FC1);
			Mat descriptors1(90,128,CV_32FC1);
			Mat descriptors2(90,128,CV_32FC1);
			Mat descriptors3(400,128,CV_32FC1);
			extractor.compute(eyeL,keypoints,descriptors);
			extractor.compute(eyeR,keypoints1,descriptors1);
			extractor.compute(Mouth,keypoints2,descriptors2);
			extractor.compute(img2,keypoints3,descriptors3);
			
			//conversion from Matrix to Array
			int r=0;
			for(int p=0;p<90;p++)
			{
				for(int q=0;q<128;q++)
					{
						data.at<float>(0,r)=descriptors.at<float>(p,q);
						data1.at<float>(0,r)=descriptors1.at<float>(p,q);
						data2.at<float>(0,r)=descriptors2.at<float>(p,q);
						r++;
					}
			}
			r=0;
			for(int p=0;p<400;p++)
			{
				for(int q=0;q<128;q++)
				{
					data3.at<float>(0,r)=descriptors3.at<float>(p,q);
					r++;
				}
			}
			
			// Use support vector machines to classify if faces belong in authorized set or not
			float prediction=SVM1.predict(data,90*128);
			float prediction1=SVM2.predict(data1,90*128);
			float prediction2=SVM3.predict(data2,90*128);
			float prediction3=SVM4.predict(data3,400*128);
	 		
			CvFont font;
			cvInitFont(&font,CV_FONT_NORMAL, 1.0, 1.0, 0, 1, CV_AA);
			int predict;
			
			//rule set-1
			if(prediction==prediction1) predict=prediction;
			else if(prediction==prediction2) predict=prediction;
			else if(prediction1=prediction2) predict=prediction1;
			else predict=200;
			
			//rule set-2
			/* if ((prediction==prediction1)&&(prediction1=prediction2)&&(prediction2=prediction)) predict=prediction;
			 else predict=200;*/
				 
			cout<<prediction<<prediction1<<prediction2<<prediction3;

			//Displaying the results
			 switch(predict)
			{
			 case 0:putText(img,"person1",Point(bbox[0]-20,bbox[1]),3,2,Scalar(255,255,0));break;
			 case 1:putText(img,"person2",Point(bbox[0]-20,bbox[1]),3,2,Scalar(255,255,0));break;
			 case 2:putText(img,"person3",Point(bbox[0]-20,bbox[1]),3,2,Scalar(255,255,0));break;
			 case 3:putText(img,"person4",Point(bbox[0]-20,bbox[1]),3,2,Scalar(255,255,0));break;
			 default:putText(img,"unknown",Point(bbox[0]-20,bbox[1]),3,2,Scalar(255,255,0));break;
			}
			
			 //time calculation
			int ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) ); char fps[50];
			//   sprintf(fps, "%.2f fps", 1000.0/( t/((double)cvGetTickFrequency() * 1000.0) ) );
            //   putText(img, fps, Point(10, 40),3,2, Scalar(255, 0, 0));
		   
			imshow("ss",img);imshow("ss1",eyeL);imshow("ss2",eyeR);imshow("ss3",Mouth);waitKey(20);

			
   }
  }
 }
 
	return 0;
}

  	