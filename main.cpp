#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

// we're NOT "using namespace std;" here, to avoid collisions between the beta variable and std::beta in c++17
using std::cout;
using std::endl;
// using namespace cv;

namespace
{
/** Global Variables */
int alpha = 100;
int beta = 100;
int gamma_cor = 50;
std::string save_path = "./roi.jpg";
cv::Mat img_original, img_corrected, img_gamma_corrected, for_save;

void basicLinearTransform(const cv::Mat &img, const double alpha_, const int beta_)
{
    cv::Mat res;
    img.convertTo(res, -1, alpha_, beta_);

    hconcat(img, res, img_corrected);
    imshow("Brightness and contrast adjustments", img_corrected);
}

void gammaCorrection(const cv::Mat &img, const double gamma_)
{
    CV_Assert(gamma_ >= 0);
    //! [changing-contrast-brightness-gamma-correction]
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);    //saturate_cast 溢出保护，[0,255]

    // // read image with ptr
    // for (size_t nrow=0;nrow<lookUpTable.rows;nrow++){
    //     uchar* data = lookUpTable.ptr<uchar>(nrow);
    //     for(size_t ncol=0;ncol<lookUpTable.cols;ncol++){
    //         data[ncol] = cv::saturate_cast<uchar>(pow(ncol / 255.0, gamma_) * 255.0);
    //         cout << int(data[ncol]) << std::endl;;
    //     }
    // }
    // read image by row col
    int n = 0;
    for(size_t row=0;row<lookUpTable.rows;row++){
        for(int col=0;col<lookUpTable.cols;col++){
            int b = lookUpTable.at<cv::Vec3b>(row,col)[0];
            int g = lookUpTable.at<cv::Vec3b>(row,col)[1];
            int r = lookUpTable.at<cv::Vec3b>(row,col)[2];
            n++;
            std::cout << "b g r n: " << b << g << r << " " << n << std::endl;
        }
    }
    cv::Mat res = img.clone();
    cv::LUT(img, lookUpTable, res); //LUT 查找表，void LUT(InputArray src, InputArray lut, OutputArray dst)
    //! [changing-contrast-brightness-gamma-correction]

    hconcat(img, res, img_gamma_corrected);
    imshow("Gamma correction", img_gamma_corrected);
}

void on_linear_transform_alpha_trackbar(int, void *)
{
    double alpha_value = alpha / 100.0;
    int beta_value = beta - 100;
    basicLinearTransform(img_original, alpha_value, beta_value);
}

void on_linear_transform_beta_trackbar(int, void *)
{
    double alpha_value = alpha / 100.0;
    int beta_value = beta - 100;
    basicLinearTransform(img_original, alpha_value, beta_value);
}

void on_gamma_correction_trackbar(int, void *)
{
    double gamma_value = gamma_cor / 100.0;
    gammaCorrection(img_original, gamma_value);
}

// return the brightness value of image
double get_avg_gray(IplImage *img)
{
    CvSize size = cvSize(img->width, (img->height)/2);
    cvSetImageROI(img,cvRect(0, (img->height)/2, size.width, (img->height)/2));//设置源图像ROI
    IplImage* pDest = cvCreateImage(size,img->depth,img->nChannels);//创建目标图像
    cvCopy(img, pDest); //复制图像
    cvResetImageROI(pDest);//源图像用完后，清空ROI
    for_save = cv::cvarrToMat(pDest);
    // imshow("test", for_save);
    // waitKey(0);
    cv::imwrite(save_path.c_str(),for_save);//保存目标图像
    IplImage *gray = cvCreateImage(cvGetSize(pDest),IPL_DEPTH_8U,1);
    cvCvtColor(pDest,gray,CV_RGB2GRAY);
    CvScalar scalar = cvAvg(gray);
    cvReleaseImage(&gray);
    return scalar.val[0];
} // end-get-avg-gray
}

int main( int argc, char** argv )
{
    cv::CommandLineParser parser( argc, argv, "{@input | ../data/test1.jpg | input image}" );
    img_original = cv::imread( parser.get<cv::String>( "@input" ) );
    std::cout << "ori_img_channels: " << img_original.channels();
    if( img_original.empty() )
    {
      cout << "Could not open or find the image!\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    }

    IplImage img = IplImage(img_original);
    std::cout<<"Average brightness: " << get_avg_gray(&img)<<std::endl;
    img_corrected = cv::Mat(img_original.rows, img_original.cols*2, img_original.type());
    img_gamma_corrected = cv::Mat(img_original.rows, img_original.cols*2, img_original.type());

    hconcat(img_original, img_original, img_corrected);
    hconcat(img_original, img_original, img_gamma_corrected);

    cv::namedWindow("Brightness and contrast adjustments");
    cv::namedWindow("Gamma correction");

    cv::createTrackbar("Alpha gain (contrast)", "Brightness and contrast adjustments", &alpha, 500, on_linear_transform_alpha_trackbar);
    cv::createTrackbar("Beta bias (brightness)", "Brightness and contrast adjustments", &beta, 200, on_linear_transform_beta_trackbar);
    cv::createTrackbar("Gamma correction", "Gamma correction", &gamma_cor, 200, on_gamma_correction_trackbar);

    on_linear_transform_alpha_trackbar(0, 0);
    on_gamma_correction_trackbar(0, 0);

    cv::waitKey();

    cv::imwrite("linear_transform_correction.png", img_corrected);
    cv::imwrite("gamma_correction.png", img_gamma_corrected);

    return 0;
}
