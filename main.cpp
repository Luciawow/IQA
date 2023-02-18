#include <opencv2/opencv.hpp>

using namespace cv;

struct Box
{
    int xmin, xmax, ymin, ymax;
};
double psnr(Mat& I1, Mat& I2) { //注意，当两幅图像一样时这个函数计算出来的psnr为0 
    Mat s1;
    absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);//转换为32位的float类型，8位不能计算平方  
    s1 = s1.mul(s1);
    Scalar s = sum(s1);  //计算每个通道的和  
    double sse = s.val[0] + s.val[1] + s.val[2];
    if (sse <= 1e-10) // for small values return zero  
        return 0;
    else
    {
        double mse = sse / (double)(I1.channels() * I1.total()); //  sse/(w*h*3)  
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}
double ssim(Mat& i1, Mat& i2) {
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);
    Mat I1_2 = I1.mul(I1);
    Mat I2_2 = I2.mul(I2);
    Mat I1_I2 = I1.mul(I2);
    Mat mu1, mu2;
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);
    Mat sigma1_2, sigam2_2, sigam12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigam2_2, Size(11, 11), 1.5);
    sigam2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigam12, Size(11, 11), 1.5);
    sigam12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigam12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigam2_2 + C2;
    t1 = t1.mul(t2);

    Mat ssim_map;
    divide(t3, t1, ssim_map);
    Scalar mssim = mean(ssim_map);

    double ssim = (mssim.val[0] + mssim.val[1] + mssim.val[2]) / 3;
    return ssim;
}

Box calMaskBox(cv::Mat mask)
{
    int xmin = INT_MAX, ymin = INT_MAX;
    int xmax = 0, ymax = 0;
    for (int h = 0; h < mask.rows; h++) {
        for (int w = 0; w < mask.cols; w++) {
            int mask_pixel = (int)mask.at<uchar>(h, w);
            if (mask_pixel == 255) {
                if (h < ymin)
                    ymin = h;
                if (h > ymax)
                    ymax = h;
                if (w < xmin)
                    xmin = w;
                if (w > xmax)
                    xmax = w;
            }
        }
    }
    Box box = { box.xmin = xmin, box.xmax = xmax, box.ymin = ymin, box.ymax = ymax };
    return box;
}
int main(int argc, char* argv[])
{
    std::string gtName = argv[1];
    std::string maskName = argv[2];
    std::string resName = argv[3];

    cv::Mat gtImg = cv::imread(gtName);
    cv::Mat mask = cv::imread(maskName);
    cv::Mat mask_gray = cv::imread(maskName,IMREAD_GRAYSCALE);
    cv::Mat resImg = cv::imread(resName);

    Box maskBox;
    Mat gtMask, resMask,gtArea, resArea;
    bitwise_and(gtImg,mask, gtMask);
    bitwise_and(resImg,mask, resMask);
    maskBox = calMaskBox(mask_gray);
    gtArea = gtMask(Rect(maskBox.xmin,maskBox.ymin,maskBox.xmax- maskBox.xmin, maskBox.ymax - maskBox.ymin));
    resArea = resMask(Rect(maskBox.xmin,maskBox.ymin,maskBox.xmax- maskBox.xmin, maskBox.ymax - maskBox.ymin));
    
    //imshow("resMask", resMask);
    double ssim_score, psnr_score;
    ssim_score = ssim(gtArea, resArea);
    psnr_score = psnr(gtArea, resArea);
    std::cout << "ssim score:" << ssim_score << "  " << "psnr score:" << psnr_score << std::endl;
    waitKey();
    return 0;
}