#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "util.hpp"

#include <iostream>
#include <cctype>

using namespace cv;
using namespace std;

int main(){
    /****************** EPIPOLAR GEOMETRY **************************/
    printf("OpenCV: %s", getBuildInformation().c_str());

    // FIXME:
    Mat img_1 = imread("/Users/wang3450/Downloads/left.jpeg");
    Mat img_2 = imread("/Users/wang3450/Downloads/right.jpeg");

    // Mat img_1 = imread("img/perra_7.jpg");
    // Mat img_2 = imread("img/perra_8.jpg");

    // Mat img_1 = imread("img/madera_1.jpg");
    // Mat img_2 = imread("img/madera_2.jpg");

    Mat fund_mat = Mat::zeros(3,3,CV_64F);

    Vec3d epipole;

    vector<Vec3d> lines_1, lines_2;
    vector<Point2d> good_matches_1, good_matches_2;

    // Get epipolar geometry
    computeEpiLines(img_1, img_2,
                    epipole, fund_mat,
                    lines_1, lines_2,
                    good_matches_1, good_matches_2);

    Mat A, B, Ap, Bp;

    Mat e_x = crossProductMatrix(epipole);

    /****************** PROJECTIVE **************************/

    // Get A,B matrix for minimizing z
    obtainAB(img_1, e_x, A, B);
    obtainAB(img_2, fund_mat, Ap, Bp);

    // Get initial guess for z
    Vec3d z = getInitialGuess(A, B, Ap, Bp);

    // Optimizes the z solution
    optimizeRoot(A, B, Ap, Bp, z);

    // Get w
    Mat w = e_x * Mat(z);
    Mat wp = fund_mat * Mat(z);

    w /= w.at<double>(2,0);
    wp /= wp.at<double>(2,0);

    // Get final H_p and Hp_p matrix for projection
    Mat H_p = Mat::eye(3, 3, CV_64F);
    H_p.at<double>(2,0) = w.at<double>(0,0);
    H_p.at<double>(2,1) = w.at<double>(1,0);

    Mat Hp_p = Mat::eye(3, 3, CV_64F);
    Hp_p.at<double>(2,0) = wp.at<double>(0,0);
    Hp_p.at<double>(2,1) = wp.at<double>(1,0);

    /****************** SIMILARITY **************************/

    // Get the translation term
    double vp_c = getTranslationTerm(img_1, img_2, H_p, Hp_p);

    // Get the H_r and Hp_r matrix directly
    Mat H_r = Mat::zeros(3, 3, CV_64F);

    H_r.at<double>(0,0) = fund_mat.at<double>(2,1) - w.at<double>(1,0) * fund_mat.at<double>(2,2);
    H_r.at<double>(1,0) = fund_mat.at<double>(2,0) - w.at<double>(0,0) * fund_mat.at<double>(2,2);

    H_r.at<double>(0,1) = w.at<double>(0,0) * fund_mat.at<double>(2,2) - fund_mat.at<double>(2,0);
    H_r.at<double>(1,1) = H_r.at<double>(0,0);

    H_r.at<double>(1,2) = fund_mat.at<double>(2,2) + vp_c;
    H_r.at<double>(2,2) = 1.0;

    Mat Hp_r = Mat::zeros(3, 3, CV_64F);

    Hp_r.at<double>(0,0) = wp.at<double>(1,0) * fund_mat.at<double>(2,2) - fund_mat.at<double>(1,2);
    Hp_r.at<double>(1,0) = wp.at<double>(0,0) * fund_mat.at<double>(2,2) - fund_mat.at<double>(0,2);

    Hp_r.at<double>(0,1) = fund_mat.at<double>(0,2) - wp.at<double>(0,0) * fund_mat.at<double>(2,2);
    Hp_r.at<double>(1,1) = Hp_r.at<double>(0,0);

    Hp_r.at<double>(1,2) = vp_c;
    Hp_r.at<double>(2,2) = 1.0;

    /******************* SHEARING ***************************/

    Mat H_1 = H_r*H_p;
    Mat H_2 = Hp_r*Hp_p;

    Mat H_s, Hp_s;

    // Get shearing transforms with the method described on the paper
    getShearingTransforms(img_1, img_2, H_1, H_2, H_s, Hp_s);

    /****************** RECTIFY IMAGES **********************/

    Mat H = H_s * H_r * H_p;
    Mat Hp = Hp_s * Hp_r * Hp_p;


            // Get homography image of the corner coordinates from all the images
            vector<Point2d> corners_all(4), corners_all_t(4);
            double min_x, min_y, max_x, max_y;
            min_x = min_y = +INF;
            max_x = max_y = -INF;

            corners_all[0] = Point2d(0,0);
            corners_all[1] = Point2d(img_1.cols,0);
            corners_all[2] = Point2d(img_1.cols,img_1.rows);
            corners_all[3] = Point2d(0,img_1.rows);

            perspectiveTransform(corners_all, corners_all_t, H);

            for (int j = 0; j < 4; j++) {
                min_x = min(corners_all_t[j].x, min_x);
                max_x = max(corners_all_t[j].x, max_x);

                min_y = min(corners_all_t[j].y, min_y);
                max_y = max(corners_all_t[j].y, max_y);
            }

            int img_1_cols = max_x - min_x;
            int img_1_rows = max_y - min_y;

            // Get homography image of the corner coordinates from all the images
            min_x = min_y = +INF;
            max_x = max_y = -INF;

            corners_all[0] = Point2d(0,0);
            corners_all[1] = Point2d(img_2.cols,0);
            corners_all[2] = Point2d(img_2.cols,img_2.rows);
            corners_all[3] = Point2d(0,img_2.rows);

            perspectiveTransform(corners_all, corners_all_t, Hp);

            for (int j = 0; j < 4; j++) {
                min_x = min(corners_all_t[j].x, min_x);
                max_x = max(corners_all_t[j].x, max_x);

                min_y = min(corners_all_t[j].y, min_y);
                max_y = max(corners_all_t[j].y, max_y);
            }

            int img_2_cols = max_x - min_x;
            int img_2_rows = max_y - min_y;

    // Apply homographies
    Mat img_1_dst(img_1_rows, img_1_cols, CV_64F);
    Mat img_2_dst(img_2_rows, img_2_cols, CV_64F);

    warpPerspective( img_1, img_1_dst, H, img_1_dst.size() );
    warpPerspective( img_2, img_2_dst, Hp, img_2_dst.size() );

    Vec3d epipole_dst;

    vector<Vec3d> lines_1_dst, lines_2_dst;
    vector<Point2d> good_matches_1_dst, good_matches_2_dst;

    perspectiveTransform(good_matches_1, good_matches_1_dst, H);
    perspectiveTransform(good_matches_2, good_matches_2_dst, Hp);

    // Get epipolar geometry and draw epilines
    computeEpiLines(img_1_dst, img_2_dst, epipole_dst, fund_mat, lines_1_dst, lines_2_dst, good_matches_1_dst, good_matches_2_dst);

    drawEpilines(img_1, img_2, lines_1, lines_2, good_matches_1, good_matches_2, 150);
    drawEpilines(img_1_dst, img_2_dst, lines_1_dst, lines_2_dst, good_matches_1_dst, good_matches_2_dst, 150);

    cout << "\nH = " << H << "\nHp = " << Hp << endl;

    cout << "\nEpipolo antes: " << epipole/epipole[2] << "\nEpipolo después: " << epipole_dst << endl;

    draw(img_1, "1");
    draw(img_1_dst, "1 rectificada");

    char c = 'a';

    draw(img_2, "2");
    draw(img_2_dst, "2 rectificada");

    c = 'a';

    while (tolower(c) != 'q')
      c = waitKey();
    
    destroyAllWindows();

}
