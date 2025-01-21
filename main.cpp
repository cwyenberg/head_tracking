#include <opencv2/opencv.hpp>

int main() {
    cv::CascadeClassifier face_cascade, eye_cascade;
    face_cascade.load("/opt/homebrew/Cellar/opencv/4.11.0/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    eye_cascade.load("/opt/homebrew/Cellar/opencv/4.11.0/share/opencv4/haarcascades/haarcascade_eye.xml");

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::Mat frame, gray;
    while (true) {
        cap >> frame;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);

            cv::Mat faceROI = gray(face);
            std::vector<cv::Rect> eyes;
            eye_cascade.detectMultiScale(faceROI, eyes);

            for (const auto& eye : eyes) {
                cv::Rect eye_rect(eye.x + face.x, eye.y + face.y, eye.width, eye.height);
                cv::rectangle(frame, eye_rect, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("Eye Detection", frame);
        if (cv::waitKey(10) == 27) break;
    }
    return 0;
}
