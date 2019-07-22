// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/tracking.hpp>

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }";

using namespace cv;
using namespace dnn;
using namespace std;

namespace {

// Initialize the parameters
const float confThreshold = 0.5f; // Confidence threshold
const float nmsThreshold = 0.4f;  // Non-maximum suppression threshold
const int inpWidth = 416;  // Width of network's input image
const int inpHeight = 416; // Height of network's input image

enum { numTrackingFrames = 10 };

vector<string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
auto postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (const auto & out : outs)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        auto* data = reinterpret_cast<float*>(out.data);
        for (int j = 0; j < out.rows; ++j, data += out.cols)
        {
            Mat scores = out.row(j).colRange(5, out.cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = static_cast<int>(data[0] * frame.cols);
                int centerY = static_cast<int>(data[1] * frame.rows);
                int width = static_cast<int>(data[2] * frame.cols);
                int height = static_cast<int>(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back(static_cast<float>(confidence));
                boxes.emplace_back(left, top, width, height);
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    vector<int> outClassIds;
    vector<float> outConfidences;
    vector<Rect> outBoxes;

    outBoxes.resize(indices.size());
    outClassIds.resize(indices.size());
    outConfidences.resize(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        outBoxes[i] = boxes[idx];
        outClassIds[i] = classIds[idx];
        outConfidences[i] = confidences[idx];
    }

    return std::make_tuple(outClassIds, outConfidences, outBoxes);
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, 
        Point(left, top - round(1.5*labelSize.height)), 
        Point(left + round(1.5*labelSize.width), top + baseLine), 
        Scalar(255, 255, 255), 
        FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

void drawPreds(Mat& frame, vector<int>& classIds,
    vector<float>& confidences,
    vector<Rect>& boxes)
{
    //int idx = indices[i];
    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);
    }
}


// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i) {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}

void putInformation(Net &net, cv::Mat &frame, bool tracking)
{
    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("%s Inference time for a frame : %.2f ms", (tracking ? "Tracking" : "Looking"), t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
}

} // namespace


int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // Load names of classes
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }

    // Give the configuration and weight files for the model
    const String modelConfiguration = "yolov3.cfg";
    const String modelWeights = "yolov3.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    VideoWriter video;
    Mat frame;
    Mat blob;

    try {

        string outputFile = "yolo_out_cpp.avi";
        if (parser.has("image"))
        {
            // Open the image file
            string str = parser.get<String>("image");
            ifstream ifile(str);
            if (!ifile) {
                throw std::runtime_error("File opening error");
            }
            cap.open(str);
            str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.jpg");
            outputFile = str;
        }
        else if (parser.has("video"))
        {
            // Open the video file
            string str = parser.get<String>("video");
            ifstream ifile(str);
            if (!ifile) {
                throw std::runtime_error("File opening error");
            }
            cap.open(str);
            str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.avi");
            outputFile = str;
        }
        // Open the webcaom
        else {
            cap.open(parser.get<int>("device"));
        }


        // Get the video writer initialized to save the output video
        if (!parser.has("image")) {
            video.open(outputFile,
                VideoWriter::fourcc('M', 'J', 'P', 'G'),
                28,
                Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
        }

        // Create a window
        //static const string kWinName = "Deep learning object detection in OpenCV";
        //namedWindow(kWinName, WINDOW_NORMAL);

        // Process frames.

        Ptr<MultiTracker> multiTracker;

        int trackingFrame = 0;

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;


        while (waitKey(/*1*/) < 0)
        {
            // get frame from the video
            cap >> frame;

            // Stop the program if reached end of video
            if (frame.empty()) {
                cout << "Done processing !!!\nOutput file is stored as " << outputFile << '\n';
                waitKey(3000);
                break;
            }

            const bool tracking = trackingFrame > 0 && multiTracker->update(frame);

            if (tracking)
            {
                // draw tracked objects
                for (unsigned i = 0; i < multiTracker->getObjects().size(); i++)
                {
                    boxes[i] = multiTracker->getObjects()[i];
                }

                --trackingFrame;
            }
            else
            {
                trackingFrame = numTrackingFrames;

                // Create a 4D blob from a frame.
                blobFromImage(frame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

                //Sets the input to the network
                net.setInput(blob);

                // Runs the forward pass to get output of the output layers
                vector<Mat> outs;
                net.forward(outs, getOutputsNames(net));

                // Remove the bounding boxes with low confidence
                std::tie(classIds, confidences, boxes) = postprocess(frame, outs);

                multiTracker = cv::MultiTracker::create();
                for (auto & boxe : boxes)
                {
                    multiTracker->add(
                        //TrackerCSRT::create(params), 
                        TrackerCSRT::create(),
                        frame, Rect2d(boxe));
                }
            }

            drawPreds(frame, classIds, confidences, boxes);

            putInformation(net, frame, tracking);

            // Write the frame with the detection boxes
            Mat detectedFrame;
            frame.convertTo(detectedFrame, CV_8U);
            if (parser.has("image")) {
                imwrite(outputFile, detectedFrame);
            }
            else {
                video.write(detectedFrame);
            }


            //imshow(kWinName, frame);

        }

        cap.release();
        if (!parser.has("image")) {
            video.release();
        }

    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}

