package com.iem.fyp.shape.application;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.videoio.VideoCapture;

import com.iem.fyp.shape.utils.Utils;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

/**
 * The controller associated with the only view of our application. The
 * application logic is implemented here. It handles the button for
 * starting/stopping the camera, the acquired video stream, the relative
 * controls and the image segmentation process.
 *
 * @author Sayantan Dey
 * @author Arijit Basu
 * @version 1.0 (15.5.2018)
 *
 */
public class ObjRecognitionController
{
    // FXML camera button
    @FXML
    private Button cameraButton;
    // the FXML area for showing the current frame
    @FXML
    private ImageView originalFrame;
    // the FXML area for showing the mask
    @FXML
    private ImageView maskImage;
    // the FXML area for showing the output of the morphological operations
    @FXML
    private ImageView morphImage;
    // FXML slider for setting HSV ranges
    @FXML
    private Slider hueStart;
    @FXML
    private Slider hueStop;
    @FXML
    private Slider saturationStart;
    @FXML
    private Slider saturationStop;
    @FXML
    private Slider valueStart;
    @FXML
    private Slider valueStop;

    @FXML
    private Slider epsilon;
    // FXML label to show the current values set with the sliders
    @FXML
    private Label hsvCurrentValues;

    // a timer for acquiring the video stream
    private ScheduledExecutorService timer;
    // the OpenCV object that performs the video capture
    private VideoCapture capture = new VideoCapture();
    // a flag to change the button behavior
    private boolean cameraActive;

    // property for object binding
    private ObjectProperty<String> hsvValuesProp;

    /**
     * The frame action triggered by clicking the button on the GUI
     */
    @FXML
    private void startCamera()
    {
        // bind a text property with the string containing the current range of
        // HSV values for object detection
        hsvValuesProp = new SimpleObjectProperty<>();
        this.hsvCurrentValues.textProperty().bind(hsvValuesProp);

        // set a fixed width for all the image to show and preserve image ratio
        this.imageViewProperties(this.originalFrame, 400);
        this.imageViewProperties(this.maskImage, 200);
        this.imageViewProperties(this.morphImage, 200);

        if (!this.cameraActive)
        {
            // start the video capture
            this.capture.open(0);

            // is the video stream available?
            if (this.capture.isOpened())
            {
                this.cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = new Runnable() {

                    @Override
                    public void run()
                    {
                        // effectively grab and process a single frame
                        Mat frame = grabFrame();
                        // convert and show the frame
                        Image imageToShow = Utils.mat2Image(frame);
                        updateImageView(originalFrame, imageToShow);
                    }
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

                // update the button content
                this.cameraButton.setText("Stop Camera");
            }
            else
            {
                // log the error
                System.err.println("Failed to open the camera connection...");
            }
        }
        else
        {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.cameraButton.setText("Start Camera");

            // stop the timer
            this.stopAcquisition();
        }
    }

    /**
     * Get a frame from the opened video stream (if any)
     *
     * @return the {@link Image} to show
     */
    private Mat grabFrame()
    {
        Mat frame = new Mat();

        // check if the capture is open
        if (this.capture.isOpened())
        {
            try
            {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty())
                {
                    // init
                    Mat blurredImage = new Mat();
                    Mat hsvImage = new Mat();
                    Mat mask = new Mat();
                    Mat morphOutput = new Mat();

                    // remove some noise
                    Imgproc.blur(frame, blurredImage, new Size(7, 7));

                    // convert the frame to HSV
                    Imgproc.cvtColor(blurredImage, hsvImage, Imgproc.COLOR_BGR2HSV);

                    // get threshold values from the UI
                    Scalar epsilonValue = new Scalar(this.epsilon.getValue());
                    // remember: H ranges 0-180, S and V range 0-255
                    Scalar minValues = new Scalar(this.hueStart.getValue(), this.saturationStart.getValue(),
                            this.valueStart.getValue());
                    Scalar maxValues = new Scalar(this.hueStop.getValue(), this.saturationStop.getValue(),
                            this.valueStop.getValue());

                    // show the current selected HSV range
                    String valuesToPrint = "Hue range: " + minValues.val[0] + "-" + maxValues.val[0]
                            + "\tSaturation range: " + minValues.val[1] + "-" + maxValues.val[1] + "\tValue range: "
                            + minValues.val[2] + "-" + maxValues.val[2]+"\nEpsilon value: " + epsilonValue.val[0];
                    Utils.onFXThread(this.hsvValuesProp, valuesToPrint);

                    // threshold HSV image to select tennis balls
                    Core.inRange(hsvImage, minValues, maxValues, mask);
                    // show the partial output
                    this.updateImageView(this.maskImage, Utils.mat2Image(mask));

                    // morphological operators
                    // dilate with large element, erode with small ones
                    Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(24, 24));
                    Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));

                    Imgproc.erode(mask, morphOutput, erodeElement);
                    Imgproc.erode(morphOutput, morphOutput, erodeElement);

                    Imgproc.dilate(morphOutput, morphOutput, dilateElement);
                    Imgproc.dilate(morphOutput, morphOutput, dilateElement);

                    // show the partial output
                    this.updateImageView(this.morphImage, Utils.mat2Image(morphOutput));

                    // find the tennis ball(s) contours and show them
                    frame = this.drawContour(morphOutput, frame, epsilonValue);

                }

            }
            catch (Exception e)
            {
                // log the (full) error
                System.err.print("Exception during the image elaboration...");
                e.printStackTrace();
            }
        }

        return frame;
    }

    /**
     * Given a binary image containing one or more closed surfaces, use it as a
     * mask to find and highlight the objects contours
     *
     * @param maskedImage
     *            the binary image to be used as a mask
     * @param frame
     *            the original {@link Mat} image to be used for drawing the
     *            objects contours
     * @param epsilonValue
     * @return the {@link Mat} image with the objects contours framed
     */
    private Mat drawContour(Mat maskedImage, Mat frame, Scalar epsilonValue)
    {
        // init
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        boolean isCircle = false;

        // find contours of the shape
        Imgproc.findContours(maskedImage, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

        //first check if the shape is circle or not by Hough circle detection
        Mat circles = new Mat();

        //blur the maskedImage to get better result and then find the circles
        Imgproc.blur(maskedImage, maskedImage, new Size(7, 7), new Point(2, 2));
        Imgproc.HoughCircles(maskedImage, circles, Imgproc.CV_HOUGH_GRADIENT, 2, 100, 100, 90, 0, 1000);

        if (circles.cols() > 0) {
            for (int x = 0; x < Math.min(circles.cols(), 5); x++) {
                double circleVec[] = circles.get(0, x);

                if (circleVec == null) {
                    break;
                }

                Point center = new Point((int) circleVec[0], (int) circleVec[1]);
                int radius = (int) circleVec[2];

                System.out.print(" center(x): "+center.x+" center(y): "+center.y+" radius: " + radius+",");
                if(radius > 0 ) isCircle = true;

                System.out.println(" Shape: Circle"+",");
                Imgproc.putText(frame, "Circle", new Point(center.x, center.y), Core.FONT_HERSHEY_SIMPLEX ,
                        1, new Scalar(255, 255, 255),4);
//                this.updateImageView(this.originalFrame, Utils.mat2Image(frame));
            }
        }

        //if shape is not circle then process image to find shape
        if (!isCircle) {
            //loop through each contour and find the object present in a single frame
            Iterator<MatOfPoint> iterator = contours.iterator();
            while (iterator.hasNext()){
                MatOfPoint contour = iterator.next();

                //pass the contour to find the center of it
                Point center = findCenter(contour);
                System.out.print(" center(x): "+center.x+" center(y): "+center.y+",");

                try {
                    double epsilon = epsilonValue.val[0]*Imgproc.arcLength(new MatOfPoint2f(contour.toArray()),true);
                    System.out.print(" epsilon: "+epsilonValue.val[0]+",");

                    MatOfPoint2f approx = new MatOfPoint2f();
                    MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());
                    Imgproc.approxPolyDP(curve,approx,epsilon,true);

                    MatOfPoint point = new MatOfPoint(approx.toArray());
                    long ptCount = point.total();
                    System.out.print(" ptCount: "+ptCount+",");

                    if(ptCount == 3) {
                        System.out.println(" Shape: Triangle"+",");
                        Imgproc.putText(frame, "Triangle", new Point(center.x, center.y), Core.FONT_HERSHEY_SIMPLEX ,
                                1, new Scalar(255,255,255),4);
//                    this.updateImageView(this.originalFrame, Utils.mat2Image(frame));
                    }
                    if(ptCount == 4) {
                        System.out.println(" Shape: Rect"+",");
                        Imgproc.putText(frame, "Rect", new Point(center.x, center.y), Core.FONT_HERSHEY_SIMPLEX ,
                                1, new Scalar(255,255,255),4);
//                    this.updateImageView(this.originalFrame, Utils.mat2Image(frame));
                    }
                    if(ptCount == 5) {
                        System.out.println(" Shape: Penta"+",");
                        Imgproc.putText(frame, "Penta", new Point(center.x, center.y), Core.FONT_HERSHEY_SIMPLEX ,
                                1, new Scalar(255,255,255),4);
//                    this.updateImageView(this.originalFrame, Utils.mat2Image(frame));
                    }
                    if(ptCount == 6) {
                        System.out.println(" Shape: Hexa"+",");
                        Imgproc.putText(frame, "Hexa", new Point(center.x, center.y), Core.FONT_HERSHEY_SIMPLEX ,
                                1, new Scalar(255,255,255),4);
//                    this.updateImageView(this.originalFrame, Utils.mat2Image(frame));
                    }
                    if(ptCount == 7) {
                        System.out.println(" Shape: Septa"+",");
                        Imgproc.putText(frame, "Septa", new Point(center.x, center.y), Core.FONT_HERSHEY_SIMPLEX ,
                                1, new Scalar(255,255,255),4);
//                    this.updateImageView(this.originalFrame, Utils.mat2Image(frame));
                    }
                    if(ptCount == 8) {
                        System.out.println(" Shape: Octa"+",");
                        Imgproc.putText(frame, "Octa", new Point(center.x, center.y), Core.FONT_HERSHEY_SIMPLEX ,
                                1, new Scalar(255,255,255),4);
//                    this.updateImageView(this.originalFrame, Utils.mat2Image(frame));
                    }
                    this.updateImageView(this.originalFrame, Utils.mat2Image(frame));
                } catch (Exception e) {
                    System.out.println("no object detected");
                }

            }

        }


        // if any contour exist...
        if (hierarchy.size().height > 0 && hierarchy.size().width > 0)
        {
            // for each contour, display it in blue
            for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0])
            {
                Imgproc.drawContours(frame, contours, idx, new Scalar(250, 0, 0), 1);
            }
        }
        return frame;
    }

    /**
     * Set typical {@link ImageView} properties: a fixed width and the
     * information to preserve the original image ration
     *
     * @param image
     *            the {@link ImageView} to use
     * @param dimension
     *            the width of the image to set
     */
    private void imageViewProperties(ImageView image, int dimension)
    {
        // set a fixed width for the given ImageView
        image.setFitWidth(dimension);
        // preserve the image ratio
        image.setPreserveRatio(true);
    }

    /**
     * Stop the acquisition from the camera and release all the resources
     */
    private void stopAcquisition()
    {
        if (this.timer!=null && !this.timer.isShutdown())
        {
            try
            {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            }
            catch (InterruptedException e)
            {
                // log any exception
                System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
            }
        }

        if (this.capture.isOpened())
        {
            // release the camera
            this.capture.release();
        }
    }

    /**
     * Update the {@link ImageView} in the JavaFX main thread
     *
     * @param view
     *            the {@link ImageView} to update
     * @param image
     *            the {@link Image} to show
     */
    private void updateImageView(ImageView view, Image image)
    {
        Utils.onFXThread(view.imageProperty(), image);
    }

    /**
     * On application close, stop the acquisition from the camera
     */
    protected void setClosed()
    {
        this.stopAcquisition();
    }

    private Point findCenter(MatOfPoint contour){

        Moments moments = Imgproc.moments(contour);

        Point centroid = new Point();

        centroid.x = moments.get_m10() / moments.get_m00();
        centroid.y = moments.get_m01() / moments.get_m00();

        return  centroid;
    }

}
