package main

import (
	"bytes"
	"fmt"
	"image/color"
	"net/http"
	"time"

	"gocv.io/x/gocv"
)

func main() {
	// Open the video capture device
	// deviceID := 0
	// video, err := gocv.VideoCaptureDevice(deviceID)
	rtspUrl := "rtsp://cam1:berry639@192.168.1.61:554/live/ch0"
	video, err := gocv.OpenVideoCapture(rtspUrl)

	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", err)
		return
	}
	defer video.Close()

	// Load the face detection classifier
	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()
	if !classifier.Load("haarcascade_frontalface_default.xml") {
		fmt.Printf("Error loading face detection classifier file\n")
		return
	}

	// Create a window to display the video stream
	window := gocv.NewWindow("Face Detection")
	defer window.Close()

	// Initialize the Kerberos.io API endpoint
	url := "http://localhost:80/api/1/event"
	client := &http.Client{}

	// Loop over the video frames and perform face detection
	img := gocv.NewMat()
	defer img.Close()
	for {
		if ok := video.Read(&img); !ok {
			fmt.Printf("Error reading video frame\n")
			break
		}
		if img.Empty() {
			continue
		}

		// Convert the image to grayscale and equalize the histogram
		gray := gocv.NewMat()
		defer gray.Close()
		gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)
		gocv.EqualizeHist(gray, &gray)

		// Detect faces in the image
		rects := classifier.DetectMultiScale(gray)
		for _, r := range rects {
			// Draw a rectangle around each detected face
			gocv.Rectangle(&img, r, color.RGBA{0, 255, 0, 0}, 2)

			// Assuming buf is a *gocv.NativeByteBuffer
			imgBytes, err := gocv.IMEncode(".jpg", img)
			if err != nil {
				fmt.Printf("Error encoding image: %v\n", err)
				continue
			}

			// Get the byte slice of the encoded image
			buf := imgBytes.GetBytes()

			// Create a bytes.Reader for sending to Kerberos.io
			data := bytes.NewReader(buf)

			// Send the image to Kerberos.io
			req, err := http.NewRequest("POST", url, data)
			if err != nil {
				fmt.Printf("Error creating HTTP request: %v\n", err)
				continue
			}
			req.Header.Set("Content-Type", "image/jpeg")
			resp, err := client.Do(req)
			if err != nil {
				fmt.Printf("Error sending HTTP request: %v\n", err)
				continue
			}
			defer resp.Body.Close()

			// Wait for a short time before sending the next frame
			time.Sleep(100 * time.Millisecond)
		}

		// Display the video stream with face detection rectangles
		window.IMShow(img)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
