import React, { useState, useRef } from "react";

const GestureRecognition = () => {
  const [gesture, setGesture] = useState("");
  const videoRef = useRef(null);

  const captureFrame = async () => {
    const video = videoRef.current;
    if (!video) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(async (blob) => {
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      try {
        const response = await fetch("http://127.0.0.1:8000/predict/", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        if (data.gesture) {
          setGesture(data.gesture);
        } else {
          setGesture("No hand detected");
        }
      } catch (error) {
        console.error("Error:", error);
        setGesture("Error processing request");
      }
    }, "image/jpeg");
  };

  return (
    <div>
      <h1>Real-Time Gesture Recognition</h1>
      <video ref={videoRef} autoPlay width="640" height="480"></video>
      <button onClick={captureFrame}>Capture Gesture</button>
      <h2>Detected Gesture: {gesture}</h2>
    </div>
  );
};

export default GestureRecognition;
