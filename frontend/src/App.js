import React, { useRef, useEffect, useState } from "react";

const App = () => {
  const videoRef = useRef(null);
  const [gesture, setGesture] = useState("");
  const [capturedGesture, setCapturedGesture] = useState([]);

  useEffect(() => {
    // Access the webcam stream
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((error) => console.error("Error accessing webcam:", error));

    // Add event listener for keyboard input
    const handleKeyPress = (event) => {
      if (event.key === "s") {
        captureFrame();
      } else if (event.key === "o") {
        sendToLLM();
      }
      // a training model part needs to be implemented
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, []);

  // Function to capture frame from webcam
  const captureFrame = () => {
    console.log("Capturing Frame...");
    const video = videoRef.current;
    if (!video) return;
  
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
    canvas.toBlob(async (blob) => {
      if (!blob) {
        console.error("Failed to capture frame");
        return;
      }
  
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");
  
      try {
        const response = await fetch("http://127.0.0.1:8000/predict/", {
          method: "POST",
          body: formData,
        });
  
        const data = await response.json();
        console.log("Captured Gesture:", data.gesture); // Debugging
  
        if (data.gesture) {
          setCapturedGesture((prevGestures) => {
            const newGestures = [...prevGestures, data.gesture];
            console.log("Updated Captured Gestures:", newGestures); // Debugging
            return newGestures;
          });
        } else {
          console.error("No valid gesture detected.");
        }
      } catch (error) {
        console.error("Error:", error);
      }
    }, "image/jpeg");
  };
  
  

  // Function to send collected gestures to LLM
  const sendToLLM = async () => {
  console.log("Captured Gestures before sending:", capturedGesture); // Debugging

  if (!capturedGesture || capturedGesture.length === 0) {
    console.error("Error: No gestures captured.");
    return;
  }

  try {
    const response = await fetch("http://127.0.0.1:8000/generate_sentence/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gestures: capturedGesture }),
    });

    const data = await response.json();
    console.log("Response from API:", data); // Debugging

    if (data.sentence) {
      setGesture(data.sentence);
    } else {
      setGesture("Failed to generate sentence");
    }
  } catch (error) {
    console.error("Error:", error);
  }
};


  return (
    <div>
      <h1>Sign Language Recognition</h1>
      <video ref={videoRef} autoPlay playsInline style={{ width: "640px", height: "480px", border: "1px solid black" }} />
      <p>Press "S" to capture gesture, "O" to generate sentence.</p>
      <p>Recognized Sentence: {gesture}</p>
      <div>
        <h2>Captured Gestures:</h2>
        <ul>
          {capturedGesture.map((gesture, index) => (
            <li key={index}>{gesture}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default App;
