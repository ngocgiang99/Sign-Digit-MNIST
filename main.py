import numpy as np 
import cv2
import keras
#pylint: disable=all

def load_model():
    model_path = "Model"
    model_name = "model1"
    json_file = open(model_path + "/" + model_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(model_path + "/" + model_name + ".h5")

    return model

def main():

    model = load_model()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened:
        print("Camera can not open")
        exit(-1)
    
    while True:
        _, frame = cap.read()

        cv2.imshow("origin image", frame)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_frame = gray_frame / 255
        gray_frame = cv2.resize(gray_frame, (64, 64))
        _, binary_frame = cv2.threshold(gray_frame, 0.5, 1, cv2.THRESH_BINARY)
        # /binary_frame = cv2.resize(binary_frame, (64, 64))
        cv2.imshow("predict image", binary_frame)
        binary_frame = binary_frame.reshape(1, 64, 64)
        

        test = np.asarray([binary_frame], dtype = "float32")

        result = model.predict(test)
        #print(result)
        result = np.where(result[0] == np.amax(result[0]))
        print(result[0])

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    

if __name__ == "__main__":
    main()