import cv2

def main():
    check_video("/home/mihai/datasets/EPFL/Laboratory3/Video/6p-c3.avi")



def check_video(video_file):

    inference_times = []
    # video_path = os.path.join(configuration.home_path, configuration.data_path + video_file +'.mp4')
    cap = cv2.VideoCapture(video_file)

    # interpreter, inference_size = inferencecoralutils.initialize_interpreter(os.path.abspath(configuration.model_path))
    # ground_truth = groundtruthutils.GroundTruthTxt(annotation)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


