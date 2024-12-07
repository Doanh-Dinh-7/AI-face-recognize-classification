import cv2
import os
import time
import subprocess

class FaceRecognition:
    def __init__(self, save_path, cam_index=0):
        self.save_path = save_path
        self.cam_index = cam_index
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def capture_face_images(self, face_id):
        """
        Chụp ảnh khuôn mặt theo ID và lưu vào thư mục được chỉ định.
        """
        # Bật webcam
        cam = cv2.VideoCapture(self.cam_index)
        cam.set(3, 640)  # Đặt chiều rộng video
        cam.set(4, 480)  # Đặt chiều cao video

        # Đường dẫn lưu ảnh
        output_dir = os.path.join(self.save_path, face_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("\n[INFO] Đang khởi tạo nhận diện khuôn mặt. Nhìn vào camera...")

        def capture_images(duration, action, start_count):
            count = start_count
            print(f"\n[INFO] Bắt đầu: {action}.")
            start_time = time.time()
            while True:
                ret, img = cam.read()
                if not ret:
                    print("[ERROR] Không thể truy cập webcam.")
                    break
                
                img = cv2.flip(img, 1)  # Lật hình ảnh theo chiều ngang
                faces = self.face_detector.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Vẽ hình chữ nhật quanh khuôn mặt
                    count += 1
                    file_name = os.path.join(output_dir, f"User_{face_id}_{count}.jpg")
                    cv2.imwrite(file_name, img[y:y + h, x:x + w])  # Lưu ảnh màu khuôn mặt

                cv2.imshow('image', img)
                
                # Thoát nếu hết thời gian hoặc nhấn ESC
                if time.time() - start_time >= duration or (cv2.waitKey(1) & 0xff) == 27:
                    break

            print(f"[INFO] Hoàn thành: {action}.")
            return count

        # Chụp ảnh theo yêu cầu
        count = 0
        count = capture_images(10, "Nhìn thẳng", count)
        count = capture_images(10, "Gật đầu liên tục", count)
        count = capture_images(10, "Lắc đầu liên tục", count)

        # Dọn dẹp
        print("\n[INFO] Lưu dữ liệu thành công.")
        cam.release()
        cv2.destroyAllWindows()

        return output_dir

    def process_images(self, raw_path, process_path, model_path, classifier_output):
        """
        Xử lý và huấn luyện dữ liệu khuôn mặt.
        """
        try:
            # Chạy lệnh align_dataset_mtcnn.py
            print("\n[INFO] Chạy lệnh align_dataset_mtcnn.py...")
            align_command = [
                r"D:\Python\AI-recognize-traffic-signs-classification\venv\Scripts\python", #Thay đổi đường dẫn file môi trường
                "src/align_dataset_mtcnn.py", 
                raw_path, 
                process_path, 
                "--image_size", "160", 
                "--margin", "32", 
                "--random_order", 
                "--gpu_memory_fraction", "0.25"
            ]
            print(f"[DEBUG] Lệnh align_dataset_mtcnn: {' '.join(align_command)}")
            subprocess.run(align_command, check=True)

            # Chạy lệnh classifier.py
            print("\n[INFO] Chạy lệnh classifier.py...")
            classifier_command = [
                r"D:\Python\AI-recognize-traffic-signs-classification\venv\Scripts\python", #Thay đổi đường dẫn file môi trường
                "src/classifier.py", 
                "TRAIN", 
                process_path, 
                model_path, 
                classifier_output, 
                "--batch_size", "1000"
            ]
            print(f"[DEBUG] Lệnh classifier.py: {' '.join(classifier_command)}")
            subprocess.run(classifier_command, check=True)

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Lỗi khi chạy lệnh: {e}")
        except FileNotFoundError as e:
            print(f"[ERROR] Không tìm thấy file hoặc đường dẫn: {e}")

    def execute(self, face_id):
        """
        Thực thi toàn bộ quy trình chụp ảnh và xử lý.
        """
        print("[INFO] Bắt đầu quá trình nhận diện khuôn mặt.")
        raw_path = self.capture_face_images(face_id)
        raw_path = "Dataset/FaceData/raw"
        process_path = "Dataset/FaceData/processed"
        model_path = "Models/20180402-114759.pb"
        classifier_output = "Models/facemodel.pkl"
        print("[INFO] Bắt đầu quá trình xử lý khuôn mặt.")
        self.process_images(raw_path, process_path, model_path, classifier_output)
        return process_path
    
# Sử dụng class
if __name__ == "__main__":
    # Cấu hình các tham số đầu vào cho quá trình nhận diện khuôn mặt
    save_path = r"D:\Python\AI-recognize-traffic-signs-classification\Dataset\FaceData\raw" #Thay đổi đường dẫn file lưu RAW
    raw_path = "Dataset/FaceData/raw"
    process_path = "Dataset/FaceData/processed"
    model_path = "Models/20180402-114759.pb"
    classifier_output = "Models/facemodel.pkl"
    face_recog = FaceRecognition(save_path)

    face_id = "221121514117"  # Mã sinh viên (face_id)
    final_path = face_recog.execute(face_id)
    print(f"[INFO] Dữ liệu xử lý đã lưu tại: {final_path}")


# python src/face_rec_cam.py Để bật cam test


# python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25
# python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000



# students/face_recog/



