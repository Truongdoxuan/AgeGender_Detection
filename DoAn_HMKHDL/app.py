# ui/app.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QLabel, QFileDialog, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from predict.predictor import predict_face


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Age & Gender Detection")
        self.resize(900, 520)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # ===== Buttons =====
        self.btn_img = QPushButton("📷 Chọn Ảnh")
        self.btn_cam = QPushButton("🟢 Webcam")
        self.btn_stop = QPushButton("🔴 Dừng xử lý")

        self.btn_img.clicked.connect(self.open_image)
        self.btn_cam.clicked.connect(self.open_camera)
        self.btn_stop.clicked.connect(self.stop_camera)

        # ===== Labels =====
        self.image_label = QLabel("Vui lòng chọn ảnh hoặc bật Webcam...")
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        # ===== Layout =====
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.btn_img)
        left_layout.addWidget(self.btn_cam)
        left_layout.addWidget(self.btn_stop)
        left_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.image_label)

        layout = QVBoxLayout()
        layout.addLayout(main_layout)


        self.setLayout(layout)

    # ===== Image =====
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Image Files (*.jpg *.png *.jpeg)"
        )
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.process_and_show(img)

    # ===== Webcam =====
    def open_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.process_and_show(frame)

    def stop_camera(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None

    # ===== Prediction + Display =====
    def process_and_show(self, frame):
        gender, age = predict_face(frame)

        # ===== DRAW BOUNDING BOX (FULL FRAME) =====
        h, w, _ = frame.shape
        x1, y1 = int(w * 0.1), int(h * 0.1)
        x2, y2 = int(w * 0.9), int(h * 0.9)

        color = (0, 255, 0) if gender == "Male" else (255, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{gender}, Age: {age}"
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + 220, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ===== SHOW IMAGE =====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.image_label.size()
            )
        )



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())
