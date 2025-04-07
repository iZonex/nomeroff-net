import cv2
import numpy as np
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
from datetime import datetime

class LicensePlateRecognizer:
    def __init__(self):
        # Инициализация пайплайна аналогично работающему test_recognition.py
        print("Инициализация пайплайна...")
        self.recognizer = pipeline("number_plate_detection_and_reading", 
                                 image_loader="opencv",
                                 min_accuracy=0.1,  # Очень низкий порог точности
                                 off_number_plate_classification=False,  # Включаем классификацию
                                 default_label="eu_ua_2015",
                                 default_lines_count=1)
        
        # Настройки оптимизации
        self.frame_skip = 5  # Обрабатываем каждый 6-й кадр
        self.frame_count = 0
        self.last_recognized = datetime.now()
        
    def print_recognized_plate(self, text, conf, region):
        """Вывод распознанного номера в консоль"""
        current_time = datetime.now().strftime("%H:%M:%S")
        confidence = int(np.mean(conf) * 100)
        print(f"[{current_time}] Номер: {text} | Уверенность: {confidence}% | Регион: {region}")
    
    def process_frame(self, frame):
        """Обработка кадра"""
        # Исходное изображение для отображения
        display_frame = frame.copy()
        
        try:
            # Пропускаем кадры для оптимизации
            self.frame_count += 1
            if self.frame_count % (self.frame_skip + 1) != 0:
                return display_frame
                
            # Проверяем размеры кадра
            height, width = frame.shape[:2]
            
            # Распознаем номера на кадре
            results = self.recognizer([frame])
            
            # Распаковываем результаты (как в test_recognition.py)
            images, images_bboxs, images_points, images_zones, region_ids, region_names, count_lines, confidences, texts = unzip(results)
            
            # Выводим информацию о найденных номерах
            if images_bboxs and len(images_bboxs[0]) > 0:
                print(f"Найдено {len(images_bboxs[0])} номеров")
                
                for i, (bbox, text, conf) in enumerate(zip(images_bboxs[0], texts[0], confidences[0])):
                    # Получаем данные номера
                    region = region_names[0][i] if region_names and len(region_names[0]) > i else "Неизвестно"
                    confidence = int(np.mean(conf) * 100)
                    
                    # Выводим информацию
                    self.print_recognized_plate(text, conf, region)
                    
                    # Рисуем рамку и текст
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{text} ({confidence}%)", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Обновляем время последнего распознавания
                self.last_recognized = datetime.now()
            else:
                # Проверяем, прошло ли больше 3 секунд с последнего распознавания
                if (datetime.now() - self.last_recognized).seconds > 3:
                    print("Номера не найдены")
                    
        except Exception as e:
            print(f"Ошибка при обработке кадра: {e}")
            
        return display_frame
    
    def run(self):
        """Запуск распознавания в реальном времени"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Не удалось открыть камеру")
            return
        
        # Устанавливаем разрешение камеры
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Камера запущена! Нажмите 'q' для выхода")
        print("-" * 50)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Не удалось получить кадр")
                    break
                
                # Обрабатываем кадр
                processed_frame = self.process_frame(frame)
                
                # Показываем результат
                cv2.imshow('Распознавание номеров', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = LicensePlateRecognizer()
    recognizer.run() 