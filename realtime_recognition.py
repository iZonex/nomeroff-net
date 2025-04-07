import cv2
import numpy as np
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
from datetime import datetime
import threading
import time

class LicensePlateRecognizer:
    def __init__(self):
        # Инициализация пайплайна с оптимизированными параметрами
        print("Инициализация пайплайна...")
        
        self.recognizer = pipeline("number_plate_detection_and_reading", 
                                 image_loader="opencv",
                                 min_accuracy=0.05,  # Очень низкий порог точности
                                 off_number_plate_classification=False,  # Включаем классификацию
                                 default_label="eu_ua_2015",  # Стандартный формат
                                 default_lines_count=1)  # Одна строка для номера
        
        # Настройки оптимизации
        self.frame_skip = 9  # Обрабатываем каждый 10-й кадр
        self.frame_count = 0
        self.last_recognized = datetime.now()
        self.target_width = 640  # Целевая ширина кадра для распознавания
        
        # Словарь замен для исправления распространенных ошибок OCR
        self.corrections = {
            "PYIHON": "PYTHON",
            "PYlHON": "PYTHON",
            "РУTHON": "PYTHON"
        }
        
        # Буферы для работы с кадрами в отдельном потоке
        self.current_frame = None
        self.processed_frame = None
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.recognition_results = []
        
    def print_recognized_plate(self, text, conf, region):
        """Вывод распознанного номера в консоль"""
        current_time = datetime.now().strftime("%H:%M:%S")
        confidence = int(np.mean(conf) * 100)
        # Применяем коррекцию текста, если он есть в словаре замен
        if text in self.corrections:
            corrected_text = self.corrections[text]
            print(f"[{current_time}] Номер: {text} -> {corrected_text} | Уверенность: {confidence}% | Регион: {region}")
            return corrected_text, confidence, region
        else:
            print(f"[{current_time}] Номер: {text} | Уверенность: {confidence}% | Регион: {region}")
            return text, confidence, region
    
    def resize_frame(self, frame):
        """Изменение размера кадра для оптимизации"""
        height, width = frame.shape[:2]
        scale = self.target_width / width
        new_height = int(height * scale)
        return cv2.resize(frame, (self.target_width, new_height))
    
    def process_frame_thread(self):
        """Обработка кадра в отдельном потоке"""
        while True:
            # Блокируем доступ к буферам
            with self.processing_lock:
                if self.current_frame is None or self.is_processing:
                    time.sleep(0.01)  # Небольшая задержка для снижения нагрузки на CPU
                    continue
                    
                # Копируем текущий кадр для обработки
                frame = self.current_frame.copy()
                self.is_processing = True
            
            try:
                # Уменьшаем размер кадра для ускорения распознавания
                small_frame = self.resize_frame(frame)
                    
                # Повышаем контрастность для лучшего распознавания
                enhanced_frame = self.enhance_image(small_frame)
                
                # Распознаем номера на кадре
                results = self.recognizer([enhanced_frame])
                
                # Распаковываем результаты
                images, images_bboxs, images_points, images_zones, region_ids, region_names, count_lines, confidences, texts = unzip(results)
                
                # Обрабатываем результаты распознавания
                display_frame = frame.copy()
                recognition_results = []
                
                # Выводим информацию о найденных номерах
                if images_bboxs and len(images_bboxs[0]) > 0:
                    print(f"Найдено {len(images_bboxs[0])} номеров")
                    
                    for i, (bbox, text, conf) in enumerate(zip(images_bboxs[0], texts[0], confidences[0])):
                        # Получаем данные номера
                        region = region_names[0][i] if region_names and len(region_names[0]) > i else "Неизвестно"
                        
                        # Применяем коррекцию и выводим информацию
                        corrected_text, confidence, region = self.print_recognized_plate(text, conf, region)
                        
                        # Сохраняем результат распознавания
                        recognition_results.append((bbox, corrected_text, conf, region, confidence))
                        
                        # Рисуем рамку и текст
                        # Масштабируем координаты обратно к оригинальному размеру
                        scale = frame.shape[1] / enhanced_frame.shape[1]
                        x1, y1, x2, y2 = map(int, [c * scale for c in bbox[:4]])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{corrected_text} ({confidence}%)", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Обновляем время последнего распознавания
                    self.last_recognized = datetime.now()
                else:
                    # Проверяем, прошло ли больше 3 секунд с последнего распознавания
                    if (datetime.now() - self.last_recognized).seconds > 3:
                        print("Номера не найдены")
                
                # Блокируем доступ к буферам и обновляем результаты
                with self.processing_lock:
                    self.processed_frame = display_frame
                    self.recognition_results = recognition_results
                    self.is_processing = False
                        
            except Exception as e:
                print(f"Ошибка при обработке кадра: {e}")
                # В случае ошибки разблокируем обработку
                with self.processing_lock:
                    self.is_processing = False
    
    def enhance_image(self, image):
        """Улучшение изображения для лучшего распознавания"""
        # Применяем повышение контрастности
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_image
    
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
        
        # Запускаем поток обработки
        processing_thread = threading.Thread(target=self.process_frame_thread, daemon=True)
        processing_thread.start()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Не удалось получить кадр")
                    break
                
                # Увеличиваем счетчик кадров
                self.frame_count += 1
                
                # Обновляем текущий кадр каждый N-й кадр
                if self.frame_count % (self.frame_skip + 1) == 0:
                    with self.processing_lock:
                        self.current_frame = frame
                
                # Получаем обработанный кадр, если он есть
                display_frame = frame
                with self.processing_lock:
                    if self.processed_frame is not None:
                        display_frame = self.processed_frame
                
                # Показываем результат
                cv2.imshow('Распознавание номеров', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = LicensePlateRecognizer()
    recognizer.run() 