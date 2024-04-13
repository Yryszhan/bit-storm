import cv2
import numpy as np

# Загрузите имена классов и инициализируйте цвета
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Пути к файлам весов и конфигурации YOLO
weightsPath = "yolov4.weights"
configPath = "yolov4.cfg"

# Загрузите сеть YOLO
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Получите имена всех слоев
all_layer_names = net.getLayerNames()
out_layer_indexes = net.getUnconnectedOutLayers().flatten()
out_layers = [all_layer_names[i - 1] for i in out_layer_indexes]

# Запустите поток видео
cap = cv2.VideoCapture('http://192.168.179.58:81/stream')

while True:
    # Чтение кадра из видеопотока
    ret, frame = cap.read()
    if not ret:
        break

    (H, W) = frame.shape[:2]

    # Построение blob и выполнение прямого распространения
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(out_layers)

    boxes = []
    confidences = []
    classIDs = []

    # Обработка результатов обнаружения
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Отрисовка ограничивающих рамок и меток и вывод в консоль для "bottle"
    if len(idxs) > 0:
        for i in idxs.flatten():
            if LABELS[classIDs[i]] == "bottle":  # Проверка, является ли объект баклашкой
                print("Баклашка обнаружена")  # Вывод в консоль

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
