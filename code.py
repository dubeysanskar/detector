!pip install bitsandbytes transformers accelerate peft -q
!pip install -U "huggingface_hub[cli]"
!huggingface-cli login
from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32).to(device)
processor = PaliGemmaProcessor.from_pretrained(model_id)

def draw_bounding_box(image, coordinates, label, width, height):
    global label_colors
    y1, x1, y2, x2 = coordinates
    y1, x1, y2, x2 = map(round, (y1 * height, x1 * width, y2 * height, x2 * width))

    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    text_width, text_height = text_size

    text_x = x1 + 2
    text_y = y1 - 5

    font_scale = 1
    label_rect_width = text_width + 8
    label_rect_height = int(text_height * font_scale)

    color = label_colors.get(label, None)
    if color is None:
        color = np.random.randint(0, 256, (3,)).tolist()
        label_colors[label] = color

    cv2.rectangle(image, (x1, y1 - label_rect_height), (x1 + label_rect_width, y1), color, -1)
    thickness = 2
    cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

input_video = 'input_video.mp4'
cap = cv2.VideoCapture(input_video)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = 'output_video.avi'
out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

label_colors = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    prompt = "detect person, phone, bottle"

    inputs = processor(text=prompt, images=img, padding="longest", do_convert_rgb=True, return_tensors="pt").to(device)
    inputs = inputs.to(dtype=model.dtype)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=496)

    paligemma_response = processor.decode(output[0], skip_special_tokens=True)[len(prompt):].lstrip("\n")
    detections = paligemma_response.split(" ; ")

    parsed_coordinates = []
    labels = []

    for item in detections:
        if "=" in item:
            parts = item.split("= [")
            label = parts[0].strip()
            labels.append(label)
            coordinates = parts[1].replace("]", "").split(",")
            coordinates = [float(coord) / 1024 for coord in coordinates]
            parsed_coordinates.append(coordinates)

    width = img.size[0]
    height = img.size[1]

    output_frame = frame.copy()  # Initialize output_frame with the current frame
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for coordinates, label in zip(parsed_coordinates, labels):
        output_frame = draw_bounding_box(output_frame, coordinates, label, width, height)

    out.write(output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video {output_file} saved to disk.")
