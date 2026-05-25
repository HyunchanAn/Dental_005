import cv2
import time
from ultralytics import YOLO

def run_benchmark(model_path, test_image_path, output_name):
    print(f"\n--- Loading model: {model_path} ---")
    model = YOLO(model_path)
    
    # Warmup
    print("Warming up...")
    _ = model(test_image_path, imgsz=960, conf=0.25, verbose=False)
    
    # Benchmark
    print("Benchmarking latency...")
    start_time = time.time()
    results = model(test_image_path, imgsz=960, conf=0.25)
    end_time = time.time()
    
    latency = (end_time - start_time) * 1000
    print(f"Latency: {latency:.2f} ms")
    
    # Save output
    for r in results:
        im_array = r.plot()
        cv2.imwrite(output_name, im_array)
        
    return latency

if __name__ == "__main__":
    test_image = "data/raw/images/test/test_000.jpg"
    
    # 9-class model
    latency_9 = run_benchmark("yolov8x_AlphaDent_9_classes_960px.pt", test_image, "output_marked_9_classes.jpg")
    
    # 4-class model
    latency_4 = run_benchmark("yolov8x_AlphaDent_4_classes_960px.pt", test_image, "output_marked_4_classes.jpg")
    
    with open("development_log.txt", "a", encoding="utf-8") as f:
        f.write("\n[Benchmark Results]\n")
        f.write(f"- 9-class (960px) Latency: {latency_9:.2f} ms\n")
        f.write(f"- 4-class (960px) Latency: {latency_4:.2f} ms\n")
