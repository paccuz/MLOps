from prometheus_client import start_http_server, Gauge
import time, random

model_latency = Gauge('model_inference_latency', 'Time taken for model inference')
model_accuracy = Gauge('model_accuracy', 'Accuracy of the ML model')

def generate_metrics():
    while True:
        model_latency.set(random.uniform(0.1, 1.0))
        model_accuracy.set(random.uniform(0.7, 0.99))
        time.sleep(5)

if __name__ == "__main__":
    start_http_server(8001)  # Expose metrics at localhost:8001/metrics
    generate_metrics()
