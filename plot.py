import matplotlib.pyplot as plt

m = int(input("Enter the value of m: "))
n = int(input("Enter the value of n: "))
k = int(input("Enter the value of k: "))

baseline_data = []
stream_dann_data = []
training_after_upload_data = []

print("Enter data for the STREAM-DANN model (in the format 'time elapsed current accuracy'):")
for _ in range(m):
    time_elapsed, accuracy = map(float, input().split())
    stream_dann_data.append((time_elapsed, accuracy))

print("Enter data for the baseline model (in the format 'time elapsed current accuracy'):")
for _ in range(n):
    time_elapsed, accuracy = map(float, input().split())
    baseline_data.append((time_elapsed, accuracy))

print("Enter data for the training after upload model (in the format 'time elapsed current accuracy'):")
for _ in range(k):
    time_elapsed, accuracy = map(float, input().split())
    training_after_upload_data.append((time_elapsed, accuracy))

def moving_average(data, window_size=3):
    smoothed_data = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        smoothed_value = sum(val for _, val in window) / window_size
        smoothed_data.append((window[window_size // 2][0], smoothed_value))
    return smoothed_data

plt.figure()
plt.plot(*zip(*moving_average(stream_dann_data)), color='blue', label='STREAM-DANN Model')
plt.plot(*zip(*moving_average(baseline_data)), color='orange', label='Baseline Model')
plt.plot(*zip(*moving_average(training_after_upload_data)), color='purple', label='Training after uploading Model')
plt.xlabel('Time Elapsed')
plt.ylabel('Current Accuracy')
plt.title('Model Performance Comparison')
plt.legend()

# 保存图表
plt.savefig('smooth_model_performance_plot.png')
plt.show()
