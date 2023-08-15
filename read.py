#python "C:\Users\13764\Documents\academics\sr\spring 2023\physics intern\matec\read.py" "C:\Users\13764\Documents\academics\sr\spring 2023\physics intern\2022-2023 Ultrasonics\3-22-17 top (ds)\top\7-7-23 Z+ gel heat\heat 2\200.txt"
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")
# Usage: python <program path> <.txt file path>
path = sys.argv[1]

with open(path) as f:
    lines = f.readlines()

def extract_data(input):
    # organize the text file into separate graphs, based on line separation
    output = [] 
    temp = []
    for line in input:
        if line == "\n":
            output.append(temp)
            temp = []
            continue
        temp.append(line.strip())
    output.append(temp)

    # crops the header
    output = output[1:]

    for i in range(len(output)):
        graph = output[i]
        print(len(graph))
        length = int(graph[16]) # locates the length
        print(length)

        # cuts the data portion
        for j in range(17, 17+length):
            try:
                graph[j] = (int(graph[j]) - 128) / 128 * 100
            except:
                break
        output[i] = graph[17:(17+length)]

    return output

def find_peak_amp(data, start, end):
    # start, end time are in us. 
    start *= 100
    end *= 100
    t = start
    max_time = t
    max_amp = 0
    for point in data[start:end]:
        if np.abs(point) > max_amp:
            max_amp = np.abs(point)
            max_time = t
        t += 1
    return [max_time, max_amp]

result = extract_data(lines)

for batch in result:
    x = np.arange(0, 0.01 * len(batch), 0.01)
    print(len(x), len(batch))
    plt.plot(x, np.array(batch).astype(float))
    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude (%)")
    plt.ylim([-100, 100])
plt.show()
