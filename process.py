with open("results.txt", "r") as results:
    result = [i.strip().split(" ") for i in results.read().split("\n") if len(i) > 1 and i[0] in map(str, range(10))]
tests = ["MLP256L", "MLP256M", "MLP256S", "TinyLidarNetL", "TinyLidarNetM", "TinyLidarNetS"]
tracks = ["GYM", "AUS", "MOS", "SPL"]
lists = []
for i in range(len(result)):
    if i % 10 == 0:
        lists.append([[],[]])
    if result[i][3] in ["LAP", "COLLISION:", "TIMEOUT:"]:
        lists[-1][0].append(float(result[i][-1]))
    if result[i][3] in ["LAP"]:
        lists[-1][1].append(float(result[i][-3][0:-1]))
accuracies = []
times = []
for i in range(len(lists)):
    if len(lists[i][1]) == 0:
        times.append(float('nan'))
    else:
        times.append(sum(lists[i][1])/len(lists[i][1]))
    accuracies.append(sum(lists[i][0])/len(lists[i][0]))
for i in range(len(times)):
    print(tests[i//4]+" on "+tracks[i%4]+": Time = "+str(times[i])+" and Accuracy = "+str(accuracies[i]))