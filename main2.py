import os

run_script_times = 100

def get_common(list_of_winners):
    return max(set(list_of_winners), key=list_of_winners.count)

# for i in range(run_script_times):
#     os.system('python main.py')

file = open("Winners.txt", "r")

results = []
for i in range(run_script_times):
    results.append(file.readline())

file.close()

print(results)

print("Won most times: ")
print(get_common(results))
