import sys
import os

if __name__=="__main__":
    cwd = os.getcwd()
    print(cwd)
    for i in range(1, 3):
        stream = os.popen(f'python3 ./Coding_Challenges/algorithms_200_AdaptingTopology_template/adapting_topology_template.py < ./Coding_Challenges/algorithms_200_AdaptingTopology_template/{i}.in')
        output = stream.read()
        with open(f'Coding_Challenges/algorithms_200_AdaptingTopology_template/{i}.ans', 'r') as f:
            for line in f.readlines():
                print(f'test # {i}: {str(output).strip()==line}')