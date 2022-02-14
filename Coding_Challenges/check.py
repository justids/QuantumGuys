import os

CWD = 'Coding_Challenges/algorithms_300_AdderQFT_template'
EXE = 'adder_QFT_template.py'

if __name__=="__main__":
    cwd = os.getcwd()
    print(cwd)
    for i in range(1, 3):
        stream = os.popen(f'python {CWD}/{EXE} < {CWD}/{i}.in')
        output = stream.read()
        with open(f'{CWD}/{i}.ans', 'r') as f:
            for line in f.readlines():
                print(f'test # {i}: {str(output).strip()==line}')