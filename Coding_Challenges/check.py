import os

CWD = './algorithms_500_DeutschJozsaStrikesAgain_template'
EXE = 'deustch_jozsa_strikes_again_template.py'

if __name__=="__main__":
    cwd = os.getcwd()
    print(cwd)
    for i in range(1, 5):
        stream = os.popen(f'python {CWD}/{EXE} < {CWD}/{i}.in')
        output = stream.read()
        with open(f'{CWD}/{i}.ans', 'r') as f:
            for line in f.readlines():
                print(f'test # {i}: {str(output).strip()==line}')
                print(f'{str(output).strip()}, {line}')