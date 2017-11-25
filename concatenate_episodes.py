import sys, os

result = ""

for file in os.listdir(sys.argv[1]):
    result += open(os.path.join(sys.argv[1], file), encoding='utf-8').read()

open(os.path.join(sys.argv[1], 'episodes.txt'), 'w', encoding='utf-8').write(result)
