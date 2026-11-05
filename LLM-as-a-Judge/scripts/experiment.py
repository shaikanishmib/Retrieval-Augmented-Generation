import sys
from llm_judge.study import development, heldout
if sys.argv[1] == 'development':
    development()
elif sys.argv[1] == 'heldout':
    heldout()
else:
    raise SystemExit('Choose development or heldout')
