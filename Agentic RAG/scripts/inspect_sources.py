"""Parse project Python sources and notebook code without writing bytecode."""
import ast
from agentic_rag.artifacts import ROOT,read

if __name__=='__main__':
    paths=[*list((ROOT/'src').rglob('*.py')),*list((ROOT/'scripts').glob('*.py')),*list((ROOT/'tests').glob('*.py'))]
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    print('Parsed',len(paths),'Python files')
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        notebook=read(path)
        code=[c for c in notebook['cells'] if c['cell_type']=='code']
        for index,cell in enumerate(code):
            source=''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
            ast.parse(source,filename=f'{path.name}:code-{index+1}')
        print('Parsed',len(code),'code cells in',path.name)
