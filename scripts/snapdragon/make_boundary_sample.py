"""Create a separate deterministic FIM boundary control; never adds it to evaluation."""
import argparse,json,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',required=True,type=Path);p.add_argument('--tokenizer',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--tokens',type=int,default=2016);a=p.parse_args()
 from tokenizers import Tokenizer
 tokenizer=Tokenizer.from_file(str(a.tokenizer));row=json.loads(a.corpus.read_text(encoding='utf-8').splitlines()[0]);base=row['prefix'];suffix=row['suffix']
 if a.tokens<1 or a.tokens+32>2048:p.error('Boundary prompt must leave32tokens in2048')
 for count in range(2049):
  prefix='//'+(' a'*count)+'\n'+base
  prompt='<|fim_prefix|>'+prefix+'<|fim_suffix|>'+suffix+'<|fim_middle|>'
  ids=tokenizer.encode(prompt,add_special_tokens=False).ids
  if len(ids)==a.tokens:break
 else:raise ValueError('No exact padding length found')
 row.update(id='boundary-'+str(a.tokens)+'-'+row['id'],prefix=prefix,prompt=prompt,prompt_text=prompt,prompt_ids=ids)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 print(json.dumps({'path':str(a.out),'prompt_tokens':len(ids),'max_new_tokens':32,'total_budget':len(ids)+32,'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),'scope':'separate context boundary control, excluded from200evaluation'}))
if __name__=='__main__':main()
