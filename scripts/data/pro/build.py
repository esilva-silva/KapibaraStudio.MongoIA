#!/usr/bin/env python3
"""Lote sintético complementar Pro; não modifica os datasets existentes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'data'))
import render_contract as rc

SCHEMAS = {
    'train': ('pro_sintetico_treino', 'projetos_demo', 'projetoId', 'sistema', 'situacao', 'prioridade'),
    'val': ('pro_sintetico_validacao', 'iniciativas_demo', 'codigoProjeto', 'sistemaOrigem', 'estado', 'nivel'),
    'test': ('pro_sintetico_teste', 'carteira_demo', 'idProjeto', 'sistemaResponsavel', 'status', 'grau'),
}

def make(split, i):
    db, coll, pid, system, state, rank = SCHEMAS[split]
    n = i // 6
    value_n = n + {'train': 0, 'val': 10000, 'test': 20000}[split]
    case = i % 6
    ident = f'PRJ-{value_n:05d}'
    sysname = f'SIS-{split.upper()}-{n % 71:02d}'
    status = ('ativo', 'pausado', 'concluido')[n % 3]
    p = n % 5 + 1
    q = json.dumps
    specs = [
        (f'Busque o projeto pelo ID {ident} e pelo sistema {sysname}.', f'findOne({{{pid}: {q(ident)}, {system}: {q(sysname)}}})', 'O filtro exige os dois valores; findOne retorna um documento ou null.'),
        (f'Liste projetos do sistema {sysname} com situação {status}.', f'find({{{system}: {q(sysname)}, {state}: {q(status)}}})', 'As duas condições são combinadas por AND.'),
        (f'Conte projetos do sistema {sysname} com prioridade maior ou igual a {p}.', f'countDocuments({{{system}: {q(sysname)}, {rank}: {{$gte: {p}}}}})', 'A contagem usa o sistema e o limite inclusivo de prioridade.'),
        (f'Liste projetos do sistema {sysname}, exceto os de situação {status}.', f'find({{{system}: {q(sysname)}, {state}: {{$ne: {q(status)}}}}})', 'O filtro exclui esse valor de situação; campos ausentes também satisfazem $ne.'),
        (f'Liste até 5 projetos do sistema {sysname}, em ordem decrescente de prioridade.', f'find({{{system}: {q(sysname)}}}).sort({{{rank}: -1}}).limit(5)', 'A ordenação vem antes do limite; empates não têm ordem garantida.'),
        (f'Busque os projetos {ident} no sistema {sysname}, retornando apenas ID e situação sem _id.', f'find({{{pid}: {q(ident)}, {system}: {q(sysname)}}}, {{{pid}: 1, {state}: 1, _id: 0}})', 'A projeção inclui ID e situação e exclui _id.'),
    ]
    instruction, body, explanation = specs[case]
    code = f'db.getCollection({q(coll)}).{body};'
    schema = f'Esquema sintético: {pid}: string; {system}: string; {state}: string; {rank}: int. Todos os campos estão presentes.'
    context = f'{schema}\nParâmetros sintéticos: ID={ident}; sistema={sysname}; situação={status}; prioridade={p}.'
    group = f'{db}::{case}'
    common = dict(id=f'pro-{split}-{i:05d}', schema_version='kapi-ds-record-v2', domain='pro_sintetico', schema_id=db,
                  collection=coll, split=split, schema_split_role='train' if split == 'train' else 'eval', holdout=False,
                  group=group, group_id=group, split_key=group, template_id=f'pro-{case}', origin='gen:pro-synthetic-v1',
                  source_license='generated', label_source='reference', weight=1.0, synthetic=True, scenario=case)
    if n % 4 == 0:
        return {**common, 'type':'chat', 'task':'edit_short', 'kind':'chat', 'language':'pt-BR',
                'instruction':instruction, 'context':{'connection':'local-sintetico', 'database':db, 'collection':coll,
                'selection':None, 'editor':context, 'known_collections':[coll]}, 'output':f'// {explanation}\n{code}'}
    # A instrução e os parâmetros ficam no texto do editor; o contexto de IDE usa somente campos suportados.
    doc = f'// {instruction}\n// {schema}\n// Parâmetros sintéticos: ID={ident}; sistema={sysname}; situação={status}; prioridade={p}.\n{code}'
    start = doc.index(body)
    # Alterna o local do corte sem romper a reconstrução do código.
    ends = [len(body), max(1, body.index('(') + 1), max(1, len(body)//2)]
    end = ends[(i // 6) % len(ends)]
    prefix, middle, suffix = doc[:start], body[:end], doc[start+end:]
    editor = {'target': {'connection':'local-sintetico', 'database':db, 'collection':coll},
              'known_names':['local-sintetico', db, coll], 'result_fields':[pid, system, state, rank],
              'recent_commands':[], 'input_panel':'', 'newline':'\r\n'}
    prompt = rc.qwen_fim_text(rc.render_model_prefix(prefix, editor, rc.LANGUAGE_CONSOLE), suffix)
    return {**common, 'type':'fim', 'kind':'fim', 'language':rc.LANGUAGE_CONSOLE, 'contrato':'editor-context-v1',
            'context_kind':'full', 'category':'crud', 'cut_kind':'partial', 'prefix':prefix, 'middle':middle, 'suffix':suffix,
            'target':middle, 'editor_text':prefix+'▌'+suffix, 'editor':editor, 'facts':[{'kind':'FieldSchema','path':x,'type':'int' if x == rank else 'string','presence':1.0} for x in (pid,system,state,rank)],
            'known_names':editor['known_names'], 'result_fields':editor['result_fields'], 'recent_commands':[], 'prompt':prompt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'data/generated/pro-context-v1b')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f'{kind}-{split}.jsonl').exists() for split in SCHEMAS for kind in ('fim', 'chat')) or (args.out / 'manifest.json').exists():
        raise SystemExit(f'Recusando sobrescrever arquivos em {args.out}; escolha outro --out')
    manifest = {}
    for split, count in [('train', 24000), ('val', 3000), ('test', 3000)]:
        for kind in ('fim', 'chat'):
            path = args.out / f'{kind}-{split}.jsonl'
            if path.exists():
                raise SystemExit(f'Recusando sobrescrever {path}; escolha outro --out')
            digest = hashlib.sha256()
            total = 0
            with path.open('xb') as f:
                for i in range(count):
                    r = make(split, i)
                    if r['type'] != kind:
                        continue
                    line = (json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8')
                    f.write(line); digest.update(line); total += 1
            manifest[path.name] = {'count':total, 'sha256':digest.hexdigest()}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
