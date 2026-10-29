#!/usr/bin/env python3
"""Mil FIM sintéticos de preenchimento entre prefixo e sufixo em scripts completos."""
import argparse
import hashlib
import json
from pathlib import Path
import build
import render_contract as rc

CASES = ('filtro_id_sistema', 'string_parcial', 'gte', 'projecao', 'campo_aninhado', 'in',
         'pipeline_match', 'pipeline_project', 'insert_objeto', 'insert_array', 'update_set', 'update_inc')
EXTRA = ('objeto_aninhado', 'filtro_or', 'ordenacao', 'update_subdocumento', 'update_in',
         'insert_subdocumento', 'pipeline_group', 'campo_opcional')

def make_mid(split, j, expanded=False):
    n = 1 + 2*j
    base_index = 6*n + 1
    r = build.make(split, base_index)
    db, coll, pid, system, state, rank = build.SCHEMAS[split]
    q = lambda v: json.dumps(v, ensure_ascii=False)
    ident = f"MID-{split.upper()}-{n:05d}" if expanded else f"PRJ-{n + {'train':0,'val':10000,'test':20000}[split]:05d}"
    sysname = f'SIS-{"EXTRA-" if expanded else ""}{split.upper()}-{n:05d}'
    level = n % 5 + 1
    # Estas famílias compartilham 13-gramas com avaliação sob a política oficial;
    # ficam reservadas para eval no lote expandido, não apenas renomeadas.
    train_cases = [i for i in range(20) if i not in (12, 15, 17)]
    kind = train_cases[j % len(train_cases)] if expanded and split == 'train' else j % (20 if expanded else len(CASES))
    root = f'db.getCollection({q(coll)})'
    # O caractere ▌ é apenas uma marca do gerador e não é enviado como código.
    examples = [
      (f'Liste o projeto {ident} do sistema {sysname}, incluindo só ID e situação.', f'{root}.find({{{pid}: {q(ident)}, ▌}}, {{{pid}: 1, {state}: 1, _id: 0}})', f'{system}: {q(sysname)}'),
      (f'Busque o sistema {sysname} usando a string já iniciada.', f'{root}.find({{{system}: "▌"}})', sysname),
      (f'Busque no sistema {sysname} prioridade maior ou igual a {level}.', f'{root}.find({{{system}: {q(sysname)}, {rank}: {{$gte: ▌}}}})', str(level)),
      (f'Liste o sistema {sysname} projetando somente ID e situação, sem _id.', f'{root}.find({{{system}: {q(sysname)}}}, {{▌, _id: 0}})', f'{pid}: 1, {state}: 1'),
      (f'Busque no sistema {sysname} projetos da região sul.', f'{root}.find({{{system}: {q(sysname)}, "meta.regiao": "▌"}})', 'sul'),
      (f'Busque no sistema {sysname} projetos ativos ou pausados.', f'{root}.find({{{system}: {q(sysname)}, {state}: {{$in: ["ativo", ▌]}}}})', '"pausado"'),
      (f'Agregue projetos do sistema {sysname} e ordene prioridade decrescente.', f'{root}.aggregate([{{▌}}, {{$sort: {{{rank}: -1}}}}])', f'$match: {{{system}: {q(sysname)}}}'),
      (f'Agregue o sistema {sysname} e projete somente ID e situação, sem _id.', f'{root}.aggregate([{{$match: {{{system}: {q(sysname)}}}}}, {{$project: {{▌, _id: 0}}}}])', f'{pid}: 1, {state}: 1'),
      (f'Insira o projeto sintético {ident} no sistema {sysname} com situação ativo e prioridade {level}.', f'{root}.insertOne({{{pid}: {q(ident)}, ▌, {rank}: {level}}})', f'{system}: {q(sysname)}, {state}: "ativo"'),
      (f'Insira o projeto {ident} no sistema {sysname}, ativo e prioridade {level}, com etiquetas azul e verde.', f'{root}.insertOne({{{pid}: {q(ident)}, {system}: {q(sysname)}, {state}: "ativo", {rank}: {level}, etiquetas: ["azul", "▌"]}})', 'verde'),
      (f'Atualize a situação do projeto {ident} do sistema {sysname} para concluido.', f'{root}.updateOne({{{pid}: {q(ident)}, {system}: {q(sysname)}}}, {{$set: {{{state}: "▌"}}}})', 'concluido'),
      (f'Incremente em 1 a prioridade dos projetos do sistema {sysname}.', f'{root}.updateMany({{{system}: {q(sysname)}}}, {{$inc: {{{rank}: ▌}}}})', '1'),
      (f'Busque projetos com região sul e canal web no sistema {sysname}.', f'{root}.find({{{system}: {q(sysname)}, meta: {{regiao: "sul", ▌}}}})', 'canal: "web"'),
      (f'Busque projetos do sistema {sysname} ativos ou com prioridade {level}.', f'{root}.find({{{system}: {q(sysname)}, $or: [{{{state}: "ativo"}}, ▌]}})', f'{{{rank}: {level}}}'),
      (f'Liste o sistema {sysname} em ordem crescente de prioridade.', f'{root}.find({{{system}: {q(sysname)}}}).sort({{{rank}: ▌}})', '1'),
      (f'Altere os metadados do projeto {ident} no sistema {sysname} para região sul e canal web.', f'{root}.updateOne({{{pid}: {q(ident)}, {system}: {q(sysname)}}}, {{$set: {{meta: {{regiao: "sul", canal: "▌"}}}}}})', 'web'),
      (f'Marque projetos ativos ou pausados do sistema {sysname} como concluídos.', f'{root}.updateMany({{{system}: {q(sysname)}, {state}: {{$in: ["ativo", "pausado"]}}}}, {{$set: {{{state}: ▌}}}})', '"concluido"'),
      (f'Insira projeto {ident} no sistema {sysname}, ativo e prioridade {level}, com metadados de região sul e canal web.', f'{root}.insertOne({{{pid}: {q(ident)}, {system}: {q(sysname)}, {state}: "ativo", {rank}: {level}, meta: {{regiao: "sul"▌}}}})', ', canal: "web"'),
      (f'Conte por situação os projetos do sistema {sysname}.', f'{root}.aggregate([{{$match: {{{system}: {q(sysname)}}}}}, {{$group: {{_id: "${state}", total: {{$sum: ▌}}}}}}])', '1'),
      (f'Busque no sistema {sysname} os projetos que possuem campo observacao.', f'{root}.find({{{system}: {q(sysname)}, observacao: {{$exists: ▌}}}})', 'true'),
    ]
    instruction, statement, middle = examples[kind]
    statement = statement.replace('▌', '\x00')
    prefix_stmt, suffix_stmt = statement.split('\x00')
    schema = f'Esquema sintético: {pid}: string; {system}: string; {state}: string; {rank}: int; meta: objeto (regiao e canal: string); etiquetas: array de string; observacao: string opcional.'
    head = f'// {instruction}\n// {schema}\nconst ambienteConsulta = "demonstração";\nconst resultado = '
    tail = ';\n// O código posterior usa resultado; mantenha o sufixo existente.\n'
    prefix, suffix = head + prefix_stmt, suffix_stmt + tail
    case_name = (CASES + EXTRA)[kind]
    group = f'{db}::mid-{case_name}'
    r.update(id=f'pro-mid-{split}-{j:04d}' if not expanded else f'pro-mid-extra-{split}-{j:05d}', base_index=base_index, scenario=kind,
             group=group, group_id=group, split_key=group, template_id=f'pro-mid-{case_name}',
             prefix=prefix, middle=middle, suffix=suffix, cut_kind='partial', origin='gen:pro-mid-script-v2',
             task_scenario='mid_script_suffix', synthetic=True)
    r['facts'] = [{'kind':'FieldSchema','path':x,'type':t,'presence': 0.5 if x in ('meta.regiao','etiquetas','observacao') else 1.0} for x,t in
                  [(pid,'string'),(system,'string'),(state,'string'),(rank,'int'),('meta.regiao','string'),('etiquetas','array'),('observacao','string')]]
    r['editor']['result_fields'] = [pid,system,state,rank,'meta.regiao','etiquetas']
    r['result_fields'] = r['editor']['result_fields']
    r['target'] = middle
    r['editor_text'] = prefix+'▌'+suffix
    r['prompt'] = rc.qwen_fim_text(rc.render_model_prefix(prefix,r['editor'],r['language']),suffix)
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expanded',action='store_true', help='lote adicional de 10 mil em 20 famílias')
    ap.add_argument('--out',type=Path)
    args = ap.parse_args()
    args.out = args.out or build.ROOT/'data/generated'/('pro-mid-script-extra-v4' if args.expanded else 'pro-mid-script-v4')
    args.out.mkdir(parents=True,exist_ok=True)
    paths = [args.out/f'fim-{s}.jsonl' for s in ('train','val','test')] + [args.out/'manifest.json']
    if any(p.exists() for p in paths):
        raise SystemExit('Recusando sobrescrever o lote existente; escolha outro --out')
    manifest = {}
    counts = [('train',8000),('val',1000),('test',1000)] if args.expanded else [('train',800),('val',100),('test',100)]
    for split,count in counts:
        path = args.out/f'fim-{split}.jsonl'
        h = hashlib.sha256()
        with path.open('xb') as f:
            for j in range(count):
                data = (json.dumps(make_mid(split,j,args.expanded),ensure_ascii=False,separators=(',',':'))+'\n').encode()
                f.write(data); h.update(data)
        manifest[path.name] = {'count':count,'sha256':h.hexdigest()}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()
