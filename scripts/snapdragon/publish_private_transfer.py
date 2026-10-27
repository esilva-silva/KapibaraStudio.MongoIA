"""Transfer a manifested package to a private HF branch; verify an immutable snapshot."""
import argparse
import json
import re
from pathlib import Path
from download_exports import safe_subfolder
from download_qnn_prepared import file_inventory, pin
from export_gpu import sha256
from verify_manifest import verify


def validate_package(package, manifest_sha256):
    package=Path(package).resolve(strict=True)
    manifest=package/'export_manifest.json'; expected=pin(manifest_sha256,64)
    if sha256(manifest)!=expected:raise ValueError('Manifest SHA mismatch')
    data=json.loads(manifest.read_text(encoding='utf-8-sig'))
    entries=file_inventory(data['files'])
    actual={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    if actual!={e['path'] for e in entries}|{'export_manifest.json'}:raise ValueError('Package inventory mismatch')
    if any(p.is_symlink() for p in package.rglob('*')):raise ValueError('Package symlinks unsupported')
    result=verify(package,'export_manifest.json')
    if not result['passed']:raise ValueError('Package integrity failed')
    return package,entries+[{'path':'export_manifest.json','bytes':manifest.stat().st_size,'sha256':expected}]


def require_private(api, repo):
    info=api.repo_info(repo,repo_type='model')
    if info.private is not True:raise ValueError('Repository must be private')
    return info


def save(path, report):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def transfer(package, repo, branch, subfolder, base_commit, manifest_sha256, out, verification_root):
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import RevisionNotFoundError
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+',repo):raise ValueError('Invalid repository ID')
    if not branch.startswith('transfer/') or '..' in branch or not re.fullmatch(r'[A-Za-z0-9_./-]+',branch):raise ValueError('Branch must be under transfer/')
    subfolder=safe_subfolder(subfolder);base=pin(base_commit,40)
    package,entries=validate_package(package,manifest_sha256)
    out=Path(out).resolve();verification_root=Path(verification_root).resolve()
    if verification_root.exists():raise ValueError('Verification root must be new')
    api=HfApi()
    if api.whoami()['name'] != repo.split('/')[0]:raise ValueError('Repository owner must be authenticated user')
    require_private(api,repo)
    report={'status':'preparing','repo_id':repo,'branch':branch,'subfolder':subfolder,'base_commit':base,
            'manifest_sha256':pin(manifest_sha256,64),'private':True,'files':{}}
    if out.exists():
        previous=json.loads(out.read_text(encoding='utf-8'))
        for key in ('repo_id','branch','subfolder','base_commit','manifest_sha256'):
            if previous.get(key)!=report[key]:raise ValueError('Existing report mismatches request')
        if not previous.get('commit'):raise ValueError('Existing report has no commit; inspect branch before retry')
        report=previous;commit=pin(report['commit'],40)
    else:
        try:
            head=api.repo_info(repo,repo_type='model',revision=branch).sha
        except RevisionNotFoundError:
            require_private(api,repo)
            api.create_branch(repo_id=repo,repo_type='model',branch=branch,revision=base,exist_ok=True)
            head=api.repo_info(repo,repo_type='model',revision=branch).sha
        if head!=base:raise ValueError('Branch has advanced; refusing overwrite')
        require_private(api,repo)
        report['status']='uploading';report['parent_commit']=head;save(out,report)
        result=api.upload_folder(repo_id=repo,repo_type='model',folder_path=package,path_in_repo=subfolder,
            revision=branch,parent_commit=head,commit_message='Transfer manifested Snapdragon artifact')
        commit=pin(result.oid,40);report['commit']=commit;report['status']='verifying';save(out,report)
    require_private(api,repo)
    snapshot_download(repo_id=repo,repo_type='model',revision=commit,local_dir=verification_root,
                      allow_patterns=[subfolder+'/**'],max_workers=2)
    downloaded=verification_root/Path(*subfolder.split('/'))
    expected_names={e['path'] for e in entries}
    actual_names={p.relative_to(downloaded).as_posix() for p in downloaded.rglob('*') if p.is_file()}
    if actual_names!=expected_names:raise ValueError('Downloaded inventory mismatch')
    for entry in entries:
        path=(downloaded/entry['path']).resolve(strict=True)
        if not path.is_relative_to(downloaded.resolve()) or not path.is_file() or path.stat().st_size!=entry['bytes'] or sha256(path)!=entry['sha256']:
            raise ValueError('Downloaded file mismatch')
        report['files'][entry['path']]={'bytes':entry['bytes'],'sha256':entry['sha256']}
    require_private(api,repo)
    report.update(status='verified',private=True,commit=commit,verification_root=str(verification_root));save(out,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','repo-id','branch','subfolder','base-commit','manifest-sha256','out','verification-root'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    try:
        r=transfer(a.package,a.repo_id,a.branch,a.subfolder,a.base_commit,a.manifest_sha256,a.out,a.verification_root)
        print(json.dumps({'status':r['status'],'repo_id':r['repo_id'],'commit':r['commit'],'private':r['private']},indent=2))
    except Exception:
        p.exit(1,'Private transfer failed; inspect the local report and branch before retry.\n')

if __name__=='__main__':main()
