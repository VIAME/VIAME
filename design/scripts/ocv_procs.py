"""OpenCV-backed process names.

Two registration shapes: `PLUGIN_INFO( "name", ... )` inside the process
class, and an explicit `add_attribute( PLUGIN_NAME, "name" )` beside a
`create_new_process< Class >` in register_processes.cxx.

Run from the repository root:

    viame registry-dump --json --output /tmp/registry.json
    python3 design/scripts/ocv_procs.py /tmp/registry.json
"""
import re, os, subprocess, json, sys

ocv_files = set(subprocess.check_output(
    ["git","grep","-ln","opencv2/","--","library","tools","plugins"],text=True).split())
def twins(p):
    stem=p.rsplit('.',1)[0]
    return [x for x in (stem+'.h',stem+'.cxx') if os.path.exists(x)]
ocv_dirs = {os.path.dirname(f) for f in ocv_files}

def ocv(p):
    """OpenCV-backed: this file, its twin, or the library it is part of.

    A process class often keeps OpenCV out of its own header and reaches it
    through the algorithm it wraps -- `pair_stereo_detections_process.h` has
    no `opencv2/` in it but `pair_stereo_detections.cxx`, beside it and in the
    same library, is nothing but OpenCV. Directory granularity is what makes
    that visible, and it is the granularity phase 7 removes things at anyway.
    """
    if p in ocv_files or any(t in ocv_files for t in twins(p)):
        return "direct"
    if os.path.dirname(p) in ocv_dirs:
        return "library"
    return None

registered = set(json.load(open(sys.argv[1]))['processes'])

names = {}
info_pat = re.compile(r'PLUGIN_INFO\s*\(\s*"([^"]+)"')
for path in subprocess.check_output(["git","ls-files","library","plugins","tools"],
                                    text=True).split():
    if not path.endswith(('.h','.cxx')): continue
    for m in info_pat.finditer(open(path,errors='replace').read()):
        names.setdefault(m.group(1), path)

blk_pat = re.compile(
    r'create_new_process<\s*([\w:]+)\s*>[\s\S]{0,400}?'
    r'add_attribute\(\s*kvpf::PLUGIN_NAME\s*,\s*"([^"]+)"')
for reg_src in subprocess.check_output(
        ["git","ls-files","*register_processes.cxx"],text=True).split():
    text = open(reg_src, errors='replace').read()
    for m in blk_pat.finditer(text):
        cls = m.group(1).split('::')[-1]
        d = os.path.dirname(reg_src)
        for cand in (os.path.join(d, cls + '.h'), os.path.join(d, cls + '.cxx')):
            if os.path.exists(cand):
                names.setdefault(m.group(2), cand)
                break
        else:
            names.setdefault(m.group(2), reg_src)

hit = sorted((n, p, ocv(p)) for n, p in names.items() if ocv(p))
print("%d process names resolved, %d OpenCV-backed (%d directly)\n"
      % (len(names), len(hit), sum(1 for _, _, k in hit if k == "direct")))
for n, p, kind in hit:
    print("%-40s %-8s %s%s" % (n, kind, p,
                               "" if n in registered else "   [NOT REGISTERED]"))
unreg = sorted(n for n in names if n not in registered)
if unreg:
    print("\nnot registered by this install: %s" % ", ".join(unreg))
