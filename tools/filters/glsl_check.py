"""Extract the filter compute shaders (kCountCS, kToneCS) from core/src/gpu.cpp and compile them with glslc.
  python -I glsl_check.py GLSLC OUTDIR"""
import os, re, subprocess, sys
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'core', 'src', 'gpu.cpp'), encoding='utf-8').read()
tone = re.search(r'#define TONE_COMMON R"\((.*?)\)"', src, re.S).group(1)
for name in ('kCountCS', 'kToneCS'):
    m = re.search(r'static const char\* ' + name + r' = "(.*?)" TONE_COMMON R"\((.*?)\)";', src, re.S)
    head = m.group(1).encode().decode('unicode_escape')
    code = head + tone + m.group(2)
    f = os.path.join(sys.argv[2], name + '.comp')
    open(f, 'w').write(code)
    r = subprocess.run([sys.argv[1], '--target-env=opengl', '-fauto-map-locations', '-fauto-bind-uniforms', '-fshader-stage=compute', f, '-o', f + '.spv'], capture_output=True, text=True)
    print(name, 'OK' if r.returncode == 0 else 'FAIL', r.stderr[-800:])
