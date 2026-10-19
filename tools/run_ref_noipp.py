"""Run a reference script with OpenCV's IPP disabled (Android OpenCV has no IPP).
usage: python tools/run_ref_noipp.py REF_CODE_DIR SCRIPT.py ARGS..."""
import sys, os, runpy, cv2
cv2.ipp.setUseIPP(False)
code = os.path.abspath(sys.argv[1]); os.chdir(code); sys.path.insert(0, code)
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
