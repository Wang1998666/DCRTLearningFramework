"""One-command reproduction of the current two-stage system."""
import subprocess, sys
from pathlib import Path
HERE=Path(__file__).parent
for script in ["26_occurrence_features_v9.py","27_dcrank_searoute_repaired.py","31_occurrence_stage_v9.py",
               "32_redsea_transfer_dcrank_v9.py","33_two_stage_system_v9.py",
               "28_dcrank_searoute_figures.py","34_two_stage_figures_v9.py",
               "36_ml_diagnostics_figures_v9.py","37_geo_network_figures_v9.py",
               "29_dcrank_searoute_audit.py","35_two_stage_audit_v9.py"]:
    print(f"\n=== Running {script} ===",flush=True)
    subprocess.run([sys.executable,"-B",str(HERE/script)],cwd=HERE,check=True)
print("\nTwo-stage v9 reproduction completed and audited.")
