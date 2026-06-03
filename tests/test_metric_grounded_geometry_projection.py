import pathlib
import importlib.util


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "test_metric_grounded_geometry_projection.py"
spec = importlib.util.spec_from_file_location("_metric_grounded_geometry_projection_script_tests", SCRIPT_PATH)
script_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script_tests)

for name, value in vars(script_tests).items():
    if name.startswith("test_"):
        globals()[name] = value


if __name__ == "__main__":
    script_tests.main()
