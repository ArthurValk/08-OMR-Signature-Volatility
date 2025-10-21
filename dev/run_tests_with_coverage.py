"""Run pytest with coverage and open HTML report in browser."""

import subprocess
import sys
import webbrowser
from pathlib import Path


def main():
    """Run pytest with coverage and open the HTML report.

    To run this, execute:
        python -m dev.run_tests_with_coverage
    """
    # Get the project root directory (parent of dev/)
    project_root = Path(__file__).parent.parent

    # Run pytest with coverage
    print("Running pytest with coverage...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=signature_vol",
            "--cov-report=html",
        ],
        cwd=project_root,
    )

    # Check if tests passed
    if result.returncode != 0:
        print("\n⚠️  Tests failed or had errors.")
        sys.exit(result.returncode)

    # Open the HTML coverage report
    html_report = project_root / "htmlcov" / "index.html"
    if html_report.exists():
        print(f"\n✓ Opening coverage report: {html_report}")
        webbrowser.open(html_report.as_uri())
    else:
        print(f"\n⚠️  Coverage report not found at {html_report}")
        sys.exit(1)


if __name__ == "__main__":
    main()
