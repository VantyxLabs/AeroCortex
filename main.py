import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config

def run_demo():
    from demo import main as demo_main
    demo_main()

def run_evaluate():
    from evaluate import run_evaluation
    run_evaluation()

def run_api():
    import uvicorn
    print(f"[AeroCortex API] Launching FastAPI on {config.api_host}:{config.api_port}...")
    uvicorn.run("api.telemetry_api:app", host=config.api_host, port=config.api_port, reload=False)

def run_dashboard():
    import uvicorn
    print(f"[AeroCortex Dashboard] Launching monochrome SPA on {config.api_host}:{config.dashboard_port}...")
    uvicorn.run(
        "dashboard.server:app",
        host=config.api_host,
        port=config.dashboard_port,
        reload=False,
    )

def main():
    parser = argparse.ArgumentParser(description="AeroCortex Cognitive Edge UAV Architecture")
    parser.add_argument("--demo", action="store_true", help="Run the one-command terminal demo")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation benchmarks against baseline")
    parser.add_argument("--api", action="store_true", help="Run FastAPI telemetry gateway")
    parser.add_argument("--dashboard", action="store_true", help="Run Minimalist Monochrome live dashboard")

    args = parser.parse_args()

    if args.api:
        run_api()
    elif args.dashboard:
        run_dashboard()
    elif args.evaluate:
        run_evaluate()
    elif args.demo:
        run_demo()
    else:
        # Default behavior if no flag passed
        print("Welcome to AeroCortex! Starting demo mode by default...")
        print("Use --api, --dashboard, --evaluate, or --demo for specific modes.\n")
        run_demo()

if __name__ == "__main__":
    main()
