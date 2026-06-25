"""AgentX - Autonomous Agent System"""

import argparse
import uvicorn
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Carrega configuração do arquivo YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="AgentX Autonomous Agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Auto-reload")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    print("""
    ============================================
     AgentX v1.0 - Autonomous Agent
     Hardware: 8GB VRAM Optimized
     ReAct Loop + Tool Use + Memory
    ============================================
    """)
    
    print(f"Config: {args.config}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"UI: http://{args.host}:{args.port}/")
    print("\nStarting API server...\n")
    
    # Inicia servidor FastAPI
    uvicorn.run(
        "api.routes:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
