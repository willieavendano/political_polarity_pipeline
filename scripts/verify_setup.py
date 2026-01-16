"""
Verify that the setup is correct and all dependencies are installed.
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        logger.info(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        logger.error(f"✗ Python 3.11+ required, found {version.major}.{version.minor}")
        return False


def check_dependencies():
    """Check required dependencies."""
    required_packages = [
        "pandas",
        "numpy",
        "sklearn",
        "torch",
        "transformers",
        "datasets",
        "matplotlib",
        "seaborn",
        "yaml",
        "tqdm",
        "jsonlines",
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            if package == "sklearn":
                __import__("sklearn")
            elif package == "yaml":
                __import__("yaml")
            else:
                __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.error(f"✗ {package} not found")
            missing.append(package)
    
    return len(missing) == 0, missing


def check_directories():
    """Check that required directories exist or can be created."""
    import os
    
    required_dirs = [
        "./data/raw",
        "./data/processed",
        "./artifacts",
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(directory):
            logger.info(f"✓ Directory: {directory}")
        else:
            logger.error(f"✗ Could not create directory: {directory}")
            return False
    
    return True


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            logger.info("ℹ No GPU available, will use CPU (slower)")
            return True
    except Exception as e:
        logger.warning(f"Could not check GPU: {e}")
        return True


def main():
    """Run all verification checks."""
    logger.info("=" * 60)
    logger.info("Political Polarity Pipeline - Setup Verification")
    logger.info("=" * 60)
    
    checks = []
    
    logger.info("\n[1/4] Checking Python version...")
    checks.append(check_python_version())
    
    logger.info("\n[2/4] Checking dependencies...")
    deps_ok, missing = check_dependencies()
    checks.append(deps_ok)
    
    logger.info("\n[3/4] Checking directories...")
    checks.append(check_directories())
    
    logger.info("\n[4/4] Checking GPU availability...")
    check_gpu()
    
    logger.info("\n" + "=" * 60)
    if all(checks):
        logger.info("✓ All checks passed! Setup is complete.")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("✗ Some checks failed. Please fix the issues above.")
        if not deps_ok:
            logger.error(f"Missing packages: {', '.join(missing)}")
            logger.error("Install with: pip install -r requirements.txt")
        logger.info("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
