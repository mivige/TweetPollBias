from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


class ElectionPaths:
    """Per-election output directory paths."""

    def __init__(self, election_subdir: str):
        self.raw_dir = RAW_DATA_DIR / election_subdir
        self.processed_dir = PROCESSED_DATA_DIR / election_subdir
        self.figures_dir = FIGURES_DIR / election_subdir
        self.reports_dir = REPORTS_DIR / election_subdir

    def ensure_dirs(self):
        """Create all election-specific directories if they don't exist."""
        for d in (self.processed_dir, self.figures_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_election_paths(election_code: str) -> ElectionPaths:
    """
    Return an ElectionPaths object with paths scoped to the given election.

    Args:
        election_code: Short identifier (e.g. "us20").

    Returns:
        ElectionPaths with processed_dir, figures_dir, reports_dir.
    """
    from bias_analysis.election_configs import get_election_config

    config = get_election_config(election_code)
    paths = ElectionPaths(config["processed_subdir"])
    paths.ensure_dirs()
    return paths


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
