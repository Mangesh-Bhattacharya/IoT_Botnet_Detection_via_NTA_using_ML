"""
Data Preparation Script
Copy user-provided CSV files to the project structure and prepare for training
"""

import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_user_data():
    """Copy user's uploaded files to project data directory"""
    
    # Source directory (uploaded files)
    source_dir = Path("/mnt/user-data/uploads")
    
    # Destination directory
    dest_dir = Path("/home/claude/iot_botnet_detection/data/raw")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy benign traffic file
    benign_source = source_dir / "benign_traffic.csv"
    if benign_source.exists():
        benign_dest = dest_dir / "benign_traffic.csv"
        shutil.copy2(benign_source, benign_dest)
        logger.info(f"✓ Copied benign traffic data to {benign_dest}")
    else:
        logger.warning(f"Benign traffic file not found: {benign_source}")
    
    # Copy demonstrate structure file
    demo_source = source_dir / "demonstrate_structure.csv"
    if demo_source.exists():
        demo_dest = dest_dir / "demonstrate_structure.csv"
        shutil.copy2(demo_source, demo_dest)
        logger.info(f"✓ Copied demonstration file to {demo_dest}")
    else:
        logger.warning(f"Demonstration file not found: {demo_source}")
    
    # Copy project proposal
    proposal_source = source_dir / "Project-Proposal-Group-20.pdf"
    if proposal_source.exists():
        proposal_dest = dest_dir.parent.parent / "Project-Proposal-Group-20.pdf"
        shutil.copy2(proposal_source, proposal_dest)
        logger.info(f"✓ Copied project proposal to {proposal_dest}")
    
    # Copy dataset description
    desc_source = source_dir / "N_BaIoT_dataset_description_v1.txt"
    if desc_source.exists():
        desc_dest = dest_dir.parent.parent / "N_BaIoT_dataset_description.txt"
        shutil.copy2(desc_source, desc_dest)
        logger.info(f"✓ Copied dataset description to {desc_dest}")
    
    logger.info("\nData preparation completed!")
    logger.info(f"All data files are now in: {dest_dir}")


if __name__ == "__main__":
    prepare_user_data()
