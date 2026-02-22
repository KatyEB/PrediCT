import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_pre_yesterday_5pm(target_dir):
    target_path = Path(target_dir)
    
    # 1. Calculate Yesterday at 5:00 PM
    now = datetime.now()
    yesterday_5pm = (now - timedelta(days=1)).replace(hour=21, minute=30, second=0, microsecond=0)
    cutoff_ts = yesterday_5pm.timestamp()
    
    print(f"Target Directory: {target_path}")
    print(f"Cutoff Time: {yesterday_5pm.strftime('%Y-%m-%d %I:%M %p')}")
    print("-" * 40)

    files_deleted = 0
    folders_deleted = 0

    # 2. Walk through directory (topdown=False is critical for deleting folders)
    for root, dirs, files in os.walk(target_path, topdown=False):
        current_dir = Path(root)

        # Handle Files
        for name in files:
            file_path = current_dir / name
            if file_path.stat().st_mtime < cutoff_ts:
                try:
                    file_path.unlink()
                    files_deleted += 1
                except Exception as e:
                    print(f"  [ERROR] Could not delete file {name}: {e}")

        # Handle Folders
        for name in dirs:
            dir_path = current_dir / name
            
            # Check if folder is empty
            is_empty = not any(dir_path.iterdir())
            
            # Delete if it's empty OR if the folder itself is older than the cutoff
            if is_empty or dir_path.stat().st_mtime < cutoff_ts:
                try:
                    # We use rmtree if it's old, or rmdir if we only want to remove it if empty
                    if is_empty:
                        dir_path.rmdir()
                        folders_deleted += 1
                    elif dir_path.stat().st_mtime < cutoff_ts:
                        shutil.rmtree(dir_path)
                        folders_deleted += 1
                except Exception:
                    # Folder might be in use or already deleted by a previous step
                    pass

    print(f"\nCleanup complete!")
    print(f"Files removed: {files_deleted}")
    print(f"Folders removed: {folders_deleted}")

if __name__ == "__main__":
    IMAGES_DIR = r"C:\coca_project\data_canonical\images"
    
    if Path(IMAGES_DIR).exists():
        confirm = input(f"Delete files/folders in {IMAGES_DIR} modified BEFORE yesterday 5PM? (y/n): ")
        if confirm.lower() == 'y':
            cleanup_pre_yesterday_5pm(IMAGES_DIR)
        else:
            print("Operation cancelled.")
    else:
        print(f"Directory not found: {IMAGES_DIR}")