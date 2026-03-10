"""
Create small test cache with only 2 files to verify 512 fix works
"""

import sys
import os
import shutil
sys.path.insert(0, r'C:\Users\Huawei\Desktop\cevahir_sinir_sistemi')

# Temporarily move most files
education_dir = r'C:\Users\Huawei\Desktop\cevahir_sinir_sistemi\education'
temp_dir = r'C:\Users\Huawei\Desktop\cevahir_sinir_sistemi\education_temp'

print("Creating test cache with 2 files only...")
print("Moving files temporarily...")

# Create temp dir
os.makedirs(temp_dir, exist_ok=True)

# Move all except 2 files
keep_files = ['felsefi_sohbet.json', 'gunluk-konusma.json']
moved_count = 0

for file in os.listdir(education_dir):
    if file.endswith('.json') and file not in keep_files:
        src = os.path.join(education_dir, file)
        dst = os.path.join(temp_dir, file)
        shutil.move(src, dst)
        moved_count += 1

print(f"Moved {moved_count} files to temp")
print(f"Keeping: {keep_files}")

# Now run prepare_cache
print("\nRunning prepare_cache with 2 files...")
from training_system.prepare_cache import prepare_cache

try:
    prepare_cache(
        data_dir="education",
        cache_dir=".cache/preprocessed_data_test",
        max_seq_length=512,
        clear_old_cache=True
    )
    print("\n[OK] Test cache created!")
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

# Move files back
print(f"\nMoving {moved_count} files back...")
for file in os.listdir(temp_dir):
    src = os.path.join(temp_dir, file)
    dst = os.path.join(education_dir, file)
    shutil.move(src, dst)

# Remove temp dir
os.rmdir(temp_dir)
print("[OK] Files restored")
