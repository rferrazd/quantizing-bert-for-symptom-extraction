#!/usr/bin/env python3
"""Quick script to check PyTorch version and CUDA availability"""

import sys

try:
    import torch
    print("=" * 50)
    print("PyTorch Version Check")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("✗ CUDA is not available")
        print("  Running on CPU or MPS")
    
    # Check version requirement
    version_parts = torch.__version__.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1])
    
    if major > 2 or (major == 2 and minor >= 6):
        print(f"\n✓ PyTorch version meets requirement (>= 2.6.0)")
    else:
        print(f"\n⚠ Warning: PyTorch version {torch.__version__} is below 2.6.0")
        print("  Upgrade required to avoid torch.load security restrictions")
    
    print("=" * 50)
    
except ImportError:
    print("❌ PyTorch is not installed")
    sys.exit(1)

