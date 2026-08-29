#!/usr/bin/env python3
"""
머신러닝 패키지 검증 스크립트
"""

import sys

def check_package(package_name, import_name=None):
    """
    패키지 설치가 되어 있고 import 가능한지 확인하는 함수
    :param package_name: 설치된 패키지 이름
    :param import_name: import 시 사용할 이름 (기본값: package_name)
    :return: 설치 여부 (True/False)
    """
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Unknown Version')
        print(f"✅ {package_name}: {version}.")
        return True
    except ImportError:
        print(f"❌ {package_name}: NOT installed.")
        return False

def main():
    print("=" * 50)
    print("Python ML Environment Verification")
    print("=" * 50)
    print(f"\nPython Version: {sys.version}")
    print(f"Python Path: {sys.path}\n")

    # Core ML packages
    packages = {
        ('PyTorch', 'torch'),
        ('TensorFlow', 'tensorflow'),
        ('Transformers', 'transformers'),
        ('NumPy', 'numpy'),
        ('Pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('FastAPI', 'fastapi'),
        ('Uvicorn', 'uvicorn'),
        ('Requests', 'requests'),
        ('HTTPX', 'httpx'),
        ('PyYAML', 'yaml'),
        ('python-dotenv', 'dotenv'),
        ('pytest', 'pytest'),
        ('Black', 'black'),
        ('Flke8', 'flake8')
    }

    results = []
    print("Checking packages...\n")
    for package, import_name in packages:
        results.append(check_package(package, import_name))

    # Summary
    print("\n" + "=" * 50)
    print(f"Summary: {sum(results)}/{len(results)} packages installed.")
    print("=" * 50)

    if all(results):
        print("\n✅ All required packages are installed successfully.")
        return 0
    else:
        print("\n❌ Some packages are missing. Please install the missing packages.")
        return 1

if __name__ == "__main__":
    sys.exit(main())