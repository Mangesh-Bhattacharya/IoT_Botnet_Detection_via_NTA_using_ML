"""
Test Setup Script
Verify that the IoT Botnet Detection project is properly set up
"""

import sys
from pathlib import Path
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('tensorflow', 'TensorFlow'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('yaml', 'PyYAML'),
        ('joblib', 'Joblib'),
    ]
    
    failed = []
    
    for package_name, display_name in required_packages:
        try:
            importlib.import_module(package_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - NOT FOUND")
            failed.append(display_name)
    
    if failed:
        print(f"\n⚠️  Missing packages: {', '.join(failed)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are installed!")
        return True


def test_project_structure():
    """Test if all required directories exist"""
    print("\nTesting project structure...")
    
    project_root = Path(__file__).parent
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'src/data_processing',
        'src/models',
        'src/utils',
        'src/visualization',
        'models/saved_models',
        'results/figures',
        'results/reports',
        'logs',
        'config',
        'notebooks',
    ]
    
    missing = []
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - NOT FOUND")
            missing.append(dir_path)
    
    if missing:
        print(f"\n⚠️  Missing directories: {', '.join(missing)}")
        print("Creating missing directories...")
        for dir_path in missing:
            (project_root / dir_path).mkdir(parents=True, exist_ok=True)
        print("✓ Directories created!")
        return True
    else:
        print("\n✓ All required directories exist!")
        return True


def test_custom_modules():
    """Test if custom modules can be imported"""
    print("\nTesting custom modules...")
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    custom_modules = [
        ('utils.config_loader', 'Configuration Loader'),
        ('utils.logger', 'Logger'),
        ('data_processing.data_loader', 'Data Loader'),
        ('data_processing.preprocessor', 'Preprocessor'),
        ('models.random_forest_model', 'Random Forest Model'),
        ('models.svm_model', 'SVM Model'),
        ('models.neural_network_model', 'Neural Network Model'),
        ('visualization.visualizer', 'Visualizer'),
    ]
    
    failed = []
    
    for module_name, display_name in custom_modules:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {display_name}")
        except Exception as e:
            print(f"  ✗ {display_name} - ERROR: {str(e)[:50]}")
            failed.append(display_name)
    
    if failed:
        print(f"\n⚠️  Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All custom modules imported successfully!")
        return True


def test_data_files():
    """Test if data files exist"""
    print("\nTesting data files...")
    
    project_root = Path(__file__).parent
    data_dir = project_root / 'data' / 'raw'
    
    if not data_dir.exists():
        print(f"  ⚠️  Data directory not found: {data_dir}")
        return False
    
    files = list(data_dir.glob('*.csv'))
    
    if files:
        print(f"  ✓ Found {len(files)} CSV file(s):")
        for file_path in files[:5]:  # Show first 5
            print(f"    - {file_path.name}")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more")
        return True
    else:
        print("  ⚠️  No CSV files found in data/raw/")
        print("  Please add your dataset files to the data/raw/ directory")
        return False


def test_configuration():
    """Test if configuration file exists and can be loaded"""
    print("\nTesting configuration...")
    
    project_root = Path(__file__).parent
    config_file = project_root / 'config' / 'config.yaml'
    
    if not config_file.exists():
        print(f"  ✗ Configuration file not found: {config_file}")
        return False
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  ✓ Configuration file loaded successfully")
        print(f"    - Data test size: {config.get('data', {}).get('test_size', 'N/A')}")
        print(f"    - Random state: {config.get('data', {}).get('random_state', 'N/A')}")
        return True
    except Exception as e:
        print(f"  ✗ Error loading configuration: {str(e)}")
        return False


def test_tensorflow():
    """Test TensorFlow installation and GPU availability"""
    print("\nTesting TensorFlow...")
    
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow version: {tf.__version__}")
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ GPU available: {len(gpus)} device(s)")
        else:
            print("  ℹ️  No GPU detected (will use CPU)")
        
        return True
    except Exception as e:
        print(f"  ✗ TensorFlow test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all setup tests"""
    print("="*60)
    print("IoT Botnet Detection - Setup Verification")
    print("Group 20: Rajat Jaswal, Mangesh Bhattacharya, Riya Kriplani")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("Custom Modules", test_custom_modules()))
    results.append(("Configuration", test_configuration()))
    results.append(("Data Files", test_data_files()))
    results.append(("TensorFlow", test_tensorflow()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("\n🎉 Setup verification complete! You're ready to start training.")
        print("\nNext steps:")
        print("1. Review QUICKSTART.md for usage instructions")
        print("2. Run: python main.py --mode train --models all --benign data/raw/benign_traffic.csv")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before proceeding.")
    
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
