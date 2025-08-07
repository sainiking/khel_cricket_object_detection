#!/bin/bash
# Activation script for cricket detection project
# Run with: source activate_cricket_env.sh

echo "  Activating Cricket Detection Environment"
echo "=" * 50

# Check if virtual environment exists
if [ ! -d "cricket_env" ]; then
    echo "  Virtual environment not found!"
    echo "  Run: python3 -m venv cricket_env"
    echo "  Then: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source cricket_env/bin/activate

echo "  Virtual environment activated!"
echo "  Available commands:"
echo "   python3 check_setup.py              - Check project setup"
echo "   python3 check_separate_models.py    - Check model capabilities"
echo "   python3 train_separate_models.py    - Train separate models"
echo "   python3 simple_cricket_detector.py  - Run detection"
echo ""
echo " To deactivate: deactivate"
echo " For help: cat README.md"
