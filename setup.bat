@echo off
REM Create a virtual environment in a folder named 'venv'
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate

REM Upgrade pip to the latest version
python -m pip install --upgrade pip

REM Install required dependencies
pip install pillow transformers requests torch

REM Freeze dependencies to a requirements.txt file
pip freeze > requirements.txt

echo Virtual environment setup is complete. To activate it, run:
echo .\venv\Scripts\Activate
