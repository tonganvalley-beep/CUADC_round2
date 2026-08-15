@echo off
cd /d "%~dp0"
"D:\Anaconda\envs\yolo\python.exe" review_send_to_ground_digits.py
if errorlevel 1 pause
