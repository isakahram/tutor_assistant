@echo off
chcp 65001 >nul
title Расшифровка занятий
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Python не найден.
    echo Установите его с https://www.python.org/downloads/
    echo При установке обязательно отметьте "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

if not exist "venv\Scripts\pythonw.exe" (
    echo Первый запуск: создаю окружение и ставлю библиотеки.
    echo Это займёт несколько минут и потребует интернет.
    python -m venv venv || (echo Не удалось создать venv & pause & exit /b 1)
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt || (
        echo Не удалось установить зависимости & pause & exit /b 1
    )
)

start "" venv\Scripts\pythonw.exe main.py
exit
