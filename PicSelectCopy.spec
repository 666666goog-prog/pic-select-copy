# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：onedir 便携文件夹，无控制台窗口。

RapidOCR 的 onnx 模型和配置文件是包数据，必须显式收集；
onnxruntime 的动态库也要收集，否则真机上会缺文件打不开。
"""
from PyInstaller.utils.hooks import (collect_data_files,
                                     collect_dynamic_libs,
                                     collect_submodules)

datas = []
datas += collect_data_files('rapidocr_onnxruntime')
datas += collect_data_files('onnxruntime')

binaries = []
binaries += collect_dynamic_libs('onnxruntime')

hiddenimports = []
hiddenimports += collect_submodules('rapidocr_onnxruntime')
hiddenimports += ['onnxruntime']

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'pytest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PicSelectCopy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PicSelectCopy',
)
