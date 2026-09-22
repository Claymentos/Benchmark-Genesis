# debug.py
try:
    from sam3d_objects.model.backbone.generator.shortcut.model import ShortCut_taylorseer
    print("✅ 成功找到类！")
except ImportError as e:
    print(f"❌ 导入失败，原因: {e}")
except Exception as e:
    print(f"❌ 文件内部有错误: {e}")