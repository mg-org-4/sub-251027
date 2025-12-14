import os
import importlib

# 告诉 ComfyUI 服务器我们的 Web 目录
WEB_DIRECTORY = "./js"

# ComfyUI 节点映射字典
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 动态加载所有节点
# ======================================================================================================================
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
nodes_dir = os.path.join(current_dir, "nodes")

# 遍历 nodes 目录下的所有 python 文件
for filename in os.listdir(nodes_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = f".nodes.{filename[:-3]}"
        try:
            # 动态导入模块
            module = importlib.import_module(module_name, package=__name__)
            
            # 从模块中获取节点映射
            if hasattr(module, "NODE_CLASS_MAPPINGS") and hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                print(f"  - 成功加载模块: {module_name}")
            else:
                print(f"  - 警告: 模块 {module_name} 缺少必要的映射变量")

        except Exception as e:
            print(f"  - 错误: 加载模块 {module_name} 失败 - {e}")
# ======================================================================================================================

# 导出ComfyUI需要的变量
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("🎨 ComfyUI-Only 自定义节点已加载")
print(f"   - 前端扩展目录: {WEB_DIRECTORY}")
print(f"   📦 已注册 {len(NODE_CLASS_MAPPINGS)} 个节点:")
for name in NODE_CLASS_MAPPINGS.keys():
    display_name = NODE_DISPLAY_NAME_MAPPINGS.get(name, name)
    print(f"   - {name} ({display_name})")