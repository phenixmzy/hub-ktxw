import platform
import sys
import pkg_resources
def get_installed_packages():
    installed_packages = {p.key : p.version for p in pkg_resources.working_set}
    return installed_packages

def print_env_info():
    print("python env info")
    print(f"os: {platform.system()} - {platform.release()} - {platform.machine()}")
    print(f"python version: {sys.version}")
    print(f"curr path: {sys.path[0]}")
    print("-"*50)

    print("\n--------已安装的常用库-------------")
    required_packages = {
        "jieba": "jieba",
        "scikit-learn": "scikit-learn",
        "pytorch": "torch",
        "numpy": "numpy",
        "pandas": "pandas",
        "requests": "requests",
        "matplotlib": "matplotlib"
    }
    installed_packages = get_installed_packages()

    for name, key in required_packages.items():
        # print(f"name: {name}, key: {key}")
        if key in installed_packages:
            print(f"{key:<20} 已安装")
        else:
            print(f"{key:<20}: 未安装")

if __name__ == "__main__":
    print_env_info()