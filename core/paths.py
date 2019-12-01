import pkg_resources

_package_name = __name__

# 做文件路径的辅助函数
def get_file_path(*paths):
    path = "/".join(paths)
    return pkg_resources.resource_filename(_package_name, path)
