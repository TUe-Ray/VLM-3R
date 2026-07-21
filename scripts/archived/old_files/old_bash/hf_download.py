from huggingface_hub import snapshot_download
import os

# --- 配置参数 ---
# 从 hf_upload.py 中获取或根据需要修改
repo_id = "Journey9ni/vlm_3r_data"           # Hugging Face 仓库 ID
repo_type = "dataset"                       # 仓库类型
path_in_repo = "ARkitScenes"       # 仓库中要下载的文件夹路径
local_download_dir = "data/" # 本地保存路径
commit_hash = None                          # 可选：指定要下载的 commit hash (e.g., "main", "v1.0", specific hash)

# --- 确保本地下载目录存在 ---
os.makedirs(local_download_dir, exist_ok=True)
print(f"准备从仓库 '{repo_id}' 的 '{path_in_repo}' 下载到本地 '{local_download_dir}'...")

# --- 执行下载 ---
try:
    # snapshot_download 会下载指定仓库或其子目录的内容
    # allow_patterns 用于指定只下载特定模式的文件/文件夹
    # local_dir_use_symlinks=False 建议在 Windows 或需要明确副本时使用
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=f"{path_in_repo}/*", # 模式匹配 path_in_repo 下的所有内容
        local_dir=local_download_dir,
        local_dir_use_symlinks=False, # 建议设为 False 以避免 Windows 上的符号链接问题
        revision=commit_hash,             # 指定要下载的版本，默认为 main
        # ignore_patterns=["*.log", ".git*"], # 可选：忽略不想下载的文件模式
    )
    print("下载成功！")
    print(f"数据已下载到: {os.path.abspath(local_download_dir)}")

except Exception as e:
    print(f"下载失败：{e}")
    print("请检查：")
    print("- 是否已通过 'huggingface-cli login' 登录（如果仓库是私有的）？")
    print(f"- 仓库 '{repo_id}' 或路径 '{path_in_repo}' 是否存在？")
    print(f"- commit hash '{commit_hash}' 是否有效（如果指定了）？")
    print(f"- 网络连接是否正常？")
    print(f"- 是否有足够的磁盘空间？")
