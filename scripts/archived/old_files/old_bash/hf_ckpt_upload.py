# 在脚本顶部添加导入
from huggingface_hub import upload_folder, create_repo, HfApi
import os
import sys
import logging # 引入日志记录

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 基础配置: 定义基础路径组件 ---
scratch_env_var = 'SCRATCH'  # 环境变量名称
base_subdir = 'work_dirs_auto_eval' # SCRATCH 内的子目录
hf_username = "Journey9ni" # *** 你的 Hugging Face 用户名 ***

# --- 用户配置区域 ---
# 定义一个要上传的文件夹列表。
# 列表中每个元素都是一个字典，包含:
#   'local_name': 本地文件夹的名称 (相对于 base_subdir) - 必须
#   'hf_name'   : Hugging Face 仓库的名称 (在你的用户名下) - 可选
#                 如果省略或为 None，将使用 'local_name' 作为 HF 仓库名。
#                 *** 请确保 'hf_name' 符合 HF 命名规则 (长度 <= 96, 允许字符等) ***

folders_to_upload = [
    {
        'local_name': "llava_video_7b_qwen2_05_15_lora_cut3r_all_tokens_cross_attn_add_abs_dist_route_plan_filtered",
        'hf_name': "llava_video_7b_qwen2_05_15_lora_cut3r_all_tokens_cross_attn_add_abs_dist_route_plan_filtered" # 示例：重映射为较短的名称
    },
    {
        'local_name': "llava_video_7b_qwen2_05_15_lora_base_add_abs_dist_route_plan_filtered",
        'hf_name': "llava_video_7b_qwen2_05_15_lora_base_add_abs_dist_route_plan_filtered" # 示例：重映射为较短的名称
    },
    # {
    #     'local_name': "folder_three",
    #     'hf_name': "repo-three-custom-name"
    # },
    # 添加更多需要上传的文件夹...
]

# ------------------------------------

# --- Hugging Face 通用配置 ---
repo_type = "model" # 或 'dataset', 'space'

# --- 通用忽略模式 ---
# 这些模式相对于每个 'local_name' 指定的文件夹
ignore_patterns = [
    "checkpoint-*/global_step*",
    "checkpoint-*/rng_state_*.pth",
    # "checkpoint-*/optimizer.pt",
    # "checkpoint-*/trainer_state.json",
    # "*.log",
    # "*.pyc",
    # "__pycache__/",
]
# ------------------------------------

# --- 获取绝对基础目录路径 ---
scratch_path = os.getenv(scratch_env_var)
if scratch_path is None:
    logging.error(f"错误: 环境变量 ${scratch_env_var} 未设置。")
    logging.error("请确保在您的环境中定义了 $SCRATCH。")
    sys.exit(1)

# 使用 os.path.join 构造完整的基础目录路径 (正确处理分隔符)
base_dir = os.path.join(scratch_path, base_subdir)
logging.info(f"基础目录: {base_dir} (来自 ${scratch_env_var})")

# --- 循环处理每个要上传的文件夹 ---
success_count = 0
failure_count = 0

logging.info(f"准备处理 {len(folders_to_upload)} 个文件夹...")

for item in folders_to_upload:
    local_relative_name = item.get('local_name')
    hf_repo_suffix = item.get('hf_name') # 获取 HF 仓库名后缀

    if not local_relative_name:
        logging.warning("跳过列表中的无效条目（缺少 'local_name'）: %s", item)
        failure_count += 1
        continue

    # 如果未提供 hf_name，则使用 local_name
    if hf_repo_suffix is None:
        hf_repo_suffix = local_relative_name
        logging.info(f"未提供 'hf_name' for '{local_relative_name}', 将使用本地名称作为仓库名。")

    # 构造当前文件夹的完整路径
    local_folder_to_upload = os.path.join(base_dir, local_relative_name)
    # 构造目标 HF 仓库 ID
    repo_id = f"{hf_username}/{hf_repo_suffix}"
    # 提交消息
    commit_message = f"Upload '{local_relative_name}' to '{repo_id}', excluding patterns"

    print("-" * 60)
    logging.info(f"开始处理: '{local_relative_name}'")
    logging.info(f"  本地源路径 : {local_folder_to_upload}")
    logging.info(f"  目标 HF 仓库: {repo_id} (类型: {repo_type})")
    logging.info(f"  忽略模式   : {ignore_patterns}")
    print("-" * 60)

    # --- 确保本地上传目录存在 ---
    if not os.path.isdir(local_folder_to_upload):
        logging.error(f"错误: 源文件夹不存在 '{local_folder_to_upload}'")
        logging.error(f"请检查基础目录 '{base_dir}' 和相对文件夹 '{local_relative_name}' 是否正确且可访问。")
        failure_count += 1
        continue # 继续处理列表中的下一个

    # --- 确保仓库存在 (或创建它) ---
    try:
        logging.info(f"确保仓库 '{repo_id}' 存在...")
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            exist_ok=True # 重要：如果仓库已存在，不执行任何操作且不引发错误
        )
        logging.info(f"仓库 '{repo_id}' 已确认或创建。")
    except Exception as e:
        logging.error(f"\n错误: 创建或访问仓库 '{repo_id}' 时出错: {e}")
        logging.error("请检查您的 Hugging Face token/登录状态和权限。")
        logging.error(f"同时请确认仓库名 '{hf_repo_suffix}' 是否符合 Huggingface 的命名规则 (长度 <= 96, 允许的字符等)。")
        failure_count += 1
        continue # 继续处理列表中的下一个

    # --- 执行上传 ---
    try:
        logging.info("开始上传过程...")
        upload_folder(
            folder_path=local_folder_to_upload, # 使用构造的绝对路径
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
            ignore_patterns=ignore_patterns,
            # token=os.getenv("HF_TOKEN") # 如果需要显式传递 token，取消此行注释
        )
        logging.info("---------------------")
        logging.info("   上传成功!    ")
        logging.info("---------------------")
        logging.info(f"来自 '{local_folder_to_upload}' 的内容 (忽略文件除外) 现已在仓库 '{repo_id}' 中")
        success_count += 1

    except Exception as e:
        logging.error("-------------------")
        logging.error("   上传失败!    ")
        logging.error("-------------------")
        logging.error(f"错误详情: {e}")
        logging.error("\n故障排除提示:")
        logging.error(" - Hugging Face token 是否有效 ('huggingface-cli login')?")
        logging.error(f"- 是否有仓库 '{repo_id}' 的写入权限?")
        logging.error(f"- 本地路径 '{local_folder_to_upload}' 是否包含 LFS 未处理的过大文件?")
        logging.error(" - 检查网络连接。")
        failure_count += 1
        continue # 继续处理列表中的下一个

# --- 结束报告 ---
print("\n" + "=" * 60)
logging.info("所有任务处理完毕。")
logging.info(f"成功上传: {success_count}")
logging.info(f"失败/跳过: {failure_count}")
print("=" * 60)

if failure_count > 0:
    sys.exit(1) # 如果有失败，以错误码退出
else:
    sys.exit(0) # 全部成功，正常退出