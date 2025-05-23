# https://gemini.google.com/app/47f5037b12396573
import asyncio, json, os, re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import SecretStr
from browser_use import Agent
from PyPDF2 import PdfReader

# --- 配置 ---
load_dotenv()

# DeepSeek API Key
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
if not DEEPSEEK_API_KEY:
    raise ValueError('DEEPSEEK_API_KEY 未在 .env 文件中设置！')

# PDF 存储目录
STORAGE_DIR = './storage/en'
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)
    print(f"创建了存储目录: {STORAGE_DIR}")

FLASK_SERVER_URL = "http://localhost:8848"  # Flask 后端服务地址
TARGET_WEB_FORM_URL = f"{FLASK_SERVER_URL}/"  # 表单页面URL

# --- PDF 信息提取函数 (使用 LLM) ---
async def extract_info_from_pdf(pdf_path: str, llm_model: ChatOpenAI) -> dict:
    """
    利用 LLM 从 PDF 文本中提取结构化信息，例如标题、作者、摘要、日期等。
    """
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""

        if not full_text.strip():
            print(f"警告: PDF '{pdf_path}' 未能提取到有效文本。")
            return {
                'pdf_path': pdf_path,
                'title': os.path.basename(pdf_path).replace('.pdf', '') + " (无文本)",
                'authors': '无文本',
                'date': '无文本',
                'abstract': '无文本',
                'introduction': '无文本',
                'funding': '无文本',
                'conclusion': '无文本',
                'affiliation': '无文本'
            }

        # 限制输入文本长度，避免超出LLM的上下文窗口
        max_text_length = 15000  # 根据LLM模型限制调整，deepseek-chat可能是16k tokens
        truncated_text = full_text[:max_text_length]

        # 构建 Prompt
        messages = [
            SystemMessage(
                content="你是一个高级的信息提取助手。请根据提供的论文文本，准确提取以下信息，并以 JSON 格式返回。如果某个字段无法提取，请使用 'N/A'。"),
            HumanMessage(content=f"""
                请从以下论文文本中提取以下关键信息：
                - **标题 (title)**: 论文的完整标题。
                - **作者 (authors)**: 所有作者的姓名，用逗号分隔。
                - **机构 (affiliation)**: 作者所在的机构或单位。
                - **日期 (date)**: 论文的发表日期或版本日期。
                - **摘要 (abstract)**: 论文的摘要部分。
                - **引言 (introduction)**: 论文的引言部分。
                - **资助 (funding)**: 论文中提到的资金来源或致谢部分。
                - **结论 (conclusion)**: 论文的结论部分。

                请严格按照以下 JSON 格式返回结果：
                ```json
                {{
                    "title": "...",
                    "authors": "...",
                    "affiliation": "...",
                    "date": "...",
                    "abstract": "...",
                    "introduction": "...",
                    "funding": "...",
                    "conclusion": "..."
                }}
                ```

                论文文本：
                {truncated_text}
                """)
        ]

        # 核心修改在这里：将 .invoke() 改为 .ainvoke()
        response = await llm_model.ainvoke(messages)
        extracted_json_str = response.content

        # 从可能的Markdown代码块中提取JSON
        json_match = re.search(r'```json\n([\s\S]+?)\n```', extracted_json_str)
        if json_match:
            extracted_json_str = json_match.group(1)

        # 解析 JSON 字符串
        extracted_data = json.loads(extracted_json_str)

        # 将提取的数据与原始路径合并
        extracted_data['pdf_path'] = pdf_path
        print(f"LLM 从 {os.path.basename(pdf_path)} 提取信息: 标题='{extracted_data.get('title', 'N/A')}'")
        return extracted_data

    except Exception as e:
        print(f"使用 LLM 从 {pdf_path} 提取信息时出错: {e}")
        # 确保返回一个包含所有字段的字典，即使是错误状态
        return {
            'pdf_path': pdf_path,
            'title': os.path.basename(pdf_path).replace('.pdf', '') + " (LLM提取错误)",
            'authors': 'LLM提取错误',
            'date': 'LLM提取错误',
            'abstract': 'LLM提取错误',
            'introduction': 'LLM提取错误',
            'funding': 'LLM提取错误',
            'conclusion': 'LLM提取错误',
            'affiliation': 'LLM提取错误'
        }

# --- 智能体任务生成函数 (保持不变) ---
def generate_web_form_task(paper_info: dict, form_url: str) -> str:
    """
    生成智能体填充网页表单的任务字符串。
    你需要根据 form.html 中各个输入字段的 name 属性来定制这些指令。
    """

    def escape_for_task(text):
        if not text:
            return ""
        # 简单转义双引号，避免与任务字符串的引号冲突
        # 移除换行符，因为 HTML input/textarea 可能不直接支持多行或需要特定处理
        return text.replace('"', '\\"').replace('\n', ' ').replace('\r', '')

    title = escape_for_task(paper_info['title'])
    authors = escape_for_task(paper_info['authors'])
    affiliation = escape_for_task(paper_info['affiliation'])
    date = escape_for_task(paper_info['date'])
    abstract = escape_for_task(paper_info['abstract'])
    introduction = escape_for_task(paper_info['introduction'])
    funding = escape_for_task(paper_info['funding'])
    conclusion = escape_for_task(paper_info['conclusion'])

    # IMPORTANT: 替换这些选择器以匹配你的 form.html 中的实际 input/textarea 元素的 name 属性！
    # 例如：<input type="text" name="title"> 对应的指令就是 'name "title"'
    task = f"""
        1. Go to {form_url}
        2. Fill the input field with name "title" with the value "{title}"
        3. Fill the input field with name "authors" with the value "{authors}"
        4. Fill the input field with name "affiliation" with the value "{affiliation}"
        5. Fill the input field with name "date" with the value "{date}"
        6. Fill the textarea with name "abstract" with the value "{abstract}"
        7. Fill the textarea with name "introduction" with the value "{introduction}"
        8. Fill the textarea with name "funding" with the value "{funding}"
        9. Fill the textarea with name "conclusion" with the value "{conclusion}"
        10. Click the submit button with name "submit" or type "submit".
        11. After submission, wait for a message indicating success or failure. If a message containing "数据已保存" or "ID:" appears, return "提交成功". Otherwise, return "提交失败".
        12. Return the URL of the page after submission (which should still be {form_url} if your Flask app redirects, or the API response URL if it changes).
    """
    return task.strip()

# --- 单个 PDF 文件的处理流程 (封装成异步函数) ---
async def process_single_pdf(pdf_path: str, llm_model: ChatOpenAI, target_form_url: str):
    """
    处理单个 PDF 文件的完整流程：LLM 提取信息 -> Agent 提交表单。
    """
    pdf_file_name = os.path.basename(pdf_path)
    print(f"\n--- 正在开始处理文件: {pdf_file_name} ---")

    # 1. 使用 LLM 从 PDF 中提取信息
    paper_info = await extract_info_from_pdf(pdf_path, llm_model)

    # 2. 生成智能体任务，填充网页表单
    agent_task = generate_web_form_task(paper_info, target_form_url)
    # print(f"生成的智能体任务预览 ({pdf_file_name}):\n", agent_task[:500], "...") # 打印任务前500字符进行调试

    # 3. 初始化并运行智能体
    try:
        agent = Agent(
            task=agent_task,
            llm=llm_model,
            use_vision=True, # 启用视觉功能，帮助智能体更好地定位网页元素
            # verbose=True # 可以打开这个选项来查看智能体的详细执行过程
        )
        print(f"启动浏览器智能体为 '{pdf_file_name}' 提交表单...")
        agent_result = await agent.run()
        print(f"智能体为 '{pdf_file_name}' 执行完毕。结果: {agent_result}")

        # 根据智能体返回的结果判断提交状态
        if "提交成功" in agent_result:
            print(f"论文 '{paper_info.get('title', '未知标题')}' 提交到 Flask 后端成功！")
        else:
            print(f"论文 '{paper_info.get('title', '未知标题')}' 提交失败。智能体返回结果: {agent_result}")

    except Exception as e:
        print(f"智能体在处理 '{pdf_file_name}' 时发生错误: {e}")

    print(f"--- 完成处理文件: {pdf_file_name} ---")
    # 可以选择在这里添加一个短暂的延迟，避免过于频繁地启动浏览器或请求API
    await asyncio.sleep(1) # 1秒延迟

# --- 主自动化逻辑 (并发处理) ---
async def process_all_paper_files_concurrently():
    llm_model = ChatOpenAI(
        base_url='https://api.deepseek.com/v1',
        model='deepseek-chat', # 推荐 deepseek-chat 或 deepseek-reasoner
        api_key=SecretStr(DEEPSEEK_API_KEY),
    )

    pdf_files = [f for f in os.listdir(STORAGE_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"在 '{STORAGE_DIR}' 中没有找到 PDF 文件。请将你的 PDF 文件放在该目录中。")
        return

    # 创建所有 PDF 文件的处理任务
    tasks = []
    for pdf_file_name in pdf_files:
        pdf_path = os.path.join(STORAGE_DIR, pdf_file_name)
        # 为每个 PDF 创建一个独立的协程任务
        tasks.append(process_single_pdf(pdf_path, llm_model, TARGET_WEB_FORM_URL))

    # 使用 asyncio.gather 并发运行所有任务
    print(f"准备并发处理 {len(tasks)} 个 PDF 文件...")
    await asyncio.gather(*tasks)
    print(f"所有 {len(tasks)} 个 PDF 文件处理完毕。")

# --- 主程序入口 ---
if __name__ == '__main__':
    # 确保 Flask 后端服务 (server.py) 已经启动并在运行
    print(f"请确保你的 Flask 服务器在 {FLASK_SERVER_URL} 运行。")
    print("你可以通过在另一个终端运行 'python server.py' 来启动它。")
    print("开始并发处理 PDF 文件...")
    asyncio.run(process_all_paper_files_concurrently())