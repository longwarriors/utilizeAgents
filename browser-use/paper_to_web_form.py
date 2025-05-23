# https://grok.com/chat/7c6c7c4c-9e17-43a1-916e-cc5aee4c9cfd
import asyncio
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import pdfplumber
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from browser_use import Agent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_api_key():
    """加载API密钥"""
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY', '')
    if not api_key:
        raise ValueError('DEEPSEEK_API_KEY is not set')
    return api_key


def serialize_result(result):
    """将结果转换为可JSON序列化的格式"""
    if isinstance(result, dict):
        return {k: serialize_result(v) for k, v in result.items()}
    elif isinstance(result, list):
        return [serialize_result(item) for item in result]
    elif isinstance(result, (str, int, float, bool, type(None))):
        return result
    else:
        return str(result)


def save_results(data, filename_prefix='result'):
    """保存结果到JSON文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{filename_prefix}_{timestamp}.json'
    try:
        serialized_data = serialize_result(data)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, ensure_ascii=False, indent=2)
        logger.info(f'Results saved to {filename}')
    except Exception as e:
        logger.error(f'Failed to save results: {str(e)}')
        raise


def extract_paper_elements(pdf_path, llm):
    """从PDF论文中提取关键元素"""
    try:
        elements = {
            'title': '',
            'authors': '',
            'affiliation': '',
            'date': '',
            'abstract': '',
            'introduction': '',
            'funding': '',
            'conclusion': ''
        }

        # 提取PDF文本
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages[:5]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
            if not text.strip():
                logger.warning(f'No text extracted from {pdf_path}. PDF may be scanned.')
                return elements

        # 限制文本长度，避免token超限
        text = text[:5000]  # 减到5000字符，确保prompt不过长

        # 优化后的中文 prompt
        prompt = f"""
        你是一个专业的学术论文分析助手。请从以下论文文本中提取以下关键元素，并确保输出为有效的 JSON 格式：
        - 标题：论文的完整标题，通常位于文档开头或第一页顶部。
        - 作者：所有作者的姓名（用逗号分隔），通常在标题下方。
        - 单位：作者的机构隶属（如大学、研究所），通常在作者下方或脚注中。
        - 日期：出版、提交或会议日期（如“2023年5月”或“2023-05”），可能出现在标题下方或元数据中。
        - 摘要：以“摘要”或“Abstract”开头的段落，通常在标题和正文之间。
        - 引言：引言部分的第一段，通常以“引言”或“Introduction”开头；如果没有明确标签，提取正文第一段。
        - 资助：资助信息，通常包含“基金”“资助”“支持”“acknowledgment”或“grant”等关键词，可能在致谢、脚注或文章末尾。
        - 结论：以“结论”“总结”或“Conclusion”开头的最后部分；如果没有明确标签，提取最后一段或最后几段的总结性内容。

        **要求**：
        1. 输出必须是有效的 JSON 格式，键名与上述元素一致。
        2. 如果某个元素缺失，返回空字符串 ""。
        3. 如果文本过长，仅处理提供的内容。
        4. 确保提取的文本干净，无多余的换行符或乱码。
        5. 如果元素内容超过500字符，截断并保留前500字符。
        6. 对于资助，确保提取完整的资助信息，包括基金名称和编号（如“National Science Foundation NSF-123456”）。
        7. 如果无法提取任何元素，返回空的 JSON 对象 {{}}。
        8. 用 ```json 标记 JSON 输出，例如：
           ```json
           {{}}
           ```

        **示例**：
        输入文本：
        “Title: Advances in Deep Learning
        Authors: John Doe, Jane Smith
        Affiliation: MIT
        Date: May 2023
        Abstract: This paper explores...
        Introduction: In recent years...
        Acknowledgment: Supported by NSF-123456
        Conclusion: We find that...”

        输出：
        ```json
        {{
          "title": "Advances in Deep Learning",
          "authors": "John Doe, Jane Smith",
          "affiliation": "MIT",
          "date": "May 2023",
          "abstract": "This paper explores...",
          "introduction": "In recent years...",
          "funding": "Supported by NSF-123456",
          "conclusion": "We find that..."
        }}
        ```

        **输入文本**：
        {text}

        **输出**：
        """

        # 调用 LLM 并记录响应
        response = llm.invoke(prompt)
        raw_content = response.content if hasattr(response, 'content') else ''
        logger.info(f'LLM raw response: {raw_content[:500]}...')  # 记录前500字符

        # 尝试解析 JSON
        try:
            # 提取 ```json 标记中的内容
            import re
            json_match = re.search(r'```json\n([\s\S]*?)\n```', raw_content)
            if json_match:
                json_str = json_match.group(1)
                extracted = json.loads(json_str)
            else:
                # 如果没有 ```json 标记，直接尝试解析
                extracted = json.loads(raw_content)
        except json.JSONDecodeError as e:
            logger.error(f'JSON parsing failed: {str(e)}. Raw content: {raw_content[:1000]}')
            # 尝试修复常见问题（如纯文本响应）
            if raw_content.strip():
                # 假设 LLM 返回纯文本，尝试手动构造 JSON
                extracted = {}
                for key in elements:
                    # 简单提取：查找关键词后的内容
                    if key.capitalize() in raw_content:
                        extracted[key] = raw_content.split(key.capitalize() + ':')[1].split('\n')[0][:500]
            else:
                extracted = {}

        # 更新元素
        elements.update({k: v[:500] if isinstance(v, str) else v for k, v in extracted.items()})
        logger.info(f'Successfully extracted elements from {pdf_path}: {elements}')
        return elements

    except Exception as e:
        logger.error(f'Failed to extract elements from {pdf_path}: {str(e)}')
        return elements  # 返回默认空元素


async def fill_web_form(elements, form_url, filename_prefix='paper_form'):
    """将论文元素填充到网页表单"""
    try:
        # 动态生成任务描述
        task = f"""
            1. Go to {form_url}
            2. Fill the form field with id 'title' with '{elements['title']}'
            3. Fill the form field with id 'authors' with '{elements['authors']}'
            4. Fill the form field with id 'affiliation' with '{elements['affiliation']}'
            5. Fill the form field with id 'date' with '{elements['date']}'
            6. Fill the form field with id 'abstract' with '{elements['abstract'][:500]}'
            7. Fill the form field with id 'introduction' with '{elements['introduction'][:500]}'
            8. Fill the form field with id 'funding' with '{elements['funding'][:500]}'
            9. Fill the form field with id 'conclusion' with '{elements['conclusion'][:500]}'
            10. Click the submit button
            11. Wait for the confirmation message '提交成功！数据已保存。' to appear
            12. Return the confirmation message or the page URL
        """

        # 初始化Agent
        api_key = load_api_key()
        agent = Agent(
            task=task,
            llm=ChatOpenAI(
                base_url='https://api.deepseek.com/v1',
                model='deepseek-chat',
                api_key=SecretStr(api_key),
            ),
            use_vision=False,
        )

        # 运行任务
        logger.info('Starting form filling task...')
        result = await agent.run()
        result = {'elements': elements, 'form_result': result}

        # 保存结果
        save_results(result, filename_prefix)
        logger.info('Form filling completed successfully')
        return result

    except Exception as e:
        logger.error(f'Failed to fill form: {str(e)}')
        raise


async def main():
    """主函数"""
    # pdf_file = './原子喷泉频标原理与发展.pdf'
    # pdf_file = './量子优化算法综述.pdf'
    # pdf_file = './写给物理学家的生成模型.pdf'
    # pdf_file = './基于辅助单比特测量的量子态读取算法.pdf'
    # pdf_file = './量子态制备及其在量子机器学习中的前景.pdf'
    pdf_file = './微分万物：深度学习的启示.pdf'  # 论文pdf文件路径
    form_url = 'http://localhost:8848/'  # 以后要替换为实际表单URL

    # 初始化LLM
    api_key = load_api_key()
    llm = ChatOpenAI(
        base_url='https://api.deepseek.com/v1',
        model='deepseek-chat',
        api_key=SecretStr(api_key),
    )

    # 提取论文元素
    elements = extract_paper_elements(pdf_file, llm)

    # 运行表单填充任务
    result = await fill_web_form(elements, form_url, 'paper_form')

    # 打印结果
    serialized_result = serialize_result(result)
    print('Form fill results:')
    print(json.dumps(serialized_result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
